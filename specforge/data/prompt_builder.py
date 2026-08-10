"""Build text prompt tasks for the built-in online training providers.

Pre-tokenized JSONL files stay on a standard-library-only fast path. Raw
conversation files load Hugging Face ``datasets`` and the existing Eagle3
preprocessor lazily so importing this module does not initialize either stack.
Other modalities own prompt preparation through ``ServerInputAdapter``.
"""

from __future__ import annotations

import json
import os
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Integral
from typing import Any

PromptTaskDict = dict[str, Any]


def prepare_prompt_tasks(
    path: str | os.PathLike[str],
    tokenizer: Any,
    *,
    chat_template: str | None,
    max_length: int,
    is_preformatted: bool,
    train_only_last_turn: bool,
    cache_dir: str | None,
    cache_key: str | None,
    num_proc: int | None,
    min_loss_tokens: int = 1,
    max_prompts: int | None = None,
) -> list[PromptTaskDict]:
    """Prepare runtime prompt dictionaries from a JSONL file.

    Each returned item has the control-plane shape
    ``{"payload": {"input_ids": [...], "loss_mask": [...]}}`` and contains no
    tensors. Files whose first record contains ``input_ids`` and ``loss_mask``
    are treated as pre-tokenized. Other files are treated as raw conversation
    data and processed through :func:`build_eagle3_dataset`.

    ``max_prompts`` caps accepted prompts; ``None`` and ``0`` mean no cap.
    """

    _validate_options(
        max_length=max_length,
        min_loss_tokens=min_loss_tokens,
        max_prompts=max_prompts,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    path_string = os.fspath(path)
    first_record = next(_iter_records(path_string), None)
    if first_record is None:
        return []

    _, record = first_record
    has_input_ids = "input_ids" in record
    has_loss_mask = "loss_mask" in record
    if has_input_ids != has_loss_mask:
        missing = "loss_mask" if has_input_ids else "input_ids"
        raise ValueError(
            f"pre-tokenized prompt record must contain both input_ids and "
            f"loss_mask; missing {missing} in {path_string!r}"
        )

    limit = None if max_prompts in (None, 0) else max_prompts
    if has_input_ids:
        rows = (
            (record, f"{path_string}:{line_number}")
            for line_number, record in _iter_records(path_string)
        )
        return _materialize_prompt_tasks(
            rows,
            max_length=max_length,
            min_loss_tokens=min_loss_tokens,
            limit=limit,
        )

    return _prepare_raw_prompts(
        path_string,
        tokenizer,
        chat_template=chat_template,
        max_length=max_length,
        is_preformatted=is_preformatted,
        train_only_last_turn=train_only_last_turn,
        cache_dir=cache_dir,
        cache_key=cache_key,
        num_proc=num_proc,
        min_loss_tokens=min_loss_tokens,
        limit=limit,
    )


def _prepare_raw_prompts(
    path: str,
    tokenizer: Any,
    *,
    chat_template: str | None,
    max_length: int,
    is_preformatted: bool,
    train_only_last_turn: bool,
    cache_dir: str | None,
    cache_key: str | None,
    num_proc: int | None,
    min_loss_tokens: int,
    limit: int | None,
) -> list[PromptTaskDict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - package dependency in production
        raise ImportError(
            "preparing raw conversation prompts requires the 'datasets' package"
        ) from exc

    from .preprocessing import build_eagle3_dataset

    dataset = load_dataset("json", data_files=path, split="train")
    if limit is not None and limit < len(dataset):
        dataset = dataset.select(range(limit))

    processed_dataset = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
        num_proc=num_proc,
        cache_dir=cache_dir,
        cache_key=cache_key,
        is_preformatted=is_preformatted,
        train_only_last_turn=train_only_last_turn,
        minimum_valid_tokens=min_loss_tokens,
    )
    # Python lists fail the hasattr(value, "tolist") test guarding the typed fast path in
    # _normalize_integer_sequence; numpy rows take it, 12.5x faster and byte-identical.
    processed_dataset = processed_dataset.with_format("numpy")
    rows = (
        (record, f"processed dataset row {index}")
        for index, record in enumerate(processed_dataset)
    )
    return _materialize_prompt_tasks(
        rows,
        max_length=max_length,
        min_loss_tokens=min_loss_tokens,
        limit=limit,
    )


def _materialize_prompt_tasks(
    rows: Iterable[tuple[Mapping[str, Any], str]],
    *,
    max_length: int,
    min_loss_tokens: int,
    limit: int | None,
) -> list[PromptTaskDict]:
    prompts: list[PromptTaskDict] = []
    for record, source in rows:
        if "input_ids" not in record or "loss_mask" not in record:
            raise ValueError(f"{source} must contain both input_ids and loss_mask")

        input_ids = _normalize_integer_sequence(
            record["input_ids"], field="input_ids", source=source, binary=False
        )
        loss_mask = _normalize_integer_sequence(
            record["loss_mask"], field="loss_mask", source=source, binary=True
        )
        if len(input_ids) != len(loss_mask):
            raise ValueError(
                f"{source} has mismatched input_ids/loss_mask lengths: "
                f"{len(input_ids)} != {len(loss_mask)}"
            )
        if not input_ids:
            raise ValueError(f"{source} contains an empty token sequence")

        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]
        if _count_supervised(loss_mask) < min_loss_tokens:
            continue

        prompts.append(
            {
                "payload": {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }
            }
        )
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def _normalize_integer_sequence(
    value: Any,
    *,
    field: str,
    source: str,
    binary: bool,
):
    # Returns an array, not a list: the producer holds every prompt at once to shuffle
    # per epoch, and a Python int costs ~36 bytes against an array cell's 4 (or 1).
    typed_source = hasattr(value, "tolist")
    if typed_source:
        packed = _pack_integer_column(value, field=field, source=source, binary=binary)
        if packed is not None:
            return packed
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{source} field {field} must be a sequence")

    sequence = list(value)
    if len(sequence) == 1 and _is_sequence(sequence[0]):
        # Rows arrive shaped [1, seq_len]; unwrapping keeps the same integer column, so
        # typed_source must survive here or the fast path below becomes unreachable.
        sequence = list(sequence[0])
    if typed_source:
        # An Arrow or numpy column is integer-typed already, so the per-item loop
        # below cannot reject anything; pack in C and skip 1.8M x N isinstance calls.
        try:
            packed = array("b" if binary else "i", sequence)
        except (TypeError, OverflowError, ValueError):
            packed = None
        if packed is not None:
            if binary and len(packed) and (min(packed) < 0 or max(packed) > 1):
                bad = next(v for v in packed if v not in (0, 1))
                raise ValueError(
                    f"{source} field {field} must be 0 or 1, got {bad}"
                )
            return packed
    if any(_is_sequence(item) for item in sequence):
        raise ValueError(f"{source} field {field} must be one-dimensional")

    normalized: list[int] = []
    for index, item in enumerate(sequence):
        is_invalid_token_id = field == "input_ids" and isinstance(item, bool)
        if not isinstance(item, Integral) or is_invalid_token_id:
            raise ValueError(
                f"{source} field {field}[{index}] must be an integer, got "
                f"{type(item).__name__}"
            )
        integer = int(item)
        if binary and integer not in (0, 1):
            raise ValueError(
                f"{source} field {field}[{index}] must be 0 or 1, got {integer}"
            )
        normalized.append(integer)
    return array("b" if binary else "i", normalized)


def _count_supervised(loss_mask: Any) -> int:
    """Count the supervised positions, reading the mask's buffer rather than its elements.

    Builtin sum() boxes every element: 1.81M rows x up to 4096 positions on the event
    corpus. numpy reads the same bytes with no copy.
    """
    if isinstance(loss_mask, array) and loss_mask.typecode == "b":
        try:
            import numpy as np

            return int(np.frombuffer(loss_mask, dtype=np.int8).sum())
        except (ImportError, TypeError, ValueError):
            pass
    return sum(loss_mask)


def _pack_integer_column(value: Any, *, field: str, source: str, binary: bool):
    """Pack an integer numpy column into array.array, or return None if it is not one.

    tolist() on a 4096-token row builds 4096 Python ints only for array() to unbox them
    again. Measured on the 1.81M-row event corpus, that round trip was 674 s of the
    producer's 744 s startup; going through the buffer skips it entirely.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy arrives with datasets
        return None

    column = np.asarray(value)
    if column.ndim == 2 and column.shape[0] == 1:
        column = column[0]
    if column.ndim != 1 or not np.issubdtype(column.dtype, np.integer):
        return None

    if binary:
        if column.size and (column.min() < 0 or column.max() > 1):
            bad = int(column[(column != 0) & (column != 1)][0])
            raise ValueError(f"{source} field {field} must be 0 or 1, got {bad}")
        typecode, dtype = "b", np.int8
    else:
        limits = np.iinfo(np.int32)
        if column.size and (column.min() < limits.min or column.max() > limits.max):
            return None
        typecode, dtype = "i", np.int32

    packed = array(typecode)
    # frombytes trusts the element width; a platform where they disagree takes the slow path.
    if packed.itemsize != np.dtype(dtype).itemsize:
        return None
    packed.frombytes(np.ascontiguousarray(column, dtype=dtype).tobytes())
    return packed


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _iter_records(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield objects from a JSON array or newline-delimited JSON file."""
    with open(path, encoding="utf-8") as stream:
        first_character = ""
        while True:
            character = stream.read(1)
            if not character or not character.isspace():
                first_character = character
                break
        stream.seek(0)
        if first_character == "[":
            try:
                records = json.load(stream)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON array in {path!r}") from exc
            if not isinstance(records, list):
                raise ValueError(f"JSON data in {path!r} must be an array of objects")
            for index, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSON record {index} in {path!r} must be an object"
                    )
                yield index, record
            return

        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path!r} at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSON record in {path!r} at line {line_number} must be an object"
                )
            yield line_number, record


def _validate_options(
    *,
    max_length: int,
    min_loss_tokens: int,
    max_prompts: int | None,
    cache_dir: str | None,
    cache_key: str | None,
) -> None:
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length < 1
    ):
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}")
    if (
        not isinstance(min_loss_tokens, int)
        or isinstance(min_loss_tokens, bool)
        or min_loss_tokens < 0
    ):
        raise ValueError(
            f"min_loss_tokens must be a non-negative integer, got {min_loss_tokens!r}"
        )
    if max_prompts is not None and (
        not isinstance(max_prompts, int)
        or isinstance(max_prompts, bool)
        or max_prompts < 0
    ):
        raise ValueError(
            f"max_prompts must be a non-negative integer or None, got {max_prompts!r}"
        )
    if (cache_dir is None) != (cache_key is None):
        raise ValueError("cache_dir and cache_key must be provided together")


__all__ = ["PromptTaskDict", "prepare_prompt_tasks"]
