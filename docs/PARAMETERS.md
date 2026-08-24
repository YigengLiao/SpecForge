# 参数

本 fork 的对照基准是根提交 `fec8f858`（上游 `sgl-project/SpecForge` 的一次完整复刻）。
下表只列本分支相对该基准**新增或改变**的参数，以及明确沿用不变的关键项。

分支 `glm52-dspark`，19 个提交，14 个文件，594 增 / 35 删。

---

## 第一章 相较上游的全部参数调整

### 1.1 配置 schema 新增的取值

| 参数 | 上游 | 本分支 | 生效位置 | 原因 |
|---|---|---|---|---|
| `training.fsdp_sharding` | 三选一：`SHARD_GRAD_OP` / `FULL_SHARD` / `NO_SHARD`，默认 `SHARD_GRAD_OP` | 五选一，**新增 `HYBRID_SHARD` 与 `_HYBRID_SHARD_ZERO2`**，默认不变 | `specforge/config/schema.py:488-493` | 多机场景下节点内分片、节点间复制；只有这两种取值能把跨节点流量压到只剩梯度归约 |

默认值未变，所以既有配置的行为不变。

### 1.2 新增的环境变量（都不进 schema，读自 `os.environ`）

| 变量 | 默认 | 生效位置 | 作用 |
|---|---|---|---|
| `SF_GRAD_REDUCE_DTYPE` | `float32` | `specforge/training/backend.py:255-257` | FSDP 的 `MixedPrecision.reduce_dtype`。上游不设该字段，于是它回落到 `param_dtype`（bf16），而 Adam 的动量与主权重已经是 fp32 —— 梯度在 bf16 里累积并归约，精度链断在这一环 |
| `SF_PRODUCER_SKIP_PROMPTS` | `0` | `specforge/launch.py:1194-1204` | 冷重启的 producer 没有恢复状态，会把已训过的 prompt 全部重新过一遍 target 前向。跳过前 N 条 |
| `SPECFORGE_FLEX_EAGER` | `0`，**标记为 deprecated，请保持 0** | `specforge/modeling/draft/flex_attention.py:14-16` | 置 1 会跳过 `torch.compile`。跳过之后 `NoValidChoicesError` 变成静默的 eager 回落，而 eager 的注意力显存是长度的平方 |

三者都是"上游没有这个旋钮"，不是"上游默认值被改了"。

### 1.3 新增的 chat template

| 名字 | 生效位置 | 与上游 `glm-5.2` 的差别 |
|---|---|---|
| `glm-5.2-thinking` | `specforge/data/template.py:310-323` | `assistant_header` 为 `<\|assistant\|><think>`，且 `enable_thinking=True` |

配套的解析器改动（`specforge/data/parse.py:510-515`）：

- `GLMParser` 的基类从 `GeneralParser` 换成 `ThinkingParser`
- `enable_thinking` 的默认值从写死的 `False` 改成 `bool(self.chat_template.enable_thinking)`

**这两处必须成对存在。** 只改模板不改基类，`_sanitize_message` 会在进 chat template 之前把
`reasoning_content` 剥掉，于是每个 assistant 轮都训练在空的 `<think></think>` 上。

### 1.4 draft 侧的 kernel 选择

| 参数 | 上游 | 本分支 | 生效位置 |
|---|---|---|---|
| flex_attention 的 `kernel_options` | 不设 | `{"FORCE_USE_FLEX_ATTENTION": True}`，仅当 `_attn_implementation == "flex_attention"` | `specforge/modeling/draft/dflash.py:127-130` |

原因写在那三行注释里：query 位置少于 64 时 inductor 找不到 flex_decoding 配置并抛
`NoValidChoicesError`；主 kernel 在实测过的每个长度上都能编译。`block_size=7` 的 DSpark
正是"query 位置少于 64"这一类。

### 1.5 `scripts/regenerate_train_data.py` 新增的 CLI

| 参数 | 上游 | 本分支 | 生效位置 |
|---|---|---|---|
| `--max-total-tokens` | 不存在 | 默认 `None` | `scripts/regenerate_train_data.py`（`sampling_params_group`） |
| `--request-timeout` | 不存在 | 默认 `None`（不设则用 openai 库默认值） | 同上 |

`--request-timeout` 的存在理由写在它的 help 里：高并发下请求排队，库默认超时会被触及。

### 1.6 记录状态的新增取值

`set_truncated` 引入 `status: "truncated"` 及其载荷字段
（`truncated_text` / `truncated_at_turn` / `truncated_prefix` / `truncated_usage`）。
上游只有 success / skipped / error 三态。截断记录**刻意通不过官方校验**，以免被误喂进训练。

---

## 第二章 沿用上游、不是本分支变量的关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| `training.fsdp_sharding` 默认 | `SHARD_GRAD_OP` | 只加了可选值，没动默认 |
| `training.warmup_ratio` / `max_grad_norm` 默认 | 0.015 / 0.5 | schema 默认未动。配方要 0.04 / 1.0 的项目必须自己显式写 |
| `training.batch_size` / `accumulation_steps` 默认 | 1 / 1 | |
| `training.num_anchors` / `objective_chunk_blocks` 默认 | 512 / 128 | |
| `training.attention_backend` 默认 | `flex_attention` | §1.4 只改它编译时的 kernel 选择，没改默认后端 |
| `model.sglang_attention_backend` 默认 | `flashinfer` | 未动。SM100 上 MLA `head_dim=256` 被它拒绝，需要由调用方留空 |
| `model.sglang_mem_fraction_static` 默认 | 0.4 | |
| `runtime.in_flight_high/low_watermark` 默认 | 256 / 192 | 未动。quantum 大于它的部署必须自己覆盖 |
| `runtime.producer_lease` 默认 | 8 | |
| `data.max_length` / `chat_template` / `train_only_last_turn` 默认 | 2048 / `llama3` / false | |
| `MixedPrecision.param_dtype` | 由 `pc.param_dtype` 决定 | §1.2 只补了 `reduce_dtype`，没动 `param_dtype` |

---

## 第三章 未进本分支的参数

以下六个键在 `patches/specforge/glm52-dspark-runtime.patch`（位于消费方仓库
`YigengLiao/glm52-dspark-sf`）里，**从未提交进本 fork**：
`grad_norm_percentile` / `grad_norm_window` / `grad_norm_adaptive_after` /
`grad_norm_min_samples` / `safe_grad_norm` / `safe_min_interval`。

后果：本分支的 `Config` 在 `extra="forbid"` 下会拒绝带这六个键的 YAML。需要它们的部署必须
自己应用那个补丁；不需要的部署（例如不训练的校准流程）应该渲染一份不含这六个键的配置。
