# 代码逻辑关系

只收录本分支相对根提交 `fec8f858` 引入的、且能给出完整 `file:line` 链条的逻辑。
链条断了、只能点到一处的不写。

---

## 1. thinking 训练：模板、解析器、消息清洗三处必须一致

```
specforge/data/template.py:310-323   注册 glm-5.2-thinking，enable_thinking=True
  → specforge/data/parse.py:510      GLMParser(ThinkingParser)      基类换了
  → specforge/data/parse.py:514      kwargs.setdefault("enable_thinking",
                                         bool(self.chat_template.enable_thinking))
  → ThinkingParser.apply_chat_template  保留 reasoning_content 进模板
```

上游这条链的三个环节是：`GLMParser(GeneralParser)` + `enable_thinking` 写死 `False`，于是
`GeneralParser` 的 `_sanitize_message` 在渲染前剥掉 `reasoning_content`。

**只改其中一处的后果是静默的**：模板注册了但基类没换，每个 assistant 轮都会训练在空的
`<think></think>` 上，序列渲染长度会少掉一半而不报错。

---

## 2. 梯度归约精度：不设 reduce_dtype 会让 fp32 优化器吃 bf16 梯度

```
specforge/training/backend.py:255-257  reduce_dtype = getattr(torch,
                                           os.environ.get("SF_GRAD_REDUCE_DTYPE","float32"))
  → specforge/training/backend.py:259-263  MixedPrecision(param_dtype=pc.param_dtype,
                                               reduce_dtype=reduce_dtype,
                                               buffer_dtype=torch.float32)
  → torch FSDP：未给 reduce_dtype 时回落到 param_dtype
```

上游只给 `param_dtype` 与 `buffer_dtype`，于是梯度在 `param_dtype`（bf16）里累积并归约，
而 Adam 的一阶/二阶动量与 fp32 主权重在下游已经是 fp32 —— 精度链断在归约这一环。

---

## 3. HYBRID_SHARD 必须撤掉 process_group，否则 FSDP 建不出那对组

```
specforge/config/schema.py:488-493   fsdp_sharding 新增 HYBRID_SHARD / _HYBRID_SHARD_ZERO2
  → specforge/training/backend.py:265  sharding = getattr(ShardingStrategy, pc.sharding_strategy)
  → specforge/training/backend.py:270-273
        if pc.sharding_strategy in ("HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"):
            fsdp_kwargs.pop("process_group", None)
```

因果写在那两行注释里：hybrid 需要一对 `(shard, replicate)` 进程组，传入单个
`process_group` 会阻止 FSDP 自己构建这对组。梯度范数的归约仍走 WORLD —— 它是全局量。

---

## 4. 短 query block 的 kernel 选择

```
specforge/modeling/draft/dflash.py:125   attn_fn = ALL_ATTENTION_FUNCTIONS[_attn_implementation]
  → specforge/modeling/draft/dflash.py:127-130
        if _attn_implementation == "flex_attention":
            kwargs.setdefault("kernel_options", {"FORCE_USE_FLEX_ATTENTION": True})
  → specforge/modeling/draft/flex_attention.py:39-48  WrappedFlexAttention 的 torch.compile
```

query 位置少于 64 时 inductor 找不到 flex_decoding 配置并抛 `NoValidChoicesError`。
`DSparkDraftModel` 继承 `DFlashDraftModel`，`block_size=7` 正落在这一区间。

同一文件的逃生口 `SPECFORGE_FLEX_EAGER`（`flex_attention.py:14-16`、`:39-48`、`:87-90`）
标记为 deprecated：置 1 会把编译失败变成静默的 eager 回落，而 eager 的注意力显存是长度的平方。

---

## 5. 恢复时账本回退到 checkpoint，而不是要求两者本来相等

```
specforge/launch.py:1463   checkpoint_step = _checkpoint_global_step(resume_from)
  → specforge/launch.py:1466  rewound = store.rewind_to_step(checkpoint_step)
  → specforge/runtime/control_plane/metadata_store.py  rewind_to_step 实现
  → specforge/launch.py:1467-1476  记录 acks_dropped / refs_dropped
  → specforge/launch.py:1490+  marker 与 checkpoint 的相等性检查（回退后应恒等）
```

为什么必须回退：marker 每个优化器步都推进，checkpoint 每 `save_interval` 步才落一次，所以
两者只在存档完成那一瞬间相等。上游在此处直接报错，等于要求停机时刻恰好落在存档边界上。

回退只向下（`launch.py` 的 `direction` 分支）：marker **落后于** checkpoint 意味着账本丢了
权重里已含的 ack，把它往上推等于凭空声称那些样本训过。

配套测试：`tests/test_runtime/test_recovery.py`
（`test_crash_after_ack_skips_durable_ref_and_requeues_tail`、
`test_marker_ahead_of_checkpoint_is_rejected`）。

---

## 6. 节点亲和的 ref 分发：让特征取用留在产它的那台机器上

```
specforge/runtime/data_plane/ref_distributor.py  __init__：
        self._nodes / self._ranks_per_node = dp_size // want_nodes
        self._server_node = {url: index for index, url in enumerate(urls)}
  → ref_distributor.py  _node_affine_batches()
  → ref_distributor.py  分发条件：len(self._window) + buffered + queue.depth() < dispatch_quantum
                        与 len(self._window) < dispatch_round_quantum
```

同一文件里 `dispatch_quantum` 与 `dispatch_round_quantum` 是两个不同的门槛：前者决定"攒够
一个 quantum 才发"，后者决定"这一轮攒够才发"。两者都不满足时 `break`，窗口继续累积。

---

## 7. 事件级语料的两处热点都在 prompt_builder

```
specforge/data/prompt_builder.py:134    processed_dataset.with_format("numpy")
  → prompt_builder.py  _normalize_integer_sequence：typed_source = hasattr(value, "tolist")
  → prompt_builder.py  _pack_integer_column：用 array("b"/"i") 在 C 层打包
  → prompt_builder.py  返回 array 而不是 list
```

`with_format("numpy")` 是前提而非优化：Python list 过不了 `hasattr(value, "tolist")` 这道
判据，快路根本进不去。注释记了实测 12.5 倍且逐字节相同。

```
prompt_builder.py:177   if _count_supervised(loss_mask) < min_loss_tokens
  → prompt_builder.py   _count_supervised：typecode == "b" 时用 numpy 读同一段 buffer
```

替换 `sum(loss_mask)` 的理由：builtin `sum` 会把每个元素装箱，事件级语料是 181 万行 ×
最多 4096 个位置。

**这条链同时是一处行为边界**：`_count_supervised` 决定哪些序列因监督 token 不足被丢弃，
所以它和 `minimum_valid_tokens` 一起构成"入库序列数少于语料行数"的原因。
