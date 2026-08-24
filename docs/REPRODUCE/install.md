# 复现：把本 fork 装成可用的训练环境

本分支是被依赖方，不是独立跑训练的项目。可复现的事项是"装出一个两棵树一致的环境"。

---

## 环境

以下为实测环境（Ubuntu 8×B200 节点，`nvidia-smi` 与 venv 内 python 读出）。

| 项 | 值 |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| kernel | 6.8.0-106-generic |
| GPU | NVIDIA B200 × 8，compute capability `sm_100`，183359 MiB/卡 |
| 驱动 | 590.48.01 |
| CUDA（驱动侧 / torch 侧） | 13.1 / 13.0 |
| cuDNN | 91900 |
| NCCL | 2.28.9 |
| Python | 3.11.15（uv venv） |
| torch | 2.11.0+cu130 |
| transformers | 5.8.1 |
| sglang | 0.5.14 |
| flashinfer | 0.6.12 |
| datasets | 5.0.0 |
| numpy | 2.3.5 |
| mooncake | `mooncake-transfer-engine-cuda13` |
| 本分支 | `glm52-dspark` @ `0e27daa` |
| 上游基线 | `sgl-project/SpecForge` @ `fec8f858`（本仓库根提交） |
| flash-attn | **不装**。`attention_backend=flex_attention` 不需要它，装它要从源码编译 30–60 分钟 |

---

## 1. 取到分支

```bash
git clone https://github.com/YigengLiao/SpecForge.git ~/SpecForge
cd ~/SpecForge
git remote add upstream https://github.com/sgl-project/SpecForge.git
git checkout glm52-dspark
```

成功：`git log -1` 显示 `0e27daa`，`git status --porcelain` 为空。

一个已知陷阱：既有 checkout 上的 `git checkout glm52-dspark` **不会**把分支推进到远端最新。
若本地分支已有被远端取代的旧提交，会呈现分叉：

```bash
git fetch origin
git rev-list --count HEAD..origin/glm52-dspark      # 落后多少
git rev-list --count origin/glm52-dspark..HEAD      # 超前多少（非 0 即分叉）
```
分叉时先确认本地那些提交的内容是否已被远端取代（`git diff origin/glm52-dspark..HEAD` 若几乎
全是删除，说明本地更旧），再决定前进方式。

## 2. 装进 venv

```bash
uv venv --python 3.11 ~/venvs/sftrain
source ~/venvs/sftrain/bin/activate
cd ~/SpecForge
uv pip install -v . --prerelease=allow
uv pip install mooncake-transfer-engine-cuda13
```

成功：`Installed 1 package … specforge==0.2.0 (from file:///…/SpecForge)`。

用 `uv pip` 而不是裸 `pip`：uv 建的 venv 里没有 pip 二进制，裸 `pip` 会落到 conda base 的
解释器上，匹配不到 wheel，然后报"no matching distribution"。

mooncake 的 `engine.so` 需要 `libcudart.so.13`，它在 venv 的 nvidia wheel 里但不在加载路径上：

```bash
export LD_LIBRARY_PATH="$(ls -d ~/venvs/sftrain/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

## 3. 验证两棵树一致（必须做）

checkout 与 `site-packages` 是两份实体拷贝，可以停在不同 revision。

```bash
CK=~/SpecForge/specforge
SP=~/venvs/sftrain/lib/python3.11/site-packages/specforge
for probe in \
  "data/prompt_builder.py:_count_supervised" \
  "data/prompt_builder.py:_pack_integer_column" \
  "modeling/draft/flex_attention.py:SPECFORGE_FLEX_EAGER" \
  "modeling/draft/dflash.py:FORCE_USE_FLEX_ATTENTION" \
  "runtime/control_plane/metadata_store.py:rewind_to_step" \
  "data/template.py:glm-5.2-thinking" \
  "training/backend.py:SF_GRAD_REDUCE_DTYPE" ; do
  f=${probe%%:*}; pat=${probe##*:}
  a=$(grep -c "$pat" "$CK/$f"); b=$(grep -c "$pat" "$SP/$f")
  printf "%-52s checkout=%s site-pkg=%s %s\n" "$probe" "$a" "$b" \
    "$([ "$a" = "$b" ] && [ "$a" != 0 ] && echo OK || echo MISMATCH)"
done
```

成功：七行全部 `OK`。任一行 `MISMATCH` 说明两棵树 revision 不同 —— 重装，别只改一棵。

为什么必须两棵都验：调用方若 `cd` 到 checkout 再执行住在别处的脚本，python 把 `sys.path[0]`
设成**脚本所在目录**而不是工作目录，于是实际 import 的是 `site-packages`。反之，从 checkout
目录内启动的脚本 import 的是 checkout。两条路径都存在，只验一棵会漏。

## 4. 导入自检

```bash
python - <<'PY'
import torch, transformers, specforge, mooncake.store
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("specforge", specforge.__file__)
PY
```

成功：三个 import 都不抛，且 `specforge.__file__` 指向你期望的那棵树。

## 5. 运行时旋钮链路自检（可选，无需 GPU）

```bash
python scripts/verify/specforge_patch_chain.py .
```
该脚本在消费方仓库 `glm52-dspark-sf` 里；它逐跳检查 `safe_grad_norm` /
`safe_min_interval` 是否被 callee 接受、caller 转发、endpoint 读取。**本分支不带这两个键**，
所以只有在应用了 `patches/specforge/glm52-dspark-runtime.patch` 的环境里才应通过。
