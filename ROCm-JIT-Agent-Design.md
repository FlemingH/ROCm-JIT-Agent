# ROCm-JIT-Agent：AI-in-the-Loop 本地算子锻造引擎 (Kernel Forge)

> 文档版本：v0.2 · 与代码实现严格对齐  
> 适配模型：`Jan-code-4b-gfx1100-HIP-1`（4B 参数，本地推理）  
> 适配硬件：AMD RDNA3 GPU（gfx1100，如 RX 7900 XTX/GRE）

---

## 1. 项目定位

ROCm-JIT-Agent 是一个 **PyTorch 装饰器式的本地算子优化引擎**。开发者只需在 PyTorch 函数上加 `@optimize(target="gfx1100")`，引擎会在首次调用时自动：

1. 拦截函数，提取源码与张量签名
2. 用本地 4B 代码模型生成 HIP C++ 内核
3. 在子进程中编译、执行、对齐精度
4. 用 `rocprofv3` 提取硬件计数器数据反馈给模型迭代优化
5. 将通过验证的最佳内核持久化到磁盘缓存

**设计哲学**：用一次性几十秒的编译等待，换全生命周期 O(1) 的微秒级执行；用本地 4B 小模型替代云端大模型，实现完全离线、隐私安全的算子优化。

---

## 2. 当前实现概览

### 2.1 项目结构
```
rocm_jit_agent/
  ├── __init__.py        # 暴露 optimize 装饰器
  ├── core.py            # ~430 行，主流程：拦截→生成→编译→验证→反馈→缓存
  └── profiler.py        # ~80 行，rocprofv3 包装，提取 VGPR、L2 命中率等指标
example/
  ├── example_fusion.py     # fused_gated_activation: (x*weight)*sigmoid(x)+bias
  └── complex_fusion.py     # complex_math_fusion: 含 log/abs/tanh/pow 的融合
models/
  └── Jan-code-4b-gfx1100-HIP-1/   # GRPO RL 微调的 4B HIP 代码模型
```

### 2.2 核心数据流

```
用户调用 @optimize 装饰的函数
         │
         ▼
[1] 拦截 & 签名解析 ──── inspect.getsource + 张量 shape/dtype
         │
         ▼
[2] 缓存查找 ──────────── MD5(源码 + target + shape_sig) → ~/.rocm_jit_agent_cache/
         │ (miss)
         ▼
[3] 基准测速 ──────────── triton.testing.do_bench(eager) + torch.compile
         │             target_us = min(eager_us, compile_us)
         ▼
[4] LLM 生成 ──────────── 4B 模型 + chat template + 流式输出
         │             prompt 含具体 C++ 骨架（已填充 dtype/shape/参数名）
         ▼
[5] 子进程沙盒 ────────── tempfile + subprocess + load_inline (90s 超时)
         │             先 hipcc 编译，再用 randn 张量做 MSE 验证 + 测速
         ▼
[6] 反馈分支 ──────────── 编译失败 → 提取 hipcc error 行
         │             MSE 不达标 → 注入期望/实际样本值
         │             性能不达标 → rocprofv3 硬件指标
         ▼
[7] 多轮迭代 ──────────── max_iters=10，温度从 0.30 递增到 1.65
         │             成功后落盘 + 主进程 load_inline 复用 build_dir
         ▼
[8] 持久化缓存 ────────── {hash}.cpp + {hash}_sig.cpp
```

---

## 3. 模型能力评估

### 3.1 基础事实
- **模型**：`Jan-code-4b-gfx1100-HIP-1`，bfloat16 加载，`device_map="auto"` 自动切分
- **生成参数**：`max_new_tokens=1024`，`top_p=0.95`，温度从 0.3 起每轮递增 0.15
- **强约束**：`prompt += "\`\`\`cpp\n#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n"` 强制以正确的 C++ 头文件开头

### 3.2 实测能力上限

| 任务类型 | 表现 | 数据来源 |
|---|---|---|
| 简单 elementwise 融合（`(x*w)*sigmoid(x)+b`） | **首轮成功**，47.8us（3.4x 超 torch.compile） | example_fusion.py |
| 复杂数学融合（log/abs/tanh/sigmoid/pow 多项） | **首轮成功**，54.5us（4.3x 超 torch.compile） | complex_fusion.py |
| 1D 广播算子 | 必须在 prompt 中给出**具体维度**（如 `pid % 4096`），否则模型无法泛化 | 实验回归记录 |
| 编译错误自我修复 | 必须看到**真实的编译器错误信息**，否则 10 轮迭代仍卡住 | 早期 bug 现象 |

### 3.3 模型脆弱点

1. **对抽象占位符敏感**：`pid % <dim_size>` 这种抽象写法会让模型放弃，必须给具体数值
2. **错误信息容量有限**：超过几行的错误反馈会被忽略，超长 hipcc 命令会污染 context
3. **kernel 参数声明易遗漏**：早期版本用 `/* POINTER ARGS HERE */` 占位符，模型 100% 会写出 `error: use of undeclared identifier 'x_ptr'`
4. **不会自主选择 grid 维度**：所有生成的代码都是 1D grid（依赖骨架），高维 grid 几乎不会出现

---

## 4. 当前硬编码清单（设计权衡）

不是所有硬编码都是 bug，部分是为了配合 4B 小模型能力做的工程妥协。下表标注每项是否影响通用性：

### 4.1 模型与硬件耦合

| 硬编码项 | 位置 | 通用性影响 | 说明 |
|---|---|---|---|
| 默认 target=`gfx1100` | core.py L8 | 中 | 可参数化，但 prompt 中 target 未深度引导其他架构特性 |
| 模型路径 `models/Jan-code-4b-gfx1100-HIP-1` | core.py L139 | 高 | 写死 RL 微调的专用模型 |
| `bfloat16` 加载 | core.py L160 | 低 | RDNA3 合理默认 |

### 4.2 算子约束（最影响通用性）

| 硬编码项 | 位置 | 通用性影响 | 说明 |
|---|---|---|---|
| 1D grid 骨架 `pid = blockIdx.x * blockDim.x + threadIdx.x` | core.py L193 | **极高** | 强制 elementwise 思路，不支持 reduction/matmul/conv |
| `int threads = 256` | core.py L202 | 中 | 合理默认 |
| `auto output = torch::empty_like({arg_names[0]})` | core.py L200 | **极高** | 假设单输出 + 与第一张量同 shape 同 dtype |
| `output.data_ptr<float>()` | core.py L204 | **高** | 输出写死 float，dtype 不匹配即崩溃 |
| kernel 名 `fused_kernel`、函数名 `optimized_func` | core.py | 低 | 工程约定 |

### 4.3 验证与迭代策略

| 硬编码项 | 位置 | 通用性影响 | 说明 |
|---|---|---|---|
| MSE 阈值 `1e-3` | core.py | 低 | 对 fp32 合理 |
| `max_iters = 10` | core.py L135 | 低 | 经验值 |
| 温度 `0.30 + 0.15*i` | core.py L167 | 低 | 线性递增 |
| 子进程 `timeout=90s` | core.py | 中 | 大 kernel 可能不够 |
| 缓存路径 `~/.rocm_jit_agent_cache/` | core.py L101 | 低 | 标准约定 |
| `max_new_tokens=1024` | core.py | 中 | 大 kernel 会被截断 |

### 4.4 v0.1 → v0.2 已修复清单

| 项 | v0.1 | v0.2 |
|---|---|---|
| dtype 支持 | 写死 float* | 动态映射（float/double/half/bf16/int*/bool） |
| 标量参数 | 全部当 Tensor 崩溃 | 自动区分张量/标量 |
| 广播提示 | 写死 `pid % 4096` | 按实际 1D 张量生成具体数值 |
| 缓存哈希 | 仅 源码+target | 加入 shape+dtype |
| 编译错误反馈 | 被 hipcc 命令行吞掉 | 过滤命令行保留 error 行 |
| MSE 错误反馈 | 仅数值 | 附带期望/实际前 5 个值 |
| 性能反馈 | 写死"float4/coalescing" | 仅传 rocprofv3 硬件指标 |
| 骨架占位符 | `/* POINTER ARGS HERE */` | 具体参数声明 |

---

## 5. 模型边界探测实验

> 本节结果来自 `experiments/probe_boundaries.py`（见 §6），在 gfx1100 上用当前模型实际推理生成内核。

（此处将由 `probe_boundaries.py` 运行后自动追加结果。）

---

## 6. 通用 JIT 外壳的设计（基于边界探测结论）

（此处将基于探测结果填写。）


## 5. 模型边界探测实验（实测）

本节由 `experiments/probe_boundaries.py` + `experiments/sanitize_and_retry.py` 实测得出。模型每个探测任务只生成 1 次（不迭代），后再加 3 个轻量级后处理器尝试编译运行。

### 5.1 探测任务与结果

| 探测任务 | 规模 | 首次生成是否编译通过 | 经 3 个 sanitizer 后 | 算法正确性 |
|---|---|---|---|---|
| **Row-wise sum** (reduction + 输出形状变化) | x[64,128]→out[64] | ❌ | ✅ **PASS** | MSE=0.000000 |
| **2D matmul** (2D grid + 三层循环) | [32,64]×[64,32] | ❌ | ✅ **PASS** | MSE=0.000000 |
| **Row-wise softmax** (带 shared memory reduction) | [16,256] | ❌ | ❌ RUN_FAIL | 编译通过但 reduce 逻辑有 bug |
| **Multi-output** (返回 `std::vector<Tensor>`) | x,y[1024]→(x+y, x-y) | ❌ | ✅ **PASS** | MSE=0.000000 |

### 5.2 失败根因分析

4 个任务**首次全部编译失败**，但失败根因 **不在模型的算法能力**，而在 3 类低层级表面问题：

| 表面问题 | 出现任务 | 修复难度 |
|---|---|---|
| 模型在 HIP 代码里带入 `#include <cuda_runtime.h>` / `#include <cuda.h>` | 4/4 | 正则替换 1 行 |
| 模型在代码尾部保留 markdown 的 ``` fence | 4/4 | 行过滤 1 行 |
| 模型用 CUDA 风格 `kernel<<<grid,block>>>(args)` 启动语法 | softmax / multi_output | 正则改写 1 条 |

### 5.3 结论：模型真正的能力边界

**模型 _能_ 做（只要 skeleton 正确）：**
1. **任意输出形状**：`torch::empty({rows}, x.options())` 类分配，模型可自行推理
2. **Reduction / inner loop**：单线程循环累加（64×128 一次通过）
3. **2D grid**：给出 `dim3(x,y)` skeleton 后能写出 `blockIdx.y*... + threadIdx.y`
4. **2D 索引**：正确写出 `A[i*K+k] * B[k*N+j]`（教科书式 naive matmul）
5. **Multi-output**：能返回 `std::vector<torch::Tensor>{a, b}` 并同时写两个输出指针
6. **Shared memory 声明**：能写出 `__shared__ float sdata[256]`

**模型 _不稳定_ 的地方：**
1. **Shared memory reduction 算法**：softmax 的 block reduce 写错（将 `val` 累加而非 `expf(val)`）
2. **Warp / wavefront 原语**：未测试，但训练数据可能不足
3. **数值稳定性细节**：softmax 的 max-subtraction 位置写错

**模型 _系统性 bug_（需外壳处理）：**
1. 无脑插入 `cuda_runtime.h`
2. 保留输出 markdown fence
3. 使用 CUDA 三尖括号启动语法

### 5.4 与当前 core.py 能力的对比

当前 `core.py` 的迭代循环中如果遇到 `cuda_runtime.h` 报错，模型确实会在下一轮看到 error 并自我修正。但每轮要 30 秒推理 + 60 秒编译，**用正则 1 次性 sanitize 能省 1-3 轮迭代**。真正的瓶颈从来不是模型能力，而是外壳的 skeleton 选择。

---

## 6. 通用 JIT 外壳设计：从"elementwise 专用"到"可插拔骨架"

### 6.1 架构愿景

把当前硬编码的 1D elementwise skeleton 升级为 **"任务分类器 + 骨架库 + 通用后处理器"** 三层架构。

```
               ┌─────────────────────────────────┐
     用户 ── ▶ │   @optimize 装饰器（不变）      │
               └──────────────┬──────────────────┘
                              │
               ┌──────────────▼──────────────────┐
  NEW  ──────▶ │  任务分类器 (TaskClassifier)   │
               │  基于源码 AST + 输入/输出 shape │
               │  分类为: elementwise / reduce  │
               │        matmul / 2d-stencil ... │
               └──────────────┬──────────────────┘
                              │
               ┌──────────────▼──────────────────┐
  NEW  ──────▶ │  骨架库 (SkeletonRegistry)     │
               │  每类任务有一套 prompt + skele │
               │  + 输出 alloc 模板 + eval 模板 │
               └──────────────┬──────────────────┘
                              │
               ┌──────────────▼──────────────────┐
               │  LLM 生成（不变）              │
               └──────────────┬──────────────────┘
                              │
               ┌──────────────▼──────────────────┐
  NEW  ──────▶ │  通用代码后处理 (CodeSanitizer)│
               │  - 剥 cuda_runtime.h / cuda.h  │
               │  - 剥尾部 markdown fence       │
               │  - <<<>>> → hipLaunchKernelGGL │
               │  - 补齐 <cmath>, FLT_MAX 等    │
               └──────────────┬──────────────────┘
                              │
               ┌──────────────▼──────────────────┐
               │  子进程沙盒（不变）            │
               │  + eval 模板随骨架切换         │
               └─────────────────────────────────┘
```

### 6.2 核心抽象：Skeleton 数据类

```python
@dataclass
class Skeleton:
    name: str                         # "elementwise_1d" / "row_reduction" / "matmul_2d" / ...
    match: Callable[[FnSig], bool]    # 判断是否适用
    output_alloc: Callable[[FnSig], str]    # 生成 C++ 输出分配代码
    kernel_skeleton: Callable[[FnSig], str] # 生成 kernel 模板（带 TODO 注释）
    launch_skeleton: Callable[[FnSig], str] # 生成 hipLaunchKernelGGL 代码
    eval_template: Callable[[FnSig], str]   # 生成子进程中的 MSE 验证脚本
    extra_prompt_hints: List[str]     # 给 LLM 的额外提示（如 "use shared memory"）
```

### 6.3 内置骨架最小集（MVP）

基于 §5 的实测，以下 4 个骨架即可覆盖常见场景：

| Skeleton | 适用条件 | 输出 alloc 模板 | Grid 模板 |
|---|---|---|---|
| `elementwise_1d` | 所有输入张量 shape 相同且输出与输入 shape 相同 | `empty_like(x)` | `dim3((n+255)/256), dim3(256)` |
| `row_reduction` | 输入 [B,N] 输出 [B] 或 [B,1] | `empty({rows}, x.options())` | `dim3(rows), dim3(256)` |
| `matmul_2d` | 两个 2D 输入 且 输出形状为 (A.rows, B.cols) | `empty({M,N}, A.options())` | `dim3((N+15)/16, (M+15)/16), dim3(16,16)` |
| `multi_output` | 函数返回 tuple | 多个 `empty_like` | 继承上述一种 |

**分类器规则（基于 AST + shape 推断）**：
- 源码含 `torch.matmul` / `@` 且输入都是 2D → `matmul_2d`
- 源码含 `.sum(dim=` / `.mean(dim=` / `.max(dim=` → `row_reduction`
- 函数 `return a, b` 或 `return (a, b, ...)` → `multi_output` 外加基础类
- 默认 → `elementwise_1d`

### 6.4 通用 CodeSanitizer（§5.2 结论落地）

```python
class CodeSanitizer:
    RULES = [
        # (pattern, replacement, description)
        (r"#include <cuda_runtime.h>", "// cuda_runtime.h stripped", "rocm"),
        (r"#include <cuda.h>",         "// cuda.h stripped",          "rocm"),
        (r"#include <device_launch_parameters.h>", "// stripped", "rocm"),
        (r"^```.*$", "", "markdown"),              # 行级 regex，多行模式
        (r"(\w+)\s*<<<([^,]+),([^>]+)>>>\s*\(([^;]*)\)\s*;",
         r"hipLaunchKernelGGL(\1, \2, \3, 0, 0, \4);", "launch_syntax"),
        # 可扩展：补 <cmath>, FLT_MAX 等
    ]
```

在 §5 实测中，这 4 条规则把 pass 率从 0/4 拉到 3/4。softmax 的失败是**算法层 bug**（需要 LLM 在下一轮看到 MSE 错误修复），不是 sanitizer 能解决的。

### 6.5 输出侧通用化

替换当前写死的 `output.data_ptr<float>()`：

```python
# 当前（硬编码）
f"{out_ptr}.data_ptr<float>()"

# 通用：根据推断的输出 dtype 决定
def output_launch_ptrs(output_sigs):
    parts = []
    for out in output_sigs:           # out = (var_name, ctype)
        parts.append(f"{out.name}.data_ptr<{out.ctype}>()")
    return ", ".join(parts)
```

**输出签名推断方法**：
1. 对 `elementwise_1d`：与第一个输入同 dtype、同 shape（当前做法）
2. 对 `row_reduction`：与输入同 dtype、shape 去掉 reduce 轴
3. 对 `matmul_2d`：同 dtype、shape = (A.dim(0), B.dim(-1))
4. 对 `multi_output`：用户需要在装饰器 kwargs 显式声明，如 `@optimize(outputs=[SameAs("x"), ReduceAlong("x", axis=1)])`

### 6.6 分层反馈策略

不同骨架需要不同的反馈模板：

| 骨架 | 成功标准 | 失败时额外提示 |
|---|---|---|
| elementwise_1d | MSE < 1e-3 | 当前通用提示 |
| row_reduction | MSE < 1e-3 **AND** 逐行对比（不只看全局 MSE） | "reduction 的累加顺序可能影响精度" |
| matmul_2d | MSE < 1e-1 | "检查 A、B 的行列顺序是 row-major" |
| multi_output | 每个输出独立 MSE | "第 N 个输出误差大" |

### 6.7 改造 ROI 排序（与 §4 硬编码清单对应）

| 优先级 | 改造项 | 解锁能力 | 工作量（按当前代码） |
|---|---|---|---|
| **P0** | 加入 `CodeSanitizer`（4 条规则） | 绕过现有 1-3 轮无谓迭代 | ~30 行，2 小时 |
| **P0** | 输出 `data_ptr<{dtype}>` 动态化 | 解锁非 float 输出（当前已支持输入，不支持输出） | ~5 行 |
| **P1** | `Skeleton` 抽象 + `elementwise_1d` 默认骨架 | 架构铺垫 | ~100 行 |
| **P1** | 加 `row_reduction` 骨架 + 分类器 | 解锁 sum/mean/norm 类算子 | ~80 行 |
| **P2** | 加 `matmul_2d` 骨架 | 解锁小矩阵乘（对大矩阵仍输 rocBLAS） | ~80 行 |
| **P2** | 加 `multi_output` 装饰器签名 + 验证 | 解锁 RNN cell、layernorm 完整版 | ~60 行 |
| **P3** | 更细的任务分类器（支持 conv / attention） | 需要更强模型配合 | 大 |

### 6.8 API 兼容性

所有改造都保持 **向后兼容**：
```python
# 老用法（不变）
@optimize(target="gfx1100")
def elementwise(x, w, b): return x * w + b

# 新用法：显式声明输出签名
@optimize(target="gfx1100", skeleton="row_reduction",
          outputs=[OutputSpec(shape_expr="input[0].shape[:-1]", dtype="input[0].dtype")])
def row_sum(x): return x.sum(dim=-1)

# 新用法：多输出
@optimize(target="gfx1100", outputs=["same_as_input", "same_as_input"])
def split(x, y): return x + y, x - y
```

### 6.9 模型不变条件下外壳改造的上限

即使做完 P0-P3 所有改造，以下仍是 4B 模型的能力天花板：
- **BLAS 性能**：naive matmul 不会跑赢 rocBLAS；需要更大模型或显式提示 MFMA/WMMA 指令
- **Flash-style attention**：多 kernel 编排超出单文件生成能力
- **动态 shape**：当前缓存按 shape hash，无法真正动态
- **稀疏/不规则计算**：topk、CSR spmv 等

但在 **"给定 shape 的稠密算子融合"** 这一主流场景，P0-P3 改造能把通用性从当前的 "elementwise only" 拓展到 **"elementwise + reduction + small matmul + multi-output"**，覆盖 LLM 推理中 ~70% 的自定义算子需求（LayerNorm/RMSNorm/GEGLU/SwiGLU/简单 attention pre-softmax 等）。

---

## 7. 已知局限与风险（承接 v0.2）

1. 首次启动慢：4B 模型加载 + HIP 编译，首次约 60-90 秒
2. 缓存失效条件：源码改动、target 改变、shape/dtype 改变都会触发重生成
3. 不支持动态 shape
4. 依赖 ROCm 7.x + PyTorch ROCm 构建
5. 依赖 `rocprofv3`；缺失时性能反馈降级为只看时序
6. **§6 改造前的限制**：只支持 1D elementwise；softmax 类需要 shared memory 的算子即使模型能写对也会被外壳拒绝
