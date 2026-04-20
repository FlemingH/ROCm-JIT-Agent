# ROCm-JIT-Agent：AI-in-the-Loop 本地算子锻造引擎 (Kernel Forge)

> 文档版本：v1.0 · 与代码实现严格对齐  
> 适配模型：`Jan-code-4b-gfx1100-HIP-1`（4B 参数，GRPO 强化学习微调，本地 bfloat16 推理）  
> 基座模型：`Jan-code-4b`（4B 参数，无 RL，用于消融对比）  
> 适配硬件：AMD RDNA3 GPU（gfx1100，如 RX 7900 XTX/GRE）  
> 运行环境：ROCm 7.2.0 · PyTorch 2.9.1+rocm7.2.0 · conda env `ra`

---

## 1. 项目定位

ROCm-JIT-Agent 是一个 **PyTorch 装饰器式的本地算子优化引擎**。开发者只需在 PyTorch 函数上加 `@optimize(target="gfx1100")`，引擎在首次调用时自动：

1. 拦截函数，提取源码与张量签名
2. **三层架构**分类任务、选择骨架、生成 prompt
3. 用本地 4B 代码模型生成 HIP C++ 内核
4. 确定性后处理（Sanitizer）修复已知模型 surface bug
5. 子进程沙盒中编译、验证精度、稳态测速
6. 用 `rocprofv3` 提取硬件计数器反馈模型迭代优化
7. 将通过验证的最佳内核持久化到磁盘缓存

**设计哲学**：用一次性几十秒到几分钟的编译等待，换全生命周期 O(1) 微秒级执行；用本地 4B 小模型替代云端大模型，实现完全离线、隐私安全的算子优化。

---

## 2. 项目结构

```
rocm_jit_agent/
  ├── __init__.py         # 暴露 optimize 装饰器
  ├── core.py             # ~450 行，主流程：拦截→分类→生成→后处理→编译→验证→反馈→缓存
  │                       # 含 _robust_bench_us()、_pick_mse_threshold()、_DTYPE_MSE_THRESH
  ├── skeletons.py        # ~490 行，Layer 1+2：任务分类器 + 4 个可插拔骨架
  │                       # 含 classify()、_DATA_DEP_PATTERNS、SkeletonContext、Skeleton
  ├── sanitizer.py        # ~58 行，Layer 3：确定性代码后处理（3 类规则）
  └── profiler.py         # ~82 行，rocprofv3 包装，提取 VGPR/L2/occupancy 等指标
example/                  # 14 个测试例子（覆盖 4 种骨架 × 3 种 dtype × 1D/2D/3D）
  ├── example_fusion.py            # elementwise_1d, fp32, 2D — baseline fusion
  ├── complex_fusion.py            # elementwise_1d, fp32, 2D — log/abs/tanh/sigmoid/pow
  ├── gelu_example.py              # elementwise_1d, fp32, 2D — GELU activation
  ├── swiglu_example.py            # elementwise_1d, fp32, 2D — SwiGLU activation
  ├── silu_bf16_example.py         # elementwise_1d, bf16, 2D — bf16 dtype path
  ├── fp16_example.py              # elementwise_1d, fp16, 2D — fp16 dtype path
  ├── scaled_activation_example.py # elementwise_1d, fp32, 2D — scalar float args
  ├── attention_score_3d_example.py# elementwise_1d, fp32, 3D — 3D shape + scalar
  ├── reduction_example.py         # row_reduction, fp32, 2D — .sum(dim=-1)
  ├── l2_norm_example.py           # row_reduction, fp32, 2D — sqrt(sum(x²))
  ├── matmul_example.py            # matmul_2d, fp32, 2D — naive matmul
  ├── multi_output_example.py      # multi_output, fp32, 1D — (x+y, x-y)
  ├── unsupported_op_example.py    # 负例：torch.cumsum，触发低置信度警告
  ├── stability_test.py            # 基准测速方法论验证
  └── README.md                    # 测试清单、发现、系统问题分析
models/
  ├── Jan-code-4b/                       # 基座模型（无 RL）
  ├── Jan-code-4b-gfx1100-HIP-1/        # GRPO RL 微调模型（生产用）
  └── grpo-jan-code-4b-b26/             # RL 训练 adapter 权重与 checkpoints
```

---

## 3. 三层架构

```
             ┌──────────────────────────────────┐
   用户 ───▶ │   @optimize(target, skeleton=?)  │  decorator, 向后兼容
             └──────────────┬───────────────────┘
                            │ inspect.getsource + 张量/标量分离
             ┌──────────────▼───────────────────┐
  Layer 1 ─▶ │  TaskClassifier (classify())     │  skeletons.py
             │  源码 regex + shape + 返回值分析  │
             │  输出：(Skeleton, confidence, reason)
             └──────────────┬───────────────────┘
                            │
             ┌──────────────▼───────────────────┐
  Layer 2 ─▶ │  SkeletonRegistry (4 骨架)       │  skeletons.py
             │  build_prompt_block(ctx, sk)      │  → LLM user message
             │  build_eval_compare(ctx, sk)      │  → 子进程验证脚本
             └──────────────┬───────────────────┘
                            │
             ┌──────────────▼───────────────────┐
             │  LLM 生成（chat template + stream)│  core.py
             └──────────────┬───────────────────┘
                            │
             ┌──────────────▼───────────────────┐
  Layer 3 ─▶ │  CodeSanitizer (sanitize())      │  sanitizer.py
             │  ① 剥 cuda_runtime.h / cuda.h    │
             │  ② 剥 markdown ``` fence         │
             │  ③ <<<>>> → hipLaunchKernelGGL   │
             └──────────────┬───────────────────┘
                            │
             ┌──────────────▼───────────────────┐
             │  子进程沙盒编译 + 验证 + 测速     │  core.py (subprocess)
             │  dtype-aware MSE threshold        │
             │  steady-state timing (median)     │
             └──────────────┬───────────────────┘
                            │
             ┌──────────────▼───────────────────┐
             │  rocprofv3 硬件反馈 → LLM 迭代   │  profiler.py + core.py
             │  max_iters=10, temp 0.30→1.65    │
             └──────────────┬───────────────────┘
                            │
             ┌──────────────▼───────────────────┐
             │  持久化缓存                       │  ~/.rocm_jit_agent_cache/
             │  {MD5(src+target+shape+sk)}.cpp   │
             └──────────────────────────────────┘
```

### 3.1 Layer 1: TaskClassifier (`classify()`)

**输入**：`SkeletonContext`（函数名、源码、target、tensor_args、scalar_args、arg_names）+ 可选 `user_hint`

**输出**：`(Skeleton, confidence: float, reason: str)`

| confidence | 含义 |
|---|---|
| 1.00 | 用户通过 `skeleton=...` 显式指定 |
| 0.90 | 结构化匹配（multi-output 返回值 / matmul op + 2D / reduction-with-dim） |
| 0.50 | 默认 fallthrough（无正面信号，选 elementwise_1d） |
| ≤0.30 | 源码含 `_DATA_DEP_PATTERNS` 中的数据依赖算子（cumsum/sort/scatter/gather 等），当前无骨架能正确表达 |

当 `confidence < 0.5` 时，`core.py` 打印 `⚠️ LOW CLASSIFIER CONFIDENCE` 警告。

**分类规则优先级**：
1. `user_hint` → 直接映射（1.00）
2. 多返回值（AST 检测）→ `multi_output`（0.90）
3. `torch.matmul / torch.mm / @` + 输入均 ≥2D → `matmul_2d`（0.90）
4. `.sum(dim= / .mean(dim= / .max(dim= / .min(dim= / .prod(dim= / .norm(dim=` + 输入 ≥2D → `row_reduction`（0.90）
5. 检查 `_DATA_DEP_PATTERNS` → `elementwise_1d`（0.30 + 警告）
6. 默认 → `elementwise_1d`（0.50）

### 3.2 Layer 2: SkeletonRegistry（4 个内置骨架）

| Skeleton | 返回类型 | Grid 策略 | Prompt 特化 | Eval 特化 |
|---|---|---|---|---|
| `elementwise_1d` | `torch::Tensor` | 1D: `(n+255)/256, 256` | `pid = blockIdx.x * blockDim.x + threadIdx.x` | 全局 MSE |
| `row_reduction` | `torch::Tensor` | 1D: `rows, blockDim` | 一个 block 处理一行，shared memory 累加 | 逐行对比 |
| `matmul_2d` | `torch::Tensor` | 2D: `(N+15)/16, (M+15)/16` 各 `16×16` | C[i][j] = dot(A[i,:], B[:,j]) | MSE ≤ MATMUL_MSE_THRESH |
| `multi_output` | `std::vector<torch::Tensor>` | 继承 elementwise_1d | 多个输出指针声明 | 每个输出独立 MSE |

每个 Skeleton 是一个 dataclass，包含：
- `build_prompt_block(ctx, sk) → str`：生成 LLM user message（含具体 dtype、shape、C++ 签名模板）
- `build_eval_compare(ctx, sk) → str`：生成子进程中的验证 Python 代码
- `signature_regex`：从生成的 C++ 中提取 `optimized_func` 原型
- `cpp_return_type`、`n_outputs`、`description`

### 3.3 Layer 3: CodeSanitizer (`sanitize()`)

确定性后处理，修复 3 类已知模型 surface bug：

| 规则 | 触发条件 | 修复 |
|---|---|---|
| Strip CUDA headers | `#include <cuda_runtime.h>` / `<cuda.h>` / `<device_launch_parameters.h>` / `<cuda_runtime_api.h>` | 注释化 |
| Strip markdown fences | 行首 ``` | 删除该行 |
| Rewrite launch syntax | `kernel<<<grid,block>>>(args);` | `hipLaunchKernelGGL(kernel, grid, block, 0, 0, args);` |

返回 `(cleaned_code, patches_applied: List[str])`，patches 会打印到日志。

---

## 4. 核心数据流详解

### 4.1 签名解析与参数分离

```python
# core.py wrapper() 内
arg_names = inspect.signature(func).parameters.keys()
tensor_args = [(name, arg) for ...]   # isinstance(arg, torch.Tensor)
scalar_args = [(name, arg) for ...]   # int / float / bool
```

标量参数自动映射 C++ 类型：`float→float`, `int→int`, `bool→bool`（`scalar_to_ctype()`）。
张量 dtype 映射：`torch.float32→float`, `torch.float16→at::Half`, `torch.bfloat16→at::BFloat16` 等（`dtype_to_ctype()`，10 种 dtype 覆盖）。

### 4.2 基准测速

使用 `_robust_bench_us(fn)` —— 通用 steady-state GPU 计时：
1. 25 次 warmup 调用
2. 5 轮 × 100 次迭代，每轮用 `torch.cuda.Event` 计时
3. 取 5 轮的中位数（微秒）

同时用于 eager baseline、torch.compile baseline 和子进程内核评估。替代了早期的 single-shot 计时，解决了 cold-cache bias 导致 speedup 虚高的问题。

验证：`stability_test.py` 确认内部报告 eager=27.4us 与外部 200 次重复中位数 26.8us 一致。

### 4.3 缓存键

```python
shape_sig = str([(list(a.shape), str(a.dtype)) for a in args if isinstance(a, torch.Tensor)])
func_hash = MD5(source_code + target + shape_sig + sk.name + str(sk.n_outputs))
```

不同 shape → 不同 key → 不同内核。不支持动态 shape。

### 4.4 dtype-aware 验证阈值

`_pick_mse_threshold(tensor_infos)` 根据输入中最低精度 dtype 自动选择阈值：

| dtype | MSE 阈值 |
|---|---|
| float64 | 1e-6 |
| float32 | 1e-3 |
| float16 | 5e-2 |
| bfloat16 | 1e-1 |
| int* / bool | 0.0（精确匹配） |

matmul 骨架额外使用 `max(threshold, 1e-1)` 作为 floor。

### 4.5 迭代策略

- `max_iters = 10`，温度从 0.30 每轮递增 0.15（到 1.65）
- **成功 → 达标（≤ target_us）则立即退出 + 缓存**
- **成功但未达标** → rocprofv3 硬件反馈注入下一轮 prompt
- **编译失败** → 提取 hipcc error 行注入下一轮
- **数值错误** → 注入期望/实际前 5 个值 + MSE
- **10 轮全败** → 保留原始 PyTorch 函数（graceful fallback），不抛异常

---

## 5. 模型能力评估

### 5.1 RL 模型 vs 基座模型 消融对比

使用 14 个例子全量冷启动（清空缓存），RL 模型 = `Jan-code-4b-gfx1100-HIP-1`，基座 = `Jan-code-4b`。

| 指标 | RL 模型 | 基座模型 | 差异 |
|---|---|---|---|
| **达标数** (beat min(eager, compile)) | **7/14** | 4/14 | **RL +75%** |
| **完全失败** (10 轮全败退回 PyTorch) | **0/14** | 1/14 | RL 更稳 |
| 平均迭代数 | 6.8 | 7.4 | RL 少 0.6 轮 |
| **平均失败次数/例** | **1.4** | 3.4 | **RL -59%** |
| 平均 speedup vs torch.compile (13 个都成功的) | **2.55x** | 1.96x | **RL +30%** |
| 总编译耗时 | 5011s | 4807s | 基本持平 |

### 5.2 RL 增益集中在 reduction / matmul 场景

| Example | RL kernel | Base kernel | RL 优势 |
|---|---|---|---|
| reduction (.sum) | 4.7us | 22.4us | **4.8x faster** |
| l2_norm (sqrt(sum(x²))) | 5.3us | 39.4us | **7.4x faster** |
| matmul | 12.4us | 13.5us | 1.1x faster |
| example_fusion | 34.5us (PASS) | FAIL (10/10 全败) | RL: 100% vs 0% |

RL 训练教会了模型：(a) 用合理的 block size 做 shared memory reduction；(b) 避开常见 HIP C++ 编译陷阱。

### 5.3 两者无差异的场景

简单 elementwise op（fp16/attention_3d/multi_output）两者 1-iter 达标，speedup 相同（4.3-4.8x）。基座模型本身就有足够的 elementwise HIP kernel 知识。

### 5.4 模型能力边界（RL 和基座共同的天花板）

**能做：**
- 任意 elementwise 融合（含 tanh/sigmoid/exp/pow/sqrt 等超越函数）
- 简单 shared memory reduction（sum/mean/norm，RL 模型更可靠）
- Naive matmul（for 循环逐元素累加，不会跑赢 rocBLAS）
- Multi-output kernel（`std::vector<torch::Tensor>`）
- 正确的 dtype 处理（fp32/fp16/bf16 各用正确的 C++ 类型）
- 超出 skeleton 范围的简单算子（如 cumsum — 模型忽略 elementwise 提示写了正确的 prefix sum）

**不能做：**
- Tiling / vectorized load（float4）/ memory coalescing 优化
- Warp/wavefront-level reduction（`__shfl_xor_sync` 等内在函数）
- Flash-style attention（多 kernel 编排）
- 动态 shape kernel（shape 值硬编码在生成的内核中）
- 大 tensor 高性能（numel ≥ 4M 系统性输给 torch.compile）

**模型 3 类系统性 surface bug（由 Sanitizer 处理）：**
1. 插入 `#include <cuda_runtime.h>` / `<cuda.h>`
2. 保留 markdown ``` fence
3. 使用 CUDA `<<<>>>` 启动语法

### 5.5 性能分布

| 场景 | vs torch.compile | vs eager | 备注 |
|---|---|---|---|
| 小 tensor (numel ≤ 1M) | **1.8–4.7x** | 2–19x | torch.compile 有固定 launch overhead |
| 大 tensor (numel ≥ 4M) | **0.6–0.9x** | 1.5–3x | 模型只能生成 naive per-thread kernel |

---

## 6. 全量测试结果（RL 模型，v1.0）

14 例全部清缓存冷启动，所有例子正确性 100% 通过（MSE=0 或在 dtype 容差内）。

| # | Example | Skeleton | Conf | dtype | shape | Iters | Fails | 达标 | Eager | tc | Opt | Speedup vs tc | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | example_fusion | elem_1d | 0.50 | fp32 | 2D | 10 | 0 | ✨ | 109us | 25us | 35us | 0.7x | 558s |
| 2 | complex_fusion | elem_1d | 0.50 | fp32 | 2D | 10 | 0 | ✨ | 691us | 47us | 58us | 0.8x | 626s |
| 3 | gelu | elem_1d | 0.50 | fp32 | 2D | 10 | 1 | ✨ | 1479us | 57us | 77us | 0.7x | 519s |
| 4 | swiglu | elem_1d | 0.50 | fp32 | 2D | 10 | 2 | ✨ | 247us | 42us | 47us | 0.9x | 530s |
| 5 | silu_bf16 | elem_1d | 0.50 | **bf16** | 2D | 10 | 4 | ✨ | 41us | 22us | 33us | 0.7x | 489s |
| 6 | fp16 | elem_1d | 0.50 | **fp16** | 2D | 1 | 0 | **🏆** | 9us | 20us | **4us** | **4.7x** | 59s |
| 7 | scaled_activation | elem_1d | 0.50 | fp32 | 2D | 10 | 0 | ✨ | 62us | 23us | 30us | 0.8x | 504s |
| 8 | attention_3d | elem_1d | 0.50 | fp32 | **3D** | 1 | 0 | **🏆** | 8us | 21us | **5us** | **4.3x** | 62s |
| 9 | reduction | row_red | 0.90 | fp32 | 2D | 2 | 0 | **🏆** | 8us | 20us | **5us** | **4.3x** | 111s |
| 10 | l2_norm | row_red | 0.90 | fp32 | 2D | 10 | 7 | **🏆** | 20us | 21us | **5us** | **3.9x** | 433s |
| 11 | matmul | matmul_2d | 0.90 | fp32 | 2D | 9 | 1 | **🏆** | 13us | 46us | **12us** | **3.7x** | 504s |
| 12 | multi_output | multi_out | 0.90 | fp32 | 1D | 1 | 0 | **🏆** | 8us | 22us | **5us** | **4.2x** | 59s |
| 13 | cumsum (负例) | elem_1d | **0.30** | fp32 | 2D | 10 | 4 | ✨ | 6us | 21us | 9us | 2.3x | 496s |
| 14 | stability_test | elem_1d | 0.50 | fp32 | 2D | 1 | 0 | **🏆** | 27us | 22us | **12us** | **1.8x** | 61s |

- 🏆 = 达成 target_us（≤ min(eager, compile)）：7/14
- ✨ = 编译成功取最优但未达标：7/14
- ❌ = 全部失败：0/14
- 平均编译耗时：~360s/例（1-iter 达标 ~60s，10-iter 耗尽 ~530s）

---

## 7. 工程实现细节

### 7.1 dtype 全链路支持

| 环节 | 实现 |
|---|---|
| 输入 dtype → C++ 类型 | `dtype_to_ctype()`：10 种 dtype → C type 映射 |
| 标量参数 → C++ 类型 | `scalar_to_ctype()`：bool/int/float |
| Prompt 中的 dtype 提示 | 骨架自动嵌入具体的 `data_ptr<at::BFloat16>()` 等 |
| 验证阈值 | `_pick_mse_threshold()`：按最低精度 dtype 自动选择 |
| 输出分配 | `torch::empty_like(input[0])` 继承 dtype（elementwise） |

### 7.2 缓存机制

- 路径：`~/.rocm_jit_agent_cache/`
- 文件：`{hash}.cpp`（HIP 内核源码）+ `{hash}_sig.cpp`（C++ 函数签名）
- 命中时：直接 `load_inline` 加载，跳过 LLM 推理和编译
- 失效条件：源码修改、target 改变、输入 shape/dtype 改变、skeleton 改变
- `force_recompile=True` 可强制跳过缓存

### 7.3 LLM 推理

- 加载：`AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")`
- 生成：`max_new_tokens=1024`, `top_p=0.95`, `do_sample=True`, 温度递增
- Prompt 前缀强制：`prompt += "```cpp\n#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n"`
- 流式输出：`TextStreamer` 子类实时显示生成进度
- 模型路径：`models/Jan-code-4b-gfx1100-HIP-1`（硬编码，RL 微调版本）

### 7.4 子进程沙盒

- 用 `tempfile.NamedTemporaryFile` 写 eval script
- `subprocess.run` + `timeout=120s`
- 内置 `load_inline` 编译 HIP 内核
- 验证：MSE against original PyTorch function
- 测速：25 warmup + median of 5×100 (与主进程一致)
- 输出协议：`SUCCESS:mse:opt_us:build_dir` 或 `ERROR:message`

---

## 8. API 参考

```python
from rocm_jit_agent import optimize

# 最简用法：自动分类骨架
@optimize(target="gfx1100")
def my_op(x, w, b):
    return torch.relu(x * w + b)

# 显式指定骨架 + 强制重编译
@optimize(target="gfx1100", skeleton="row_reduction", force_recompile=True)
def row_sum(x):
    return x.sum(dim=-1)

# 多输出（自动检测）
@optimize(target="gfx1100")
def split_op(x, y):
    return x + y, x - y

# 带标量参数
@optimize(target="gfx1100")
def scaled(x, alpha: float, beta: float):
    return alpha * torch.relu(x) + beta
```

**参数**：
| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `target` | str | `"gfx1100"` | GPU 架构标识 |
| `backend` | str | `"local:..."` | 模型 backend（目前仅本地） |
| `force_recompile` | bool | `False` | 跳过缓存强制重编译 |
| `skeleton` | str/None | `None` | 显式指定骨架：`elementwise_1d` / `row_reduction` / `matmul_2d` / `multi_output` |

---

## 9. 已知局限

| # | 局限 | 影响 | 缓解 |
|---|---|---|---|
| 1 | 大 tensor (numel ≥ 4M) 系统性输给 torch.compile | 无法用于大矩阵/大 batch 优化 | 4B 模型的知识边界，需要更大模型或显式 tiling 模板 |
| 2 | 冷启动慢（1-iter: ~60s, 10-iter: ~530s） | 不适合真正的 JIT 场景 | 依赖缓存；缓存命中后 O(1) 加载 |
| 3 | 不支持动态 shape | shape 变化触发重编译 | cache key 含 shape_sig |
| 4 | 分类器基于源码 regex | lambda / 装饰链 / 跨文件 helper 失效 | `skeleton=...` 显式指定 |
| 5 | 子进程 benchmark 仍有 ~10% 波动 (CV=0.117) | 边界处的达标/未达标判定不稳定 | 已用 median-of-5-rounds 缓解 |
| 6 | 依赖 ROCm 7.x + PyTorch ROCm 构建 | 无法在 CUDA 环境运行 | 项目定位即为 ROCm 专用 |
| 7 | 依赖 `rocprofv3`；缺失时性能反馈降级 | 迭代优化效果下降 | 仍可通过时序反馈迭代 |
| 8 | 模型路径硬编码 | 更换模型需改 core.py | 未来可参数化到 backend 字段 |

---

## 10. 版本历史

| 版本 | 主要变更 |
|---|---|
| v0.1 | 初始实现：单一 elementwise 骨架，写死 float dtype，2 个例子 |
| v0.2 | dtype 动态化、标量参数支持、广播提示、编译错误过滤、MSE 反馈增强 |
| **v1.0** | **三层架构**（分类器 + 4 骨架 + Sanitizer）；**14 个测试例子**覆盖 4 骨架 × 3 dtype × 3 维度；**稳态计时**（median-of-rounds 替代 single-shot）；**dtype-aware MSE 阈值**；**分类器置信度** + 低置信度警告 + 数据依赖检测；**RL vs 基座模型消融对比**（RL +75% 达标率、-59% 编译失败率、+30% 平均 speedup）；**graceful fallback**（10 轮全败保留原 PyTorch）|
