# ROCm-JIT-Agent：AI-in-the-Loop 本地算子锻造引擎 (Kernel Forge)

## 1. 愿景与目标用户 (The Vision & Target Audience)

**“打破云端算力霸权，让端侧开发者在消费级显卡上，就地锻造出压榨硬件极限的底层算子。”**

传统的 AI 算子优化是一个漫长且痛苦的离线过程，需要顶尖的工程师手写 C++/HIP、调试内存对齐、解决寄存器溢出。更致命的是，这些能力往往被锁死在财力雄厚的机房或需要联网调用的超大杯闭源模型（如 GPT-4o）中。

**ROCm-JIT-Agent** 敏锐地切入了**“算力下沉与端侧 AI 爆发”**的蓝海。它的核心产品故事并非面向普通终端消费者的实时加速，而是**专为开源极客、中小型科研团队、边缘计算（Edge AI）部署工程师打造的“本地高性能算子锻造炉”**。

通过 GRPO 强化学习，我们成功将 AMD RDNA 架构的底层汇编智慧，压缩到了一个名为 **`Jan-code-4b-gfx1100-HIP-1`** 的 4B 专精代码模型中。这意味着，任何开发者都能在他们那张单薄的 RX 7900 XTX 甚至 7900 GRE 上，于完全离线、隐私安全的环境下，瞬间生成比肩官方驱动工程师水准的极速 C++ 算子。

### 1.1 极客感十足的交互体验

用户无需离开熟悉的 PyTorch 环境，更不需要了解底层的 C++ 逻辑。只需在性能瓶颈的模块上添加一个简单的装饰器 `@optimize`：

```python
import torch
from rocm_jit_agent import optimize

# 只需要加一行装饰器，并指定目标架构 (如 "gfx1100" 对应 RX 7900 XTX)
@optimize(target="gfx1100", force_recompile=False)
def fused_gated_activation(x, weight, bias):
    """
    一个极其经典的 Memory Bound (访存密集型) 融合算子。
    ROCm-JIT-Agent 会将其融合成一个单纯的 O(1) 访存 HIP C++ 算子。
    """
    return (x * weight) * torch.sigmoid(x) + bias

# 像调用普通的 PyTorch 函数一样调用它！
out = fused_gated_activation(x, weight, bias)
```

**沉浸式的编译战报：**
当该算子被部署或训练脚本**第一次**运行触发时，控制台将弹出一个极具极客风格的代码流式输出（打字机动画），系统在后台与 `torch.compile` 刺刀见红地对比速度，并将大模型现场手写的底层 C++ 内核展示给您：

```text
[START] Triggering LLM for code inference and HIP compilation...
[rocm_jit_agent] 🚀 Intercepted PyTorch operator: fused_gated_activation (Target: gfx1100)
[rocm_jit_agent] 🔥 Waking up Kernel Forge. Baseline Eager: 252.9us | Torch Compile: 136.6us | Target: < 136.6us
[rocm_jit_agent] 🔄 Iteration 1/10 - Thinking & Generating (Temp: 0.30)...
[LLM Streaming] 🤖: #include <torch/extension.h> ... (伴随着炫酷的逐行打字动画输出代码)
[rocm_jit_agent] ⏳ Compiling generated HIP C++ Kernel...
[rocm_jit_agent] ✅ Validation Passed (MSE=0.00000). Opt Execution Time: 34.0us
[rocm_jit_agent] 🏆 Goal Reached! (34.0us <= 136.6us)
[rocm_jit_agent] ✨ Forging Successful! AI HIP Kernel: 34.0us (Speedup 4.0x vs Torch Compile 136.6us) -> Persistent Cache Saved.

[DONE] Operator execution completed!
```
随后的所有调用（甚至跨进程重启后），将以 `O(1)` 的时间复杂度从硬盘直读缓存，达成零延迟的极速启动。

---

## 2. 核心架构设计：四层解耦外壳 (Model-Agnostic Shell)

引擎的外壳部分（`core.py`）设计得极其坚固，它扮演着“冷酷监工”与“安全沙盒”的角色，全权接管模型推理到硬件调优的完整工程管线。

### 2.1 第一层：前端拦截与 AST 签名解析层
*   **职责**：在 Python 运行时动态拦截原生函数，提取上下文。
*   **机制**：通过 `inspect` 提取函数源码的 AST。最关键的是，它会动态抓取传入的 `args` 的张量名称、`shape` 维度、`dtype` 数据类型，作为骨架注入到给 LLM 的指令中。这杜绝了模型因“盲猜”张量大小而写出错误的广播逻辑（Broadcasting）。

### 2.2 第二层：多卡切分与大模型推理层 (Multi-GPU Pipeline)
*   **模型**：采用了经过深度 RL 微调的 `Jan-code-4b-gfx1100-HIP-1`。它被强行戒除了 Nvidia 的 CUDA 幻觉，形成了编写 `<hip/hip_runtime.h>` 与 1D Grid 展开结构的肌肉记忆。
*   **智能分配**：借助 Hugging Face `transformers` 的 `device_map="auto"` 黑科技，系统能自动将这个大模型的权重以流水线并行（Pipeline Parallelism）的方式均匀切分到所有可用的 AMD 加速卡上。在处理超长上下文代码生成时，彻底免除了单卡显存溢出（OOM）的危机。

### 2.3 第三层：沙盒验证与专家级调优状态机
这是整个引擎的“大脑”，也是顶尖算子调优专家的 AI 化克隆：
*   **保姆级 C++ 骨架约束**：外壳并没有让模型自由发挥接口定义。它在 Prompt 中强制生成了基于 PyBind11 和 `load_inline` 所需的最严苛的 `torch::Tensor optimized_func()` C++ 接口签名，杜绝了 `incompatible function arguments` 的低级链接错误。
*   **静默沙盒测速**：所有的正确性跑分（MSE 对齐）和耗时评测均在安全的 Subprocess 沙盒中用虚拟张量执行，如果因为生成的代码逻辑错误导致段错误（Segfault）甚至编译语法错误，主进程会精准捕获 `ninja/hipcc` 异常并重新发回给大模型进行重试修复。
*   **专家经验注入**：如果生成的算子耗时没能跑赢 `torch.compile`，外壳会自动下发包含 `memory coalescing`、`float4 vectorized loads` 等极具压迫感的高阶专家 Prompt；若多次未达标，则会唤醒 `rocprofv3` 提取硬件底层寄存器泄漏数据，进行降维指导。

### 2.4 第四层：O(1) 持久化缓存层 (Persistent Disk Caching)
*   **机制**：一旦最佳内核在角斗场中胜出，外壳会根据原始 Python 代码和目标架构生成唯一的 MD5 哈希（如 `dynamic_hip_1b6554...cpp`）。
*   **零延迟加载**：底层 C++ 源码及签名被硬编码落盘在 `~/.rocm_jit_agent_cache/` 目录下。在未来的任何进程、任何时候调用该算子，只要源码未改动，外壳都将完全绕过大语言模型，利用 PyTorch 的 `load_inline` 在秒级将 C++ 扩展挂载为原生函数直接执行。

---

## 3. 核心运行数据流 (Data Flow)

以下是 ROCm-JIT-Agent 在实际项目中的完整时序流：

1. **触发拦截与基准标定 (Interception & Baseline)**：
   * 用户首次执行带有 `@optimize` 的函数。
   * 系统挂起执行，计算 `Torch Eager` 原生耗时，并随后启动 `torch.compile` 作为终极打靶目标。
2. **流水线推理 (Multi-GPU Inference)**：
   * 唤醒 `gfx1100-HIP-1` 大模型，执行流式的代码生成动画。
3. **沙盒验证与 C++ 编译 (Sandbox Compilation)**：
   * 将模型吐出的代码落盘为临时文件，调起 `ninja` 与 `hipcc` 构建动态共享库。
   * 执行张量运算并强制对齐 MSE 精度，若报错则进入 `Iteration N` 循环重构。
4. **性能榨取评定 (Performance Profiling)**：
   * 对通过正确性验证的内核使用 `torch.cuda.Event` 测速。只有算子耗时小于等于 `torch.compile` 的基准，才会被判定为胜利（Goal Reached）。
5. **本地固化与热替换 (Persistent Cache)**：
   * 将大获全胜的极致 C++ 算子源码存档入硬盘。随后自动放行原生参数的执行，并用此高速通道处理后续所有的训练和推理数据流。

---

## 4. 设计哲学：坚定的工程底线

**为什么是纯粹的 C++ HIP 管线，而不是 Triton？**
在早期开发中，我们曾尝试让大模型输出 OpenAI Triton Python 代码。但我们很快发现，Triton 中复杂的指针隐式转换、块状计算 `tl.load` 和严苛的边界掩码（Mask）使得小模型极易陷入难以自拔的语法死胡同（Hallucination Loop）。
反而，退回到最原教旨主义的 **HIP C++ 底层**，大模型依靠庞大清晰的 C/C++ 语法树和我们施加的“保姆级 C++ 外壳接口模板”，其一次性生成正确率（Zero-shot Accuracy）飙升。凭借着 `hipcc -O3` 编译器的神级指令重排，即使大模型写的 C++ 数学逻辑并不优美，依然能爆发出**碾压官方 `torch.compile` 高达 4 ~ 7 倍**的绝对性能统治力！

**为什么不联网用 API 调用 GPT-4 或 Claude 3.5？**
*   **企业级的隐私隔离**：端侧科研和部署工程师绝对不希望将他们的前沿算法代码及私密模型架构上传至公共网络。
*   **私有化调教的心智**：我们将 4B 的轻量级模型作为核心底座，就是为了证明经过高强度、定向的强化学习（RL），在“写 AMD 底层算子”这一绝对狭窄的切面中，本地 8G 显存的小模型一样能干翻通用万亿大模型。

---

## 5. 总结

ROCm-JIT-Agent (`v0.1`) 完美兑现了它的初衷：**用几十秒钟的一次性编译等待代价，换取大模型全生命周期中无数次调用的几微秒级极速执行**。它是一把插在端侧玩家兵器库里的利刃，让每一个极客都能无需手写一行底层 C++，就能让 AMD 加速卡在复杂前沿算法中跑出令人生畏的带宽上限。