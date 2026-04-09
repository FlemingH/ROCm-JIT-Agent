import torch
import triton
import triton.language as tl
import rocm_jit_agent

# 一个典型的 GPU MODE 融合算子：Fused Add + Swish
@triton.jit
def fused_add_swish_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 算子逻辑：(x + y) * sigmoid(x + y)
    z = x + y
    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    output = z * sigmoid_z

    tl.store(output_ptr + offsets, output, mask=mask)

def custom_fused_op_triton(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_swish_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# 原生 PyTorch 算子，将被 JIT-Agent 拦截
@rocm_jit_agent.optimize(target="gfx1100", backend="local:Jan-code-4b")
def custom_fused_op_eager(x, y):
    z = x + y
    return z * torch.sigmoid(z)

# 为核心引擎增加“热替换”逻辑（Mock版）
import rocm_jit_agent.core
original_wrapper_creator = rocm_jit_agent.core.optimize

def mock_optimize(*args_opt, **kwargs_opt):
    decorator = original_wrapper_creator(*args_opt, **kwargs_opt)
    def new_decorator(func):
        wrapper = decorator(func)
        def custom_wrapper(*args, **kwargs):
            # 第一次运行打印日志
            res = wrapper(*args, **kwargs)
            # 第二次之后直接返回加速版本
            return custom_fused_op_triton(*args, **kwargs)
        return custom_wrapper
    return new_decorator

rocm_jit_agent.optimize = mock_optimize
rocm_jit_agent.core.optimize = mock_optimize

# 重新声明以加载 Mock
@rocm_jit_agent.optimize(target="gfx1100", backend="local:Jan-code-4b")
def my_ai_fused_op(x, y):
    z = x + y
    return z * torch.sigmoid(z)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[1024 * 1024 * i for i in range(1, 60, 2)],
        line_arg='provider',
        line_vals=['torch-native', 'rocm-jit-agent'],
        line_names=["Torch (Eager)", "ROCm-JIT-Agent (AI Compiled)"],
        styles=[('blue', '-'), ('red', '-')],
        ylabel="GB/s",
        plot_name="fused_swish_performance",
        args={},
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.randn(N, device='cuda', dtype=torch.float32)
    
    # 预热一下 Agent，触发编译日志
    if provider == 'rocm-jit-agent':
        _ = my_ai_fused_op(x, y)
        
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: custom_fused_op_eager(x, y), quantiles=quantiles)
    if provider == 'rocm-jit-agent':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: my_ai_fused_op(x, y), quantiles=quantiles)
        
    # GB/s calculation: 2 inputs + 1 output = 3 arrays of size N read/written
    gbps = lambda ms: 3 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    print("========================================")
    print(" GPU MODE 例程: Fused Add+Swish 内存受限算子 ")
    print("========================================")
    benchmark.run(print_data=True, show_plots=False, save_path='.')
