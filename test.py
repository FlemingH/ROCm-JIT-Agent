import torch
import rocm_jit_agent

@rocm_jit_agent.optimize(target="gfx1100", backend="local:Jan-code-4b")
def custom_swish(x, weight, bias):
    return torch.sigmoid(x) * weight + bias

if __name__ == "__main__":
    print("========================================")
    print(" 开始 ROCm-JIT-Agent 流程测试")
    print("========================================")
    
    N = 4096
    x = torch.randn(1024, N, dtype=torch.float32, device='cuda')
    w = torch.randn(N, dtype=torch.float32, device='cuda')
    b = torch.randn(N, dtype=torch.float32, device='cuda')
    
    # 第一次运行，触发 JIT 编译和模型加载
    print("\n>>> 第一次调用 custom_swish (预期触发模型伴随编译):")
    out1 = custom_swish(x, w, b)
    
    # 第二次运行，应该是零延迟（直接走缓存）
    print("\n>>> 第二次调用 custom_swish (预期零延迟命中缓存):")
    out2 = custom_swish(x, w, b)
    
    # 验证两次结果的一致性
    print(f"\n[验证] 两次输出的 MSE 误差: {torch.nn.functional.mse_loss(out1, out2).item()}")
    print("测试流程执行完毕！")
