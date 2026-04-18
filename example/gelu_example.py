"""GELU activation (GPUMODE classic).

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Exercises the `elementwise_1d` skeleton with math-intrinsic-heavy body.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def gelu_tanh_approx(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


if __name__ == "__main__":
    torch.manual_seed(0)
    # Typical transformer activation shape
    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float32)

    out = gelu_tanh_approx(x)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] output shape: {list(out.shape)}")
    print(f"[test] MSE vs F.gelu(approximate='tanh'): {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-4, f"MSE too high: {mse}"

    import triton.testing
    opt_us    = triton.testing.do_bench(lambda: gelu_tanh_approx(x)) * 1000
    torch_us  = triton.testing.do_bench(lambda: torch.nn.functional.gelu(x, approximate="tanh")) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch F.gelu: {torch_us:.1f}us")
    print("[test] ✅ gelu_example passed")
