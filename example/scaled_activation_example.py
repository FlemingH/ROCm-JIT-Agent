"""Scaled activation: alpha * relu(x) + beta — covers scalar (float) parameters.

Verifies the @optimize decorator separates scalar args correctly and the
generated kernel passes them by value.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def scaled_relu(x, alpha, beta):
    return alpha * torch.relu(x) + beta


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, 4096, device="cuda", dtype=torch.float32)
    alpha = 1.5
    beta = 0.25

    out = scaled_relu(x, alpha, beta)
    expected = alpha * torch.relu(x) + beta
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] output shape: {list(out.shape)}   alpha={alpha}  beta={beta}")
    print(f"[test] MSE: {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-4, f"MSE too high: {mse}"

    import triton.testing
    opt_us   = triton.testing.do_bench(lambda: scaled_relu(x, alpha, beta)) * 1000
    torch_us = triton.testing.do_bench(lambda: alpha * torch.relu(x) + beta) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch eager: {torch_us:.1f}us")
    print("[test] ✅ scaled_activation_example passed (scalar params)")
