"""Multi-output example: returns (sum, diff) as tuple.

Exercises the `multi_output` skeleton.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def sum_and_diff(x, y):
    return x + y, x - y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    y = torch.randn(4096, device="cuda", dtype=torch.float32)

    s, d = sum_and_diff(x, y)
    es, ed = x + y, x - y
    mse_s = torch.nn.functional.mse_loss(s, es).item()
    mse_d = torch.nn.functional.mse_loss(d, ed).item()
    print(f"\n[test] output shapes: sum={list(s.shape)} diff={list(d.shape)}")
    print(f"[test] MSE sum: {mse_s:.6f} | MSE diff: {mse_d:.6f}")
    assert mse_s < 1e-5 and mse_d < 1e-5, f"sum MSE={mse_s} diff MSE={mse_d}"

    import triton.testing
    opt_us = triton.testing.do_bench(lambda: sum_and_diff(x, y)) * 1000
    torch_us = triton.testing.do_bench(lambda: (x+y, x-y)) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch eager: {torch_us:.1f}us")
    print("[test] ✅ multi_output_example passed")
