"""Row-wise reduction example: sum along the last dim.

Exercises the `row_reduction` skeleton.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def row_sum(x):
    return x.sum(dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(512, 1024, device="cuda", dtype=torch.float32)

    # Warmup / compile
    out = row_sum(x)
    expected = x.sum(dim=-1)
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] output shape: {list(out.shape)} expected: {list(expected.shape)}")
    print(f"[test] MSE vs reference: {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-2, f"MSE too high: {mse}"

    # Bench
    import triton.testing
    opt_us = triton.testing.do_bench(lambda: row_sum(x)) * 1000
    torch_us = triton.testing.do_bench(lambda: x.sum(dim=-1)) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch eager: {torch_us:.1f}us")
    print("[test] ✅ reduction_example passed")
