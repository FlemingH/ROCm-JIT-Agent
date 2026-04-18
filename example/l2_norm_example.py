"""Row-wise L2 norm (GPUMODE / RMSNorm building block).

out[i] = sqrt(sum(x[i]**2, dim=-1))

Exercises the `row_reduction` skeleton with squared-reduce + sqrt finalize
(different math than pure sum, requires per-element transform inside the reduce).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def row_l2_norm(x):
    return torch.sqrt((x * x).sum(dim=-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256, 2048, device="cuda", dtype=torch.float32)

    out = row_l2_norm(x)
    expected = torch.linalg.vector_norm(x, ord=2, dim=-1)
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] output shape: {list(out.shape)} expected: {list(expected.shape)}")
    print(f"[test] MSE vs torch.linalg.vector_norm: {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-2, f"MSE too high: {mse}"

    import triton.testing
    opt_us   = triton.testing.do_bench(lambda: row_l2_norm(x)) * 1000
    torch_us = triton.testing.do_bench(lambda: torch.linalg.vector_norm(x, ord=2, dim=-1)) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch linalg: {torch_us:.1f}us")
    print("[test] ✅ l2_norm_example passed")
