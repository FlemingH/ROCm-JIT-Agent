"""Small matrix multiplication example.

Exercises the `matmul_2d` skeleton. Note: naive kernel won't beat rocBLAS,
we just verify correctness + end-to-end pluggable flow.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def small_matmul(A, B):
    return A @ B


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 64, device="cuda", dtype=torch.float32)

    out = small_matmul(A, B)
    expected = A @ B
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] output shape: {list(out.shape)} expected: {list(expected.shape)}")
    print(f"[test] MSE vs reference: {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-1, f"MSE too high: {mse}"

    import triton.testing
    opt_us = triton.testing.do_bench(lambda: small_matmul(A, B)) * 1000
    torch_us = triton.testing.do_bench(lambda: A @ B) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch eager (rocBLAS): {torch_us:.1f}us")
    print("[test] ✅ matmul_example passed (correctness)")
