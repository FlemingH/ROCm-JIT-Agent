"""SiLU in bfloat16 — covers the bf16 dtype path.

bf16 is the dominant dtype for LLM inference. The skeleton claims dtype-awareness
via `dtype_to_ctype()`; this test actually exercises that path.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def silu_bf16(x):
    return x * torch.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)

    out = silu_bf16(x)
    expected = torch.nn.functional.silu(x)
    print(f"\n[test] dtype: {out.dtype} expected: {expected.dtype}")
    print(f"[test] output shape: {list(out.shape)}")
    # Looser tolerance for bf16
    diff = (out.float() - expected.float()).abs().max().item()
    mse = (out.float() - expected.float()).pow(2).mean().item()
    print(f"[test] max abs err: {diff:.6f}   mse: {mse:.6f}")
    assert out.dtype == torch.bfloat16, f"dtype mismatch: {out.dtype}"
    assert diff < 5e-2, f"abs err too high (bf16 tolerance): {diff}"

    import triton.testing
    opt_us   = triton.testing.do_bench(lambda: silu_bf16(x)) * 1000
    torch_us = triton.testing.do_bench(lambda: torch.nn.functional.silu(x)) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch F.silu: {torch_us:.1f}us")
    print("[test] ✅ silu_bf16_example passed (bf16 dtype path)")
