"""Probe: fp16 (half) path. bf16 already covered; fp16 uses c10::Half which
has different API surface and triggers a different family of model bugs.
Same op shape as silu_bf16 to keep the only varying axis = dtype."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rocm_jit_agent import optimize

@optimize(target="gfx1100", force_recompile=True)
def silu_fp16(x):
    return x * torch.sigmoid(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.float16)
    y = silu_fp16(x)
    ref = x * torch.sigmoid(x)
    diff = (y.float() - ref.float()).abs().max().item()
    print(f"[test] dtype={y.dtype} shape={list(y.shape)} max_abs_err={diff:.4f}")
    assert diff < 5e-2, f"fp16 abs err too large: {diff}"
    print("[test] fp16_example passed")
