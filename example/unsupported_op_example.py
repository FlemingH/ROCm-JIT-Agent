"""NEGATIVE probe: an op that no current skeleton fits.
torch.cumsum has a sequential data dependency along the reduction axis -
neither elementwise_1d nor row_reduction skeletons can express it correctly.
Expected outcome documents the FAILURE MODE of the system:
  - classifier picks the wrong skeleton (no negative signal)
  - all 10 iterations fail validation
  - no graceful fallback; user sees a long retry loop then exception
This is exactly what we want to surface."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rocm_jit_agent import optimize

@optimize(target="gfx1100", force_recompile=True)
def row_cumsum(x):
    return torch.cumsum(x, dim=-1)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(32, 128, device="cuda")
    try:
        y = row_cumsum(x)
        ref = torch.cumsum(x, dim=-1)
        diff = (y - ref).abs().max().item()
        print(f"[test] UNEXPECTED success: max_abs_err={diff:.4f}")
        if diff < 1e-3:
            print("[test] system handled cumsum (skeleton coverage extended?)")
        else:
            print("[test] returned wrong numerics silently — VALIDATION ESCAPE BUG")
    except Exception as e:
        print(f"[test] EXPECTED failure surfaced: {type(e).__name__}: {str(e)[:200]}")
        print("[test] unsupported_op_example finished (failure mode documented)")
