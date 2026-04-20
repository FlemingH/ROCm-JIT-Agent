"""3D attention pre-softmax score: out = (q * k).sum(...) is too complex.
   Use the simpler 3D elementwise: out = (q * k) * scale  (kept as 3D shape,
   no reduction, to validate the 1D-grid skeleton on 3D tensors via numel()).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def scaled_qk(q, k, scale):
    return (q * k) * scale


if __name__ == "__main__":
    torch.manual_seed(0)
    # [Batch, Heads, Dim] — typical attention shape
    B, H, D = 8, 16, 128
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, D, device="cuda", dtype=torch.float32)
    scale = 1.0 / (D ** 0.5)

    out = scaled_qk(q, k, scale)
    expected = (q * k) * scale
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] input shape: [{B},{H},{D}]  output shape: {list(out.shape)}")
    print(f"[test] scale = {scale:.5f}   MSE = {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-5, f"MSE too high: {mse}"

    import triton.testing
    opt_us   = triton.testing.do_bench(lambda: scaled_qk(q, k, scale)) * 1000
    torch_us = triton.testing.do_bench(lambda: (q * k) * scale) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch eager: {torch_us:.1f}us")
    print("[test] ✅ attention_score_3d_example passed (3D shape + scalar)")
