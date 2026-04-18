"""SwiGLU activation (Llama/PaLM classic).

SwiGLU(x, gate, up) = (x * sigmoid(gate)) * up

Exercises the `elementwise_1d` skeleton with 3-tensor fusion
(the gated linear unit pattern that dominates modern LLM FFN blocks).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rocm_jit_agent import optimize


@optimize(target="gfx1100", force_recompile=True)
def swiglu(x, gate, up):
    return (x * torch.sigmoid(gate)) * up


if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (1024, 4096)
    x    = torch.randn(*shape, device="cuda", dtype=torch.float32)
    gate = torch.randn(*shape, device="cuda", dtype=torch.float32)
    up   = torch.randn(*shape, device="cuda", dtype=torch.float32)

    out = swiglu(x, gate, up)
    expected = (x * torch.sigmoid(gate)) * up
    mse = torch.nn.functional.mse_loss(out, expected).item()
    print(f"\n[test] output shape: {list(out.shape)}")
    print(f"[test] MSE vs reference: {mse:.6f}")
    assert out.shape == expected.shape
    assert mse < 1e-4, f"MSE too high: {mse}"

    import triton.testing
    opt_us   = triton.testing.do_bench(lambda: swiglu(x, gate, up)) * 1000
    torch_us = triton.testing.do_bench(lambda: (x * torch.sigmoid(gate)) * up) * 1000
    print(f"[test] optimized: {opt_us:.1f}us | torch eager: {torch_us:.1f}us")
    print("[test] ✅ swiglu_example passed")
