import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocm_jit_agent import optimize
import time

# Use force_recompile=True to bypass cache and trigger the full JIT + Profiling loop
@optimize(target="gfx1100", force_recompile=True)
def complex_math_fusion(x, y, weight, bias):
    """
    A highly intensive operator that combines element-wise multiplication,
    addition, exponential functions, and logarithmic scaling.
    This creates significant register pressure and tests the optimizer's ability
    to utilize rocprofv3 hardware counters efficiently for targeted tuning.
    """
    # Complex gated activation
    gate = torch.sigmoid(x * weight + bias)
    # Polynomial expansion with logarithmic scaling
    scaled_y = torch.log(1.0 + torch.abs(y)) * (x ** 2)
    # Final dense fusion
    return gate * scaled_y + torch.tanh(x - y)

if __name__ == "__main__":
    BATCH_SEQ = 1024
    HIDDEN_DIM = 4096
    
    print(f"[*] Initializing complex test tensors... (Shape: {BATCH_SEQ} x {HIDDEN_DIM})")
    x = torch.randn(BATCH_SEQ, HIDDEN_DIM, dtype=torch.float32, device='cuda')
    y = torch.randn(BATCH_SEQ, HIDDEN_DIM, dtype=torch.float32, device='cuda')
    weight = torch.randn(HIDDEN_DIM, dtype=torch.float32, device='cuda')
    bias = torch.randn(HIDDEN_DIM, dtype=torch.float32, device='cuda')
    
    print("\n[START] Triggering LLM for profiling-guided HIP compilation...")
    start_time = time.time()
    out = complex_math_fusion(x, y, weight, bias)
    duration = time.time() - start_time
    
    print(f"\n[DONE] Operator execution completed! Total time: {duration:.2f} s")
