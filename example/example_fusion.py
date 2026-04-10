import time
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rocm_jit_agent import optimize

# Just add one line of decorator and specify the target architecture (e.g., "gfx1100" for RX 7900 XTX)
# Enable force_recompile=True to force the LLM to regenerate and compile the kernel, bypassing local persistent cache
@optimize(target="gfx1100", force_recompile=True)
def fused_gated_activation(x, weight, bias):
    return (x * weight) * torch.sigmoid(x) + bias

if __name__ == "__main__":
    # Simulate typical tensor dimensions found in Large Language Models (LLMs)
    BATCH_SEQ = 1024
    HIDDEN_DIM = 4096
    
    print(f"[*] Initializing test tensors... (Shape: {BATCH_SEQ} x {HIDDEN_DIM})")
    x = torch.randn(BATCH_SEQ, HIDDEN_DIM, dtype=torch.float32, device='cuda')
    weight = torch.randn(HIDDEN_DIM, dtype=torch.float32, device='cuda')
    bias = torch.randn(HIDDEN_DIM, dtype=torch.float32, device='cuda')
    
    # Just call it like a regular PyTorch function, and the LLM will automatically intercept and optimize it in the background!
    print("\n[START] Triggering LLM for code inference and HIP compilation...")
    start_time = time.time()
    out = fused_gated_activation(x, weight, bias)
    duration = time.time() - start_time
    
    print(f"\n[DONE] Operator execution completed! Total time for compilation+inference+execution: {duration:.2f} s")
