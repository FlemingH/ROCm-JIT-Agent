"""Probe: runtime stability + cache repeatability.
Runs the same compiled kernel many times to expose:
  - Are reported speedups stable across runs, or single-shot noise?
  - Does the persistent cache reload cleanly?
Uses example_fusion (known-good, 1-iter compile)."""
import sys, os, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rocm_jit_agent import optimize

@optimize(target="gfx1100", force_recompile=False)  # use cache if present
def fused(a, b, c):
    return torch.relu(a * b + c)

def bench(fn, args, n=200):
    for _ in range(20): fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n  # us

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = torch.randn(1024, 1024, device="cuda")

    # Trigger compile / cache load
    _ = fused(a, b, c)

    samples = [bench(fused, (a, b, c)) for _ in range(5)]
    eager_samples = [bench(lambda x,y,z: torch.relu(x*y+z), (a,b,c)) for _ in range(5)]
    print(f"[test] optimized us per call: {samples}")
    print(f"[test] eager     us per call: {eager_samples}")
    print(f"[test] opt  median={statistics.median(samples):.1f} stdev={statistics.pstdev(samples):.2f}")
    print(f"[test] eag  median={statistics.median(eager_samples):.1f} stdev={statistics.pstdev(eager_samples):.2f}")
    cv = statistics.pstdev(samples) / statistics.median(samples)
    print(f"[test] coefficient_of_variation={cv:.3f}  (>0.10 means single-shot timings are unreliable)")
    print("[test] stability_test finished")
