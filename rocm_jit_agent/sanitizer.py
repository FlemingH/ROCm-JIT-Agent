"""Deterministic code post-processing for LLM-generated HIP C++.

Fixes systematic model bugs observed in experiments/probe_boundaries.py:
- CUDA-only headers (cuda_runtime.h / cuda.h)
- Trailing markdown ``` fence
- CUDA <<<grid,block>>> launch syntax
"""
import re

# CUDA-only headers that break hipcc
CUDA_HEADERS = [
    "#include <cuda_runtime.h>",
    "#include <cuda.h>",
    "#include <device_launch_parameters.h>",
    "#include <cuda_runtime_api.h>",
]

# Match:  kernel_name<<<grid, block>>>(args);  also with stream.
_LAUNCH_RE = re.compile(
    r"(\w+)\s*<<<\s*([^,]+?)\s*,\s*([^,>]+?)\s*>>>\s*\(([^;]*?)\)\s*;",
    re.DOTALL,
)


def sanitize(code: str):
    """Apply deterministic fixes. Returns (cleaned_code, patches_applied)."""
    patches = []

    # 1. Strip CUDA-only headers
    for hdr in CUDA_HEADERS:
        if hdr in code:
            code = code.replace(hdr, f"// {hdr}  // stripped by sanitizer (ROCm)")
            patches.append(f"stripped {hdr}")

    # 2. Strip markdown fences that leak into code
    lines = code.split("\n")
    cleaned = []
    fence_dropped = 0
    for ln in lines:
        if ln.strip().startswith("```"):
            fence_dropped += 1
            continue
        cleaned.append(ln)
    if fence_dropped:
        patches.append(f"stripped {fence_dropped} markdown fence line(s)")
    code = "\n".join(cleaned)

    # 3. Rewrite <<<>>> launches to hipLaunchKernelGGL
    def _repl(m):
        patches.append(f"rewrote <<<>>> launch of {m.group(1)}")
        return (
            f"hipLaunchKernelGGL({m.group(1)}, {m.group(2).strip()}, "
            f"{m.group(3).strip()}, 0, 0, {m.group(4).strip()});"
        )

    code = _LAUNCH_RE.sub(_repl, code)

    return code, patches
