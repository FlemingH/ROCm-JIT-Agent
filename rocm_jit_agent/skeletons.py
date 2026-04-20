"""Pluggable skeletons for different kernel patterns.

Each Skeleton knows how to:
- Build the skeleton C++ code embedded in the LLM prompt (with concrete dtypes/shapes).
- Build the eval/comparison code used in the subprocess validator.
- Declare the C++ return type and regex for function-signature extraction.

Classifier heuristically picks a skeleton from the PyTorch source code.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import re
import textwrap
import torch


# ---------- dtype helpers ----------
_DTYPE_TO_CTYPE = {
    torch.float32: "float", torch.float64: "double",
    torch.float16: "at::Half", torch.bfloat16: "at::BFloat16",
    torch.int32: "int", torch.int64: "long", torch.int16: "short",
    torch.int8: "int8_t", torch.uint8: "uint8_t", torch.bool: "bool",
}

def dtype_to_ctype(dtype):
    return _DTYPE_TO_CTYPE.get(dtype, "float")

def scalar_to_ctype(val):
    if isinstance(val, bool): return "bool"
    if isinstance(val, int): return "int"
    if isinstance(val, float): return "float"
    return "float"


# ---------- skeleton record ----------
@dataclass
class SkeletonContext:
    """All the info a skeleton needs to render prompts & eval."""
    func_name: str
    source_code: str                    # original python, with @optimize stripped
    target: str                         # 'gfx1100'
    tensor_args: List[Tuple[str, torch.Tensor]]  # (name, tensor)
    scalar_args: List[Tuple[str, object]]        # (name, value)
    arg_names: List[str]                # full ordered python arg names


@dataclass
class Skeleton:
    name: str
    cpp_return_type: str                # "torch::Tensor" | "std::vector<torch::Tensor>"
    n_outputs: int                      # for multi-output: number of returned tensors
    build_prompt_block: Callable[[SkeletonContext, "Skeleton"], str]
    build_eval_compare: Callable[[SkeletonContext, "Skeleton"], str]  # python code that compares out_eager vs out_opt
    signature_regex: str                # regex to pull `optimized_func` prototype out of generated code
    description: str = ""


# ============================================================
# Helper: build common C++ function signature + launch pointers
# ============================================================
def _common_signature(ctx: SkeletonContext):
    cpp_tensor_args = [f"torch::Tensor {n}" for n, _ in ctx.tensor_args]
    cpp_scalar_args = [f"{scalar_to_ctype(v)} {n}" for n, v in ctx.scalar_args]
    cpp_arg_str = ", ".join(cpp_tensor_args + cpp_scalar_args)

    kernel_ptr_parts = []
    launch_ptr_parts = []
    for n, t in ctx.tensor_args:
        ct = dtype_to_ctype(t.dtype)
        kernel_ptr_parts.append(f"const {ct}* {n}_ptr")
        launch_ptr_parts.append(f"{n}.data_ptr<{ct}>()")
    for n, v in ctx.scalar_args:
        kernel_ptr_parts.append(f"{scalar_to_ctype(v)} {n}")
        launch_ptr_parts.append(n)
    return cpp_arg_str, ", ".join(kernel_ptr_parts), ", ".join(launch_ptr_parts)


def _tensor_info_line(ctx):
    return ", ".join(f"{n}({list(t.shape)}, {t.dtype})" for n, t in ctx.tensor_args)


# ============================================================
# Skeleton 1: elementwise_1d (default — current behaviour)
# ============================================================
def _elem_prompt(ctx, sk):
    cpp_arg_str, kernel_ptr_args, launch_data_ptrs = _common_signature(ctx)
    first_tensor = ctx.tensor_args[0][0] if ctx.tensor_args else "x"
    first_ct = dtype_to_ctype(ctx.tensor_args[0][1].dtype) if ctx.tensor_args else "float"
    broadcast_hints = []
    for n, t in ctx.tensor_args:
        if t.dim() == 1:
            broadcast_hints.append(f"  - {n} has shape {list(t.shape)}: use `pid % {t.shape[-1]}` to index it")
    broadcast_hint = "\n".join(broadcast_hints) if broadcast_hints else ""
    tinfo = _tensor_info_line(ctx)

    return textwrap.dedent(f"""\
        Convert the following PyTorch code to a highly optimized HIP C++ kernel for AMD GPU ({ctx.target}).
        Task class: ELEMENTWISE_1D (one thread computes one output element).
        Requirements:
        1. Return ONLY valid C++ code.
        2. Implement the math correctly based on the tensor shapes.
        Tensor signatures: {tinfo}
        3. IMPORTANT: For 1D tensors, broadcast using modulo indexing.
        {broadcast_hint}
        4. You MUST implement `torch::Tensor optimized_func(...)` with EXACTLY these arguments: {cpp_arg_str}.
        5. Use this skeleton pattern:

        ```cpp
        #include <torch/extension.h>
        #include <hip/hip_runtime.h>

        __global__ void fused_kernel({first_ct}* output_ptr, {kernel_ptr_args}, int n_elements) {{
            int pid = blockIdx.x * blockDim.x + threadIdx.x;
            if (pid < n_elements) {{
                // DO MATH HERE
            }}
        }}

        torch::Tensor optimized_func({cpp_arg_str}) {{
            auto output = torch::empty_like({first_tensor});
            int n_elements = output.numel();
            int threads = 256;
            int blocks = (n_elements + threads - 1) / threads;
            hipLaunchKernelGGL(fused_kernel, dim3(blocks), dim3(threads), 0, 0,
                output.data_ptr<{first_ct}>(), {launch_data_ptrs}, n_elements);
            return output;
        }}
        ```

        Original code:
        ```python
        {ctx.source_code}
        ```
    """)

def _single_tensor_compare(ctx, sk):
    return textwrap.dedent("""\
        if not (isinstance(out_eager, torch.Tensor) and isinstance(out_opt, torch.Tensor)):
            print(f"ERROR: expected single tensor outputs, got {type(out_eager)} vs {type(out_opt)}"); sys.exit(1)
        if out_eager.shape != out_opt.shape:
            print(f"ERROR: shape mismatch {tuple(out_eager.shape)} vs {tuple(out_opt.shape)}"); sys.exit(1)
        mse = torch.nn.functional.mse_loss(out_eager.float(), out_opt.float()).item()
        if mse > MSE_THRESH:
            fe = out_eager.flatten()[:5].tolist()
            fo = out_opt.flatten()[:5].tolist()
            print(f"ERROR: MSE={mse:.5f} exceeds threshold {MSE_THRESH}. Expected first 5: {fe}, got: {fo}.")
            sys.exit(1)
    """)


ELEMENTWISE_1D = Skeleton(
    name="elementwise_1d",
    cpp_return_type="torch::Tensor",
    n_outputs=1,
    build_prompt_block=_elem_prompt,
    build_eval_compare=_single_tensor_compare,
    signature_regex=r"(torch::Tensor\s+optimized_func\s*\([^)]*\))",
    description="1D grid; one thread per output element; output shape = input[0].shape.",
)


# ============================================================
# Skeleton 2: row_reduction
# ============================================================
def _reduce_prompt(ctx, sk):
    cpp_arg_str, _, _ = _common_signature(ctx)
    # Assume reduction along last axis. First tensor must be 2D.
    first_name, first_t = ctx.tensor_args[0]
    first_ct = dtype_to_ctype(first_t.dtype)
    rows = first_t.shape[0] if first_t.dim() >= 1 else 0
    cols = first_t.shape[-1] if first_t.dim() >= 2 else first_t.shape[0]
    tinfo = _tensor_info_line(ctx)

    return textwrap.dedent(f"""\
        Convert the following PyTorch code to a highly optimized HIP C++ kernel for AMD GPU ({ctx.target}).
        Task class: ROW_REDUCTION (reduce along last axis, one block per row).
        Tensor signatures: {tinfo}
        Input {first_name} has shape [{rows}, {cols}]; output shape is [{rows}] with dtype {first_t.dtype}.

        Requirements:
        1. Return ONLY valid C++ code.
        2. Implement `torch::Tensor optimized_func({cpp_arg_str})`.
        3. Launch one block per row; each block reduces `cols` elements.
        4. Use this skeleton:

        ```cpp
        #include <torch/extension.h>
        #include <hip/hip_runtime.h>

        __global__ void fused_kernel({first_ct}* out_ptr, const {first_ct}* {first_name}_ptr, int rows, int cols) {{
            int row = blockIdx.x;
            if (row < rows) {{
                // TODO: reduce {first_name}_ptr[row*cols .. row*cols+cols] into out_ptr[row]
                // You MAY use single-thread loop (simple) OR shared memory reduction (fast).
            }}
        }}

        torch::Tensor optimized_func({cpp_arg_str}) {{
            int rows = {first_name}.size(0);
            int cols = {first_name}.size(-1);
            auto out = torch::empty({{rows}}, {first_name}.options());
            hipLaunchKernelGGL(fused_kernel, dim3(rows), dim3(1), 0, 0,
                out.data_ptr<{first_ct}>(), {first_name}.data_ptr<{first_ct}>(), rows, cols);
            return out;
        }}
        ```

        Original code:
        ```python
        {ctx.source_code}
        ```
    """)

ROW_REDUCTION = Skeleton(
    name="row_reduction",
    cpp_return_type="torch::Tensor",
    n_outputs=1,
    build_prompt_block=_reduce_prompt,
    build_eval_compare=_single_tensor_compare,
    signature_regex=r"(torch::Tensor\s+optimized_func\s*\([^)]*\))",
    description="One block per row; reduces along last axis; output shape = input[0].shape[:-1].",
)


# ============================================================
# Skeleton 3: matmul_2d
# ============================================================
def _matmul_prompt(ctx, sk):
    cpp_arg_str, _, _ = _common_signature(ctx)
    if len(ctx.tensor_args) < 2:
        return _elem_prompt(ctx, sk)  # fall back
    (aname, A), (bname, B) = ctx.tensor_args[0], ctx.tensor_args[1]
    ct = dtype_to_ctype(A.dtype)
    M, K = A.shape[0], A.shape[-1]
    _, N = B.shape[0], B.shape[-1]
    tinfo = _tensor_info_line(ctx)

    return textwrap.dedent(f"""\
        Convert the following PyTorch code to a highly optimized HIP C++ kernel for AMD GPU ({ctx.target}).
        Task class: MATMUL_2D (C = A @ B, 2D thread grid; one thread computes one C[i,j]).
        Tensor signatures: {tinfo}
        Output shape [{M}, {N}] with dtype {A.dtype}. Row-major storage.

        Requirements:
        1. Return ONLY valid C++ code.
        2. Implement `torch::Tensor optimized_func({cpp_arg_str})`.
        3. Use a 2D grid with block size (16, 16).
        4. Skeleton:

        ```cpp
        #include <torch/extension.h>
        #include <hip/hip_runtime.h>

        __global__ void fused_kernel({ct}* C_ptr, const {ct}* {aname}_ptr, const {ct}* {bname}_ptr,
                                     int M, int N, int K) {{
            int i = blockIdx.y * blockDim.y + threadIdx.y;
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < M && j < N) {{
                // TODO: acc = sum_k {aname}_ptr[i*K + k] * {bname}_ptr[k*N + j]; C_ptr[i*N+j] = acc;
            }}
        }}

        torch::Tensor optimized_func({cpp_arg_str}) {{
            int M = {aname}.size(0);
            int K = {aname}.size(-1);
            int N = {bname}.size(-1);
            auto C = torch::empty({{M, N}}, {aname}.options());
            dim3 blk(16, 16);
            dim3 grd((N + 15) / 16, (M + 15) / 16);
            hipLaunchKernelGGL(fused_kernel, grd, blk, 0, 0,
                C.data_ptr<{ct}>(), {aname}.data_ptr<{ct}>(), {bname}.data_ptr<{ct}>(), M, N, K);
            return C;
        }}
        ```

        Original code:
        ```python
        {ctx.source_code}
        ```
    """)

def _matmul_compare(ctx, sk):
    # looser threshold for matmul since naive accumulation has more fp drift
    return textwrap.dedent("""\
        if out_eager.shape != out_opt.shape:
            print(f"ERROR: shape mismatch {tuple(out_eager.shape)} vs {tuple(out_opt.shape)}"); sys.exit(1)
        mse = torch.nn.functional.mse_loss(out_eager.float(), out_opt.float()).item()
        if mse > MATMUL_MSE_THRESH:
            fe = out_eager.flatten()[:5].tolist()
            fo = out_opt.flatten()[:5].tolist()
            print(f"ERROR: MSE={mse:.5f} exceeds matmul threshold {MATMUL_MSE_THRESH}. Expected first 5: {fe}, got: {fo}.")
            sys.exit(1)
    """)

MATMUL_2D = Skeleton(
    name="matmul_2d",
    cpp_return_type="torch::Tensor",
    n_outputs=1,
    build_prompt_block=_matmul_prompt,
    build_eval_compare=_matmul_compare,
    signature_regex=r"(torch::Tensor\s+optimized_func\s*\([^)]*\))",
    description="2D grid (16x16 blocks); naive row-major matmul; output [M,N].",
)


# ============================================================
# Skeleton 4: multi_output (elementwise base)
# ============================================================
def _multi_out_prompt(ctx, sk):
    cpp_arg_str, kernel_ptr_args, launch_data_ptrs = _common_signature(ctx)
    first_tensor = ctx.tensor_args[0][0] if ctx.tensor_args else "x"
    first_ct = dtype_to_ctype(ctx.tensor_args[0][1].dtype) if ctx.tensor_args else "float"
    n_out = sk.n_outputs
    out_ptrs_decl = ", ".join(f"{first_ct}* out{i}_ptr" for i in range(n_out))
    out_allocs = "\n            ".join(f"auto out{i} = torch::empty_like({first_tensor});" for i in range(n_out))
    out_launch_ptrs = ", ".join(f"out{i}.data_ptr<{first_ct}>()" for i in range(n_out))
    out_return = "{" + ", ".join(f"out{i}" for i in range(n_out)) + "}"
    tinfo = _tensor_info_line(ctx)

    return textwrap.dedent(f"""\
        Convert the following PyTorch code to a highly optimized HIP C++ kernel for AMD GPU ({ctx.target}).
        Task class: MULTI_OUTPUT_ELEMENTWISE (1D grid; one kernel writes {n_out} output tensors).
        Tensor signatures: {tinfo}

        Requirements:
        1. Return ONLY valid C++ code.
        2. Implement `std::vector<torch::Tensor> optimized_func({cpp_arg_str})` returning {n_out} tensors in the same order as the Python `return` statement.
        3. Use this skeleton:

        ```cpp
        #include <torch/extension.h>
        #include <hip/hip_runtime.h>
        #include <vector>

        __global__ void fused_kernel({out_ptrs_decl}, {kernel_ptr_args}, int n_elements) {{
            int pid = blockIdx.x * blockDim.x + threadIdx.x;
            if (pid < n_elements) {{
                // DO MATH HERE: write to out0_ptr[pid], out1_ptr[pid], ...
            }}
        }}

        std::vector<torch::Tensor> optimized_func({cpp_arg_str}) {{
            {out_allocs}
            int n_elements = out0.numel();
            int threads = 256;
            int blocks = (n_elements + threads - 1) / threads;
            hipLaunchKernelGGL(fused_kernel, dim3(blocks), dim3(threads), 0, 0,
                {out_launch_ptrs}, {launch_data_ptrs}, n_elements);
            return {out_return};
        }}
        ```

        Original code:
        ```python
        {ctx.source_code}
        ```
    """)

def _multi_out_compare(ctx, sk):
    return textwrap.dedent("""\
        # out_eager is a tuple/list (from python); out_opt is a list from the C++ vector.
        if isinstance(out_eager, torch.Tensor):
            out_eager = [out_eager]
        if isinstance(out_opt, torch.Tensor):
            out_opt = [out_opt]
        if len(out_eager) != len(out_opt):
            print(f"ERROR: output count mismatch: eager={len(out_eager)} opt={len(out_opt)}"); sys.exit(1)
        for i, (oe, oo) in enumerate(zip(out_eager, out_opt)):
            if oe.shape != oo.shape:
                print(f"ERROR: output {i} shape mismatch {tuple(oe.shape)} vs {tuple(oo.shape)}"); sys.exit(1)
            mse = torch.nn.functional.mse_loss(oe.float(), oo.float()).item()
            if mse > MSE_THRESH:
                fe = oe.flatten()[:5].tolist()
                fo = oo.flatten()[:5].tolist()
                print(f"ERROR: output {i} MSE={mse:.5f} exceeds threshold. Expected first 5: {fe}, got: {fo}."); sys.exit(1)
    """)

MULTI_OUTPUT = Skeleton(
    name="multi_output",
    cpp_return_type="std::vector<torch::Tensor>",
    n_outputs=0,      # set dynamically
    build_prompt_block=_multi_out_prompt,
    build_eval_compare=_multi_out_compare,
    signature_regex=r"(std::vector<torch::Tensor>\s+optimized_func\s*\([^)]*\))",
    description="Elementwise 1D grid returning multiple tensors as std::vector.",
)


# ============================================================
# Classifier: pick skeleton from source + call info
# ============================================================
def _count_return_values(source: str) -> int:
    """Return how many tensors the python function returns (best-effort)."""
    m = re.search(r"return\s+([^\n]+)", source)
    if not m:
        return 1
    rhs = m.group(1).strip().rstrip(",")
    # e.g. "return a, b" or "return (a, b)" or "return {a:..}"
    if rhs.startswith("(") and rhs.endswith(")"):
        rhs = rhs[1:-1]
    # crude split on top-level commas (not inside parens/brackets)
    depth = 0
    parts = []
    cur = ""
    for ch in rhs:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return max(1, len(parts))


# Source-code patterns whose semantics no current skeleton can express
# (sequential data dependency along a reduction axis, indirect indexing, etc.).
# When seen inside an otherwise-elementwise function, confidence drops sharply.
_DATA_DEP_PATTERNS = (
    r"torch\.cumsum\b", r"torch\.cumprod\b", r"torch\.cummax\b", r"torch\.cummin\b",
    r"\.cumsum\s*\(", r"\.cumprod\s*\(",
    r"torch\.sort\b", r"torch\.argsort\b", r"torch\.unique\b",
    r"torch\.scatter\b", r"torch\.gather\b", r"\.scatter_?\s*\(", r"\.gather\s*\(",
    r"torch\.nonzero\b", r"torch\.where\s*\(\s*[^,]+\s*\)",  # 1-arg where = nonzero
)


def classify(ctx: SkeletonContext, user_hint: Optional[str] = None):
    """Pick a skeleton. Returns (skeleton, confidence, reason).

    confidence in [0,1]:
      1.00 - explicit user hint
      0.90 - strong structural match (multi-output / matmul w/ 2D / reduction-with-dim)
      0.50 - default fall-through (no positive signal)
      <=0.30 - default fall-through but source contains data-dependency patterns
              that no current skeleton supports (silent-correctness risk).
    """
    name_map = {
        "elementwise_1d": ELEMENTWISE_1D,
        "elementwise": ELEMENTWISE_1D,
        "row_reduction": ROW_REDUCTION,
        "reduction": ROW_REDUCTION,
        "matmul_2d": MATMUL_2D,
        "matmul": MATMUL_2D,
        "multi_output": MULTI_OUTPUT,
    }
    if user_hint:
        key = user_hint.lower()
        if key not in name_map:
            raise ValueError(f"Unknown skeleton hint {user_hint!r}. Available: {list(name_map)}")
        sk = name_map[key]
        if sk is MULTI_OUTPUT:
            sk = _attach_n_outputs(sk, _count_return_values(ctx.source_code))
        return sk, 1.0, f"user hint={user_hint!r}"

    src = ctx.source_code

    n_out = _count_return_values(src)
    if n_out > 1:
        return _attach_n_outputs(MULTI_OUTPUT, n_out), 0.9, f"detected {n_out} return values"

    if re.search(r"torch\.matmul\s*\(|torch\.mm\s*\(|\s@\s", src):
        if len(ctx.tensor_args) >= 2 and all(t.dim() >= 2 for _, t in ctx.tensor_args[:2]):
            return MATMUL_2D, 0.9, "matmul/@/mm op with >=2D tensors"

    if re.search(r"\.(sum|mean|max|min|prod|norm)\s*\(\s*dim\s*=", src):
        first_t = ctx.tensor_args[0][1] if ctx.tensor_args else None
        if first_t is not None and first_t.dim() >= 2:
            return ROW_REDUCTION, 0.9, "reduction-with-dim on >=2D tensor"

    # Fall through to elementwise. Check if the source contains semantics the
    # elementwise skeleton genuinely cannot express — if so emit low confidence.
    for pat in _DATA_DEP_PATTERNS:
        if re.search(pat, src):
            return ELEMENTWISE_1D, 0.3, f"no skeleton matches; source contains data-dep pattern /{pat}/"

    return ELEMENTWISE_1D, 0.5, "default fall-through (no positive signal)"


def _attach_n_outputs(sk: Skeleton, n: int) -> Skeleton:
    """Return a copy of sk with n_outputs set (for MULTI_OUTPUT)."""
    return Skeleton(
        name=sk.name, cpp_return_type=sk.cpp_return_type, n_outputs=n,
        build_prompt_block=sk.build_prompt_block,
        build_eval_compare=sk.build_eval_compare,
        signature_regex=sk.signature_regex,
        description=sk.description,
    )
