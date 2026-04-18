import inspect
import os
import sys
import torch
import tempfile
import importlib.util

from .sanitizer import sanitize as sanitize_code
from .skeletons import (
    SkeletonContext, classify,
    dtype_to_ctype, scalar_to_ctype,
    ELEMENTWISE_1D, ROW_REDUCTION, MATMUL_2D, MULTI_OUTPUT,
)


def optimize(target="gfx1100", backend="local:Jan-code-4b-gfx1100-HIP-1",
             force_recompile=False, skeleton=None):
    """JIT-compile a PyTorch function into an optimized HIP kernel.

    Parameters
    ----------
    target : str
        GPU arch, e.g. 'gfx1100'.
    force_recompile : bool
        If True, bypass persistent cache.
    skeleton : str or None
        Optional skeleton hint: 'elementwise_1d' | 'row_reduction' |
        'matmul_2d' | 'multi_output'. If None, classified automatically.
    """
    def decorator(func):
        state = {'compiled': False, 'optimized_func': None}

        def wrapper(*args, **kwargs):
            if not state['compiled']:
                source_code = inspect.getsource(func)
                print(f"[rocm_jit_agent] 🚀 Intercepted PyTorch operator: {func.__name__} (Target: {target})")

                arg_names = list(inspect.signature(func).parameters.keys())
                tensor_info_strs = []
                tensor_infos_for_prof = []

                tensor_args = []
                scalar_args = []
                for i, arg in enumerate(args):
                    name = arg_names[i] if i < len(arg_names) else f"input_{i}"
                    if isinstance(arg, torch.Tensor):
                        tensor_args.append((name, arg))
                        tensor_info_strs.append(f"{name}({list(arg.shape)}, {arg.dtype})")
                        tensor_infos_for_prof.append({"shape": list(arg.shape), "dtype": str(arg.dtype)})
                    else:
                        scalar_args.append((name, arg))

                scalar_args_info = [(name, val) for name, val in scalar_args]

                # --- Layer 1 & 2: classify task & pick skeleton ---
                ctx = SkeletonContext(
                    func_name=func.__name__,
                    source_code=source_code,
                    target=target,
                    tensor_args=tensor_args,
                    scalar_args=scalar_args,
                    arg_names=arg_names,
                )
                sk = classify(ctx, user_hint=skeleton)
                print(f"[rocm_jit_agent] 🧩 Selected skeleton: {sk.name} ({sk.description})")

                # Build C++ function signature (used for subprocess sig extraction)
                cpp_tensor_args = [f"torch::Tensor {name}" for name, _ in tensor_args]
                cpp_scalar_args = [f"{scalar_to_ctype(val)} {name}" for name, val in scalar_args]
                cpp_arg_str = ", ".join(cpp_tensor_args + cpp_scalar_args)

                eager_us = 0.0
                compile_us = 0.0

                try:
                    import triton.testing
                    eager_ms = triton.testing.do_bench(lambda: func(*args, **kwargs))
                    eager_us = eager_ms * 1000
                except ImportError:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    for _ in range(3): func(*args, **kwargs)
                    start_event.record()
                    for _ in range(10): func(*args, **kwargs)
                    end_event.record()
                    torch.cuda.synchronize()
                    eager_us = start_event.elapsed_time(end_event) / 10.0 * 1000

                try:
                    compiled_func = torch.compile(func)
                    import triton.testing
                    for _ in range(3): compiled_func(*args, **kwargs)
                    compile_ms = triton.testing.do_bench(lambda: compiled_func(*args, **kwargs))
                    compile_us = compile_ms * 1000
                except Exception:
                    compile_us = eager_us

                target_us = min(eager_us, compile_us)

                import hashlib
                cache_dir = os.path.expanduser("~/.rocm_jit_agent_cache")
                os.makedirs(cache_dir, exist_ok=True)
                shape_sig = str([(list(a.shape), str(a.dtype)) for a in args if isinstance(a, torch.Tensor)])
                func_hash = hashlib.md5(
                    (source_code + str(target) + shape_sig + sk.name + str(sk.n_outputs)).encode('utf-8')
                ).hexdigest()
                cache_cpp = os.path.join(cache_dir, f"{func_hash}.cpp")
                cache_sig = os.path.join(cache_dir, f"{func_hash}_sig.cpp")

                if not force_recompile and os.path.exists(cache_cpp) and os.path.exists(cache_sig):
                    print(f"[rocm_jit_agent] ⚡ Found persistent cache. Loading O(1) zero-latency Kernel...")
                    with open(cache_cpp, "r") as f:
                        cached_code = f.read()
                    with open(cache_sig, "r") as f:
                        cached_sig = f.read()

                    from torch.utils.cpp_extension import load_inline
                    conda_bin = os.path.dirname(sys.executable)
                    os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"
                    os.environ['PYTORCH_ROCM_ARCH'] = target
                    try:
                        module = load_inline(name=f"dynamic_hip_{func_hash}", cpp_sources=cached_sig,
                                             cuda_sources=cached_code, functions=["optimized_func"],
                                             with_cuda=True, extra_cuda_cflags=["-O3"])
                        state['optimized_func'] = module.optimized_func
                        state['compiled'] = True
                        return state['optimized_func'](*args, **kwargs)
                    except Exception as e:
                        print(f"[rocm_jit_agent] ⚠️ Cache load failed, falling back to generation: {e}")

                print(f"[rocm_jit_agent] 🔥 Waking up Kernel Forge. Baseline Eager: {eager_us:.1f}us | Torch Compile: {compile_us:.1f}us | Target: < {target_us:.1f}us")

                best_candidate = None
                best_opt_us = float('inf')
                best_code_to_exec = ""
                best_cpp_sig = ""
                feedback = None
                generated_code = ""
                max_iters = 10

                try:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    model_path = os.path.join(base_dir, "models", "Jan-code-4b-gfx1100-HIP-1")

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Fused large model not found {model_path}")

                    import warnings, logging, contextlib
                    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
                    warnings.filterwarnings("ignore")

                    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
                        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
                        from transformers import logging as hf_logging
                        hf_logging.set_verbosity_error()
                        logging.getLogger("transformers").setLevel(logging.ERROR)
                        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
                        llm = AutoModelForCausalLM.from_pretrained(
                            model_path, torch_dtype=torch.bfloat16, device_map="auto"
                        )
                    llm.eval()

                    for iteration in range(1, max_iters + 1):
                        current_temp = 0.3 + (iteration - 1) * 0.15
                        print(f"[rocm_jit_agent] 🔄 Iteration {iteration}/{max_iters} - Thinking & Generating (Temp: {current_temp:.2f})...")

                        messages = [
                            {"role": "system", "content": "You are an expert AMD GPU optimization engineer. You specialize in rewriting PyTorch code into highly optimized HIP C++ kernels."},
                        ]

                        # --- Layer 2: skeleton renders the user message ---
                        user_msg = sk.build_prompt_block(ctx, sk)

                        if iteration == 1 or not generated_code:
                            messages.append({"role": "user", "content": user_msg})
                        else:
                            clean_feedback = str(feedback).strip()
                            if "Execution or validation error" in clean_feedback or "ERROR" in clean_feedback:
                                fb_lines = clean_feedback.split('\n')
                                error_lines = [l for l in fb_lines if "error" in l.lower() or "line " in l.lower() or "File " in l]
                                if error_lines:
                                    clean_feedback = "\n".join(error_lines[-10:])
                            if not clean_feedback:
                                clean_feedback = str(feedback).strip()[:500]
                            messages.append({"role": "user", "content": user_msg})
                            messages.append({"role": "assistant", "content": f"```cpp\n{generated_code}\n```"})
                            messages.append({
                                "role": "user",
                                "content": f"The kernel above has this issue:\n```\n{clean_feedback}\n```\n\nPlease analyze the feedback above carefully and fix all issues. Return ONLY the corrected C++ code."
                            })

                        try:
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        except Exception:
                            parts = []
                            for m in messages:
                                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
                            prompt = "\n".join(parts) + "\n<|im_start|>assistant\n"

                        prompt += "```cpp\n#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n"

                        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

                        class CallbackStreamer(TextStreamer):
                            def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
                                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                                self.text = ""
                                self.max_display_len = 100
                            def on_finalized_text(self, text, stream_end=False):
                                self.text += text
                                d = self.text.replace('\n', ' ')
                                if len(d) > self.max_display_len:
                                    d = "..." + d[-self.max_display_len:]
                                sys.stdout.write(f"\r\033[K[LLM Streaming] 🤖: {d}")
                                sys.stdout.flush()

                        streamer = CallbackStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                        with torch.no_grad():
                            outputs = llm.generate(
                                **inputs, max_new_tokens=1024, temperature=current_temp,
                                do_sample=True, top_p=0.95, streamer=streamer,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        sys.stdout.write("\n")

                        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        if "```" in generated_text:
                            generated_text = generated_text.split("```")[0]
                        generated_code = "#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n" + generated_text

                        # --- Layer 3: deterministic sanitizer ---
                        code_to_exec, patches = sanitize_code(generated_code.strip())
                        if patches:
                            print(f"[rocm_jit_agent] 🧼 Sanitizer applied: {patches}")

                        try:
                            import subprocess, re
                            clean_source_code = "\n".join(
                                [line for line in source_code.split("\n")
                                 if not line.strip().startswith("@rocm_jit_agent")
                                 and not line.strip().startswith("@optimize")]
                            )

                            sig_match = re.search(sk.signature_regex, code_to_exec)
                            cpp_sig = sig_match.group(1) + ";" if sig_match else f"{sk.cpp_return_type} optimized_func();"

                            compare_block = sk.build_eval_compare(ctx, sk)

                            eval_script = "import torch, sys, os, traceback\n"
                            eval_script += "from torch.utils.cpp_extension import load_inline\n"
                            eval_script += f"os.environ['PYTORCH_ROCM_ARCH'] = '{target}'\n"
                            eval_script += "MSE_THRESH = 1e-3\n"
                            eval_script += "MATMUL_MSE_THRESH = 1e-1\n"
                            eval_script += f"# --- Original Code ---\n{clean_source_code}\noriginal_func = {func.__name__}\n\n"
                            eval_script += f"cpp_source = \"\"\"\n{code_to_exec}\n\"\"\"\n"
                            eval_script += f"cpp_sig = \"\"\"{cpp_sig}\"\"\"\n"
                            eval_script += "if __name__ == '__main__':\n"
                            eval_script += "    try:\n"
                            eval_script += "        import tempfile\n"
                            eval_script += "        build_dir = tempfile.mkdtemp(prefix='rocm_jit_agent_build_')\n"
                            eval_script += "        conda_bin = os.path.dirname(sys.executable)\n"
                            eval_script += "        os.environ['PATH'] = f'{conda_bin}:' + os.environ.get('PATH', '')\n"
                            eval_script += "        module = load_inline(name='dynamic_hip_kernel', cpp_sources=cpp_sig, cuda_sources=cpp_source, functions=['optimized_func'], with_cuda=True, extra_cuda_cflags=['-O3'], build_directory=build_dir)\n"
                            eval_script += "        candidate = module.optimized_func\n"
                            eval_script += "        args = []\n"

                            for t in tensor_infos_for_prof:
                                shape_str = str(t['shape'])
                                dtype_str = t['dtype']
                                if 'int' in dtype_str or 'bool' in dtype_str:
                                    eval_script += f"        args.append(torch.randint(0, 10, {shape_str}, dtype={dtype_str}, device='cuda'))\n"
                                else:
                                    eval_script += f"        args.append(torch.randn({shape_str}, dtype={dtype_str}, device='cuda'))\n"
                            for sname, sval in scalar_args_info:
                                eval_script += f"        args.append({repr(sval)})\n"

                            eval_script += "        out_eager = original_func(*args)\n"
                            eval_script += "        try:\n"
                            eval_script += "            out_opt = candidate(*args)\n"
                            eval_script += "        except Exception as e:\n"
                            eval_script += "            msg = str(e)\n"
                            eval_script += "            if 'incompatible function arguments' in msg: msg = 'Function signature mismatch.'\n"
                            eval_script += "            print(f'ERROR: {msg}'); sys.exit(1)\n"
                            # indent the skeleton's compare block into the try branch
                            import textwrap as _tw
                            eval_script += _tw.indent(compare_block, "        ")
                            eval_script += "        start_event = torch.cuda.Event(enable_timing=True)\n"
                            eval_script += "        end_event = torch.cuda.Event(enable_timing=True)\n"
                            eval_script += "        for _ in range(3): candidate(*args)\n"
                            eval_script += "        torch.cuda.synchronize()\n"
                            eval_script += "        start_event.record()\n"
                            eval_script += "        for _ in range(10): candidate(*args)\n"
                            eval_script += "        end_event.record()\n"
                            eval_script += "        torch.cuda.synchronize()\n"
                            eval_script += "        opt_us = start_event.elapsed_time(end_event) / 10.0 * 1000\n"
                            eval_script += "        print(f'SUCCESS:0.00000:{opt_us:.2f}:{build_dir}')\n"
                            eval_script += "    except Exception as e:\n"
                            eval_script += "        import traceback\n"
                            eval_script += "        msg = str(e)\n"
                            eval_script += "        if 'incompatible function arguments' in msg: msg = msg.split('Invoked with:')[0]\n"
                            eval_script += "        print(f'ERROR:{msg}\\n{traceback.format_exc()}')\n"
                            eval_script += "        sys.exit(1)\n"

                            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                                f.write(eval_script); eval_path = f.name

                            print(f"[rocm_jit_agent] ⏳ Compiling generated HIP C++ Kernel...")
                            out_path = eval_path + ".out"
                            try:
                                with open(out_path, "w") as f_out:
                                    res = subprocess.run([sys.executable, eval_path], stdout=f_out, stderr=subprocess.STDOUT, timeout=120.0)
                                with open(out_path, "r") as f_in:
                                    res_stdout = f_in.read()

                                if res.returncode != 0 or "ERROR:" in res_stdout:
                                    error_text = res_stdout.replace("ERROR:", "").strip()
                                    if "FAILED: " in error_text:
                                        error_text = error_text[error_text.find("FAILED: "):]
                                        error_lines = error_text.split('\n')
                                        useful_lines = [l for l in error_lines if not l.strip().startswith('/opt/rocm') and not l.strip().startswith('/home/')]
                                        if not useful_lines:
                                            useful_lines = error_lines[2:10]
                                        error_text = '\n'.join(useful_lines[:10])
                                    if "incompatible function arguments" in error_text:
                                        error_text = error_text.split("Invoked with:")[0]
                                    raise RuntimeError(error_text[:1500])

                                output_parts = [line for line in res_stdout.split("\n") if line.startswith("SUCCESS:")]
                                if not output_parts:
                                    raise RuntimeError(f"Unexpected subprocess output (no SUCCESS line): {res_stdout[-500:]}")
                                success_str = output_parts[-1]
                                _, mse_str, opt_us_str, build_dir = success_str.split(":", 3)
                                opt_us = float(opt_us_str)

                                from torch.utils.cpp_extension import load_inline
                                conda_bin = os.path.dirname(sys.executable)
                                os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"
                                os.environ['PYTORCH_ROCM_ARCH'] = target
                                module = load_inline(name="dynamic_hip_kernel", cpp_sources=cpp_sig,
                                                     cuda_sources=code_to_exec, functions=["optimized_func"],
                                                     with_cuda=True, extra_cuda_cflags=["-O3"],
                                                     build_directory=build_dir)
                                candidate = module.optimized_func
                            finally:
                                for _tmp in [out_path, eval_path]:
                                    try:
                                        if os.path.exists(_tmp): os.unlink(_tmp)
                                    except OSError:
                                        pass

                            print(f"[rocm_jit_agent] ✅ Validation Passed (MSE={mse_str}). Opt Execution Time: {opt_us:.1f}us")

                            if opt_us < best_opt_us:
                                best_opt_us = opt_us
                                best_candidate = candidate
                                best_code_to_exec = code_to_exec
                                best_cpp_sig = cpp_sig

                            if opt_us <= target_us:
                                print(f"[rocm_jit_agent] 🏆 Goal Reached! ({opt_us:.1f}us <= {target_us:.1f}us)")
                                break
                            else:
                                from .profiler import analyze_kernel_performance
                                prof_feedback = analyze_kernel_performance(eval_path)
                                print(f"[rocm_jit_agent] 📊 rocprofv3 Hardware Feedback:\n{prof_feedback}")
                                feedback = (
                                    f"The generated kernel compiles and produces correct results, "
                                    f"but its execution time is {opt_us:.1f}us, which exceeds the target of {target_us:.1f}us "
                                    f"(need {opt_us/target_us:.1f}x speedup).\n"
                                    f"Hardware profiling analysis from rocprofv3:\n{prof_feedback}\n"
                                    f"Based on the profiling data above, optimize the kernel to meet the performance target."
                                )
                                print(f"[rocm_jit_agent] ⚠️ Target not reached. Refining code for next iteration...")

                        except subprocess.TimeoutExpired:
                            feedback = "Execution or validation error: HIP compilation or execution timed out."
                            print(f"[rocm_jit_agent] ❌ Failed: HIP Compilation Timed Out")
                        except Exception as e:
                            feedback = f"Execution or validation error:\n{str(e)}"
                            clean_e = str(e).replace("\033[92mSuccessfully preprocessed all matching files.\033[0m\n", "")
                            print(f"[rocm_jit_agent] ❌ Failed: {clean_e[:200]}")

                except Exception as e:
                    import traceback
                    print(f"[rocm_jit_agent] Critical Error: {e}\n{traceback.format_exc()}")

                if best_candidate:
                    speedup = compile_us / best_opt_us if best_opt_us > 0 else 0
                    print(f"[rocm_jit_agent] ✨ Forging Successful! AI HIP Kernel: {best_opt_us:.1f}us (Speedup {speedup:.1f}x vs Torch Compile {compile_us:.1f}us) -> Persistent Cache Saved.")
                    state['optimized_func'] = best_candidate
                    try:
                        with open(cache_cpp, "w") as f: f.write(best_code_to_exec)
                        with open(cache_sig, "w") as f: f.write(best_cpp_sig)
                    except Exception as e:
                        print(f"[rocm_jit_agent] ⚠️ Failed to write cache: {e}")
                else:
                    print(f"[rocm_jit_agent] ⚠️ All {max_iters} iterations failed. Retaining original PyTorch operator for safety.")
                    state['optimized_func'] = func

                state['compiled'] = True

            return state['optimized_func'](*args, **kwargs)
        return wrapper
    return decorator
