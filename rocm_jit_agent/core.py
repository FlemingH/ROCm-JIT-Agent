import inspect
import os
import sys
import torch
import tempfile
import importlib.util

def optimize(target="gfx1100", backend="local:Jan-code-4b-gfx1100-HIP-1", force_recompile=False):
    def decorator(func):
        state = {'compiled': False, 'optimized_func': None}
        
        def wrapper(*args, **kwargs):
            if not state['compiled']:
                source_code = inspect.getsource(func)
                print(f"[rocm_jit_agent] 🚀 Intercepted PyTorch operator: {func.__name__} (Target: {target})")
                
                arg_names = list(inspect.signature(func).parameters.keys())
                tensor_info_strs = []
                tensor_infos_for_prof = []
                
                # Separate tensor args from scalar args
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
                
                # Map PyTorch dtypes to C types
                def dtype_to_ctype(dtype):
                    mapping = {
                        torch.float32: "float", torch.float64: "double",
                        torch.float16: "at::Half", torch.bfloat16: "at::BFloat16",
                        torch.int32: "int", torch.int64: "long", torch.int16: "short",
                        torch.int8: "int8_t", torch.uint8: "uint8_t", torch.bool: "bool",
                    }
                    return mapping.get(dtype, "float")
                
                def scalar_to_ctype(val):
                    if isinstance(val, bool): return "bool"
                    if isinstance(val, int): return "int"
                    if isinstance(val, float): return "float"
                    return "float"
                
                scalar_args_info = [(name, val) for name, val in scalar_args]
                
                # Build C++ function signature parts
                cpp_tensor_args = [f"torch::Tensor {name}" for name, _ in tensor_args]
                cpp_scalar_args = [f"{scalar_to_ctype(val)} {name}" for name, val in scalar_args]
                cpp_arg_str = ", ".join(cpp_tensor_args + cpp_scalar_args)
                
                # Build kernel pointer args (tensors become pointers, scalars pass through)
                kernel_ptr_parts = []
                launch_ptr_parts = []
                for name, t in tensor_args:
                    ctype = dtype_to_ctype(t.dtype)
                    kernel_ptr_parts.append(f"{ctype}* {name}_ptr")
                    launch_ptr_parts.append(f"{name}.data_ptr<{ctype}>()")
                for name, val in scalar_args:
                    kernel_ptr_parts.append(f"{scalar_to_ctype(val)} {name}")
                    launch_ptr_parts.append(name)
                kernel_ptr_args = ", ".join(kernel_ptr_parts)
                launch_data_ptrs = ", ".join(launch_ptr_parts)
                
                eager_us = 0.0
                compile_us = 0.0
                
                # 1. Evaluate real time of the original Eager function (Baseline)
                try:
                    import triton.testing
                    eager_ms = triton.testing.do_bench(lambda: func(*args, **kwargs))
                    eager_us = eager_ms * 1000
                except ImportError:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    for _ in range(3): func(*args, **kwargs) # Warmup
                    start_event.record()
                    for _ in range(10): func(*args, **kwargs)
                    end_event.record()
                    torch.cuda.synchronize()
                    eager_us = start_event.elapsed_time(end_event) / 10.0 * 1000
                
                # 2. Evaluate torch.compile as the ultimate target
                try:
                    compiled_func = torch.compile(func)
                    import triton.testing
                    # Warmup compile
                    for _ in range(3): compiled_func(*args, **kwargs)
                    compile_ms = triton.testing.do_bench(lambda: compiled_func(*args, **kwargs))
                    compile_us = compile_ms * 1000
                except Exception as e:
                    compile_us = eager_us
                
                target_us = min(eager_us, compile_us)
                
                import hashlib
                cache_dir = os.path.expanduser("~/.rocm_jit_agent_cache")
                os.makedirs(cache_dir, exist_ok=True)
                shape_sig = str([(list(a.shape), str(a.dtype)) for a in args if isinstance(a, torch.Tensor)])
                func_hash = hashlib.md5((source_code + str(target) + shape_sig).encode('utf-8')).hexdigest()
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
                        module = load_inline(name=f"dynamic_hip_{func_hash}", cpp_sources=cached_sig, cuda_sources=cached_code, functions=["optimized_func"], with_cuda=True, extra_cuda_cflags=["-O3"])
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
                        
                    import warnings
                    import logging
                    import contextlib
                    
                    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
                    warnings.filterwarnings("ignore")
                    
                    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
                        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
                        from transformers import logging as hf_logging
                        hf_logging.set_verbosity_error()
                        logging.getLogger("transformers").setLevel(logging.ERROR)
                        
                        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
                        llm = AutoModelForCausalLM.from_pretrained(
                            model_path, 
                            torch_dtype=torch.bfloat16, 
                            device_map="auto"
                        )
                    llm.eval()
                    
                    max_iters = 10
                    for iteration in range(1, max_iters + 1):
                        current_temp = 0.3 + (iteration - 1) * 0.15 # Gradually increase generation randomness
                        print(f"[rocm_jit_agent] 🔄 Iteration {iteration}/{max_iters} - Thinking & Generating (Temp: {current_temp:.2f})...")
                        
                        messages = [
                            {"role": "system", "content": "You are an expert AMD GPU optimization engineer. You specialize in rewriting PyTorch code into highly optimized HIP C++ kernels."},
                        ]
                        
                        # Build concrete broadcast examples from actual tensor shapes
                        broadcast_hints = []
                        for bname, bt in tensor_args:
                            if bt.dim() == 1:
                                broadcast_hints.append(f"  - {bname} has shape {list(bt.shape)}: use `pid % {bt.shape[-1]}` to index it")
                        broadcast_hint = "\n".join(broadcast_hints) if broadcast_hints else ""
                        user_msg = (
                            f"Convert the following PyTorch code to a highly optimized HIP C++ kernel for AMD GPU ({target}).\n"
                            f"Requirements:\n"
                            f"1. Return ONLY valid C++ code.\n"
                            f"2. You must implement the corresponding operations correctly based on the tensor shapes.\n"
                            f"Here are the tensor signatures: {', '.join(tensor_info_strs)}\n"
                            f"3. IMPORTANT: For 1D tensors, you must broadcast correctly using modulo indexing.\n{broadcast_hint}\n"
                            f"4. You MUST implement `torch::Tensor optimized_func(...)` taking EXACTLY the following arguments: {cpp_arg_str}.\n"
                            f"5. You must use this skeleton pattern:\n\n"
                            f"```cpp\n"
                            f"#include <torch/extension.h>\n"
                            f"#include <hip/hip_runtime.h>\n\n"
                            f"__global__ void fused_kernel(float* output_ptr, {kernel_ptr_args}, int n_elements) {{\n"
                            f"    int pid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                            f"    if (pid < n_elements) {{\n"
                            f"        // DO MATH HERE\n"
                            f"    }}\n"
                            f"}}\n\n"
                            f"torch::Tensor optimized_func({cpp_arg_str}) {{\n"
                            f"    // Allocate output based on the first tensor shape\n"
                            f"    auto output = torch::empty_like({arg_names[0] if arg_names else 'x'});\n"
                            f"    int n_elements = output.numel();\n"
                            f"    int threads = 256;\n"
                            f"    int blocks = (n_elements + threads - 1) / threads;\n"
                            f"    hipLaunchKernelGGL(fused_kernel, dim3(blocks), dim3(threads), 0, 0, output.data_ptr<float>(), {launch_data_ptrs}, n_elements);\n"
                            f"    return output;\n"
                            f"}}\n"
                            f"```\n\n"
                            f"Original code:\n```python\n{source_code}\n```"
                        )
                        
                        if iteration == 1 or not generated_code:
                            messages.append({"role": "user", "content": user_msg})
                        else:
                            clean_feedback = str(feedback).strip()
                            # Only apply error-line filtering for compilation/runtime errors
                            # For performance feedback (profiler data), keep full context
                            if "Execution or validation error" in clean_feedback or "ERROR" in clean_feedback:
                                fb_lines = clean_feedback.split('\n')
                                error_lines = [l for l in fb_lines if "error" in l.lower() or "line " in l.lower() or "File " in l]
                                if error_lines:
                                    clean_feedback = "\n".join(error_lines[-10:])
                            # Truncate only as a last resort
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
                        
                        import sys
                        # Streaming output
                        class CallbackStreamer(TextStreamer):
                            def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
                                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                                self.text = ""
                                self.max_display_len = 100
                            def on_finalized_text(self, text: str, stream_end: bool = False):
                                self.text += text
                                display_text = self.text.replace('\n', ' ')
                                if len(display_text) > self.max_display_len:
                                    display_text = "..." + display_text[-self.max_display_len:]
                                sys.stdout.write(f"\r\033[K[LLM Streaming] 🤖: {display_text}")
                                sys.stdout.flush()

                        streamer = CallbackStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                        
                        with torch.no_grad():
                            outputs = llm.generate(
                                **inputs,
                                max_new_tokens=1024,
                                temperature=current_temp,
                                do_sample=True,
                                top_p=0.95,
                                streamer=streamer,
                                pad_token_id=tokenizer.eos_token_id
                            )
                            
                        sys.stdout.write("\n")
                        
                        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        # Ensure we don't output beyond the expected stop token if any leaked
                        if "```" in generated_text:
                            generated_text = generated_text.split("```")[0]
                        generated_code = "#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n" + generated_text
                        
                        code_to_exec = generated_code.strip()
                        if "```" in code_to_exec:
                            code_to_exec = code_to_exec.split("```")[0]

                        try:
                            # Use an isolated subprocess to prevent Segfaults from killing the main process
                            import tempfile
                            import subprocess
                            
                            clean_source_code = "\n".join([line for line in source_code.split("\n") if not line.strip().startswith("@rocm_jit_agent") and not line.strip().startswith("@optimize")])
                            
                            import re
                            sig_match = re.search(r'(torch::Tensor\s+optimized_func\s*\([^)]*\))', code_to_exec)
                            cpp_sig = sig_match.group(1) + ";" if sig_match else "torch::Tensor optimized_func();"
                            
                            eval_script = f"""import torch\nimport sys\nimport os\nimport traceback\nfrom torch.utils.cpp_extension import load_inline\n\n"""
                            eval_script += f"os.environ['PYTORCH_ROCM_ARCH'] = '{target}'\n"
                            eval_script += f"# --- Original Code ---\n{clean_source_code}\noriginal_func = {func.__name__}\n\n"
                            eval_script += f"cpp_source = \"\"\"\\n{code_to_exec}\\n\"\"\"\n\n"
                            eval_script += "if __name__ == '__main__':\n"
                            eval_script += "    try:\n"
                            eval_script += "        import tempfile\n"
                            eval_script += "        build_dir = tempfile.mkdtemp(prefix='rocm_jit_agent_build_')\n"
                            eval_script += "        conda_bin = os.path.dirname(sys.executable)\n"
                            eval_script += "        os.environ['PATH'] = f'{conda_bin}:' + os.environ.get('PATH', '')\n"
                            eval_script += f"        cpp_sig = \"\"\"{cpp_sig}\"\"\"\n"
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
                                
                            eval_script += """
        out_eager = original_func(*args)
        try:
            out_opt = candidate(*args)
        except Exception as e:
            err_msg = str(e)
            if "incompatible function arguments" in err_msg:
                err_msg = "Function signature mismatch. Check generated C++ arguments vs python inputs."
            print(f"ERROR: {err_msg}")
            sys.exit(1)
            
        if isinstance(out_eager, torch.Tensor) and isinstance(out_opt, torch.Tensor):
            mse = torch.nn.functional.mse_loss(out_eager, out_opt).item()
        else:
            mse = 0.0
            
        if mse > 1e-3:
            # Show sample values to help the model understand what went wrong
            flat_e = out_eager.flatten()[:5]
            flat_o = out_opt.flatten()[:5]
            print(f"ERROR: Output correctness validation failed (MSE={mse:.5f}). Expected first 5 values: {flat_e.tolist()}, got: {flat_o.tolist()}. The mathematical logic is incorrect.")
            sys.exit(1)
            
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(3): candidate(*args) # warmup
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(10): candidate(*args)
        end_event.record()
        torch.cuda.synchronize()
        opt_us = start_event.elapsed_time(end_event) / 10.0 * 1000
        
        print(f"SUCCESS:{mse:.5f}:{opt_us:.2f}:{build_dir}")
    except Exception as e:
        import traceback
        err_msg = str(e)
        if "incompatible function arguments" in err_msg:
            err_msg = err_msg.split("Invoked with:")[0]
        print(f"ERROR:{err_msg}\\n{traceback.format_exc()}")
        sys.exit(1)
"""
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                                f.write(eval_script)
                                eval_path = f.name
                                
                            print(f"[rocm_jit_agent] ⏳ Compiling generated HIP C++ Kernel...")
                            out_path = eval_path + ".out"
                            try:
                                with open(out_path, "w") as f_out:
                                    # Give it enough time to compile HIP
                                    res = subprocess.run([sys.executable, eval_path], stdout=f_out, stderr=subprocess.STDOUT, timeout=90.0)
                                with open(out_path, "r") as f_in:
                                    res_stdout = f_in.read()
                                    
                                if res.returncode != 0 or "ERROR:" in res_stdout:
                                    error_text = res_stdout.replace("ERROR:", "").strip()
                                    # Extract ninja/hipcc errors cleanly
                                    if "FAILED: " in error_text:
                                        error_text = error_text[error_text.find("FAILED: "):]
                                        # Skip the hipcc command line (very long, not useful for diagnosis)
                                        # and extract the actual compiler error messages
                                        error_lines = error_text.split('\n')
                                        # Filter out the command repeat and keep actual errors
                                        useful_lines = [l for l in error_lines if not l.strip().startswith('/opt/rocm') and not l.strip().startswith('/home/')]
                                        if not useful_lines:
                                            useful_lines = error_lines[2:10]  # fallback: skip FAILED + command
                                        error_text = '\n'.join(useful_lines[:10])
                                        
                                    if "incompatible function arguments" in error_text:
                                        error_text = error_text.split("Invoked with:")[0]
                                        
                                    raise RuntimeError(error_text[:1500])
                                    
                                # If success, extract the timing and load it locally
                                output_parts = [line for line in res_stdout.split("\n") if line.startswith("SUCCESS:")]
                                if not output_parts and res_stdout.strip().split("\n")[-1].startswith("SUCCESS:"):
                                    output_parts = [res_stdout.strip().split("\n")[-1]]
                                
                                if not output_parts:
                                    raise RuntimeError(f"Unexpected subprocess output (no SUCCESS line): {res_stdout[-500:]}")
                                
                                success_str = output_parts[-1]
                                _, mse_str, opt_us_str, build_dir = success_str.split(":", 3)
                                opt_us = float(opt_us_str)
                                
                                # Now load it back into the main process efficiently
                                from torch.utils.cpp_extension import load_inline
                                conda_bin = os.path.dirname(sys.executable)
                                os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"
                                os.environ['PYTORCH_ROCM_ARCH'] = target
                                
                                import re
                                sig_match = re.search(r'(torch::Tensor\s+optimized_func\s*\([^)]*\))', code_to_exec)
                                cpp_sig = sig_match.group(1) + ";" if sig_match else "torch::Tensor optimized_func();"
                                
                                module = load_inline(name="dynamic_hip_kernel", cpp_sources=cpp_sig, cuda_sources=code_to_exec, functions=["optimized_func"], with_cuda=True, extra_cuda_cflags=["-O3"], build_directory=build_dir)
                                candidate = module.optimized_func
                                
                            finally:
                                for _tmp in [out_path, eval_path]:
                                    try:
                                        if os.path.exists(_tmp):
                                            os.unlink(_tmp)
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
                            # Avoid printing full traceback, truncate some lines
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
                        with open(cache_cpp, "w") as f:
                            f.write(best_code_to_exec)
                        with open(cache_sig, "w") as f:
                            f.write(best_cpp_sig)
                    except Exception as e:
                        print(f"[rocm_jit_agent] ⚠️ Failed to write cache: {e}")
                else:
                    print(f"[rocm_jit_agent] ⚠️ All {max_iters} iterations failed. Retaining original PyTorch operator for safety.")
                    state['optimized_func'] = func
                    
                state['compiled'] = True
            
            return state['optimized_func'](*args, **kwargs)
        return wrapper
    return decorator
