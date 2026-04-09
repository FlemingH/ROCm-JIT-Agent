import inspect
import os
import sys
import torch
import tempfile
import importlib.util
import multiprocessing
import queue

def safe_evaluate(file_path, args_list, kwargs_dict, result_queue):
    try:
        import torch
        import triton
        import triton.language as tl
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("dynamic_kernel", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        candidate = getattr(mod, 'optimized_func', None)
        if candidate is None:
            funcs = [f for n, f in mod.__dict__.items() if callable(f) and not n.startswith('_') and n != 'optimize' and n != 'safe_evaluate']
            if funcs:
                candidate = funcs[-1]
            else:
                raise ValueError("No callable function found in generated code (missing 'optimized_func').")
                
        out_opt = candidate(*args_list, **kwargs_dict)
        
        # Quick bench inside process
        # Use a smaller warmup to prevent timeout
        opt_ms = triton.testing.do_bench(lambda: candidate(*args_list, **kwargs_dict), warmup=2, rep=5)
        result_queue.put(('success', opt_ms * 1000))
    except Exception as e:
        result_queue.put(('error', str(e)))


def optimize(target="gfx1100", backend="local:Jan-code-4b"):
    def decorator(func):
        state = {'compiled': False, 'optimized_func': None}
        
        def wrapper(*args, **kwargs):
            if not state['compiled']:
                source_code = inspect.getsource(func)
                print(f"\n[rocm_jit_agent] ━━━━━━━━━ [1/5] 触发拦截 (Interception) ━━━━━━━━━")
                print(f"[rocm_jit_agent] 侦测到未优化的 PyTorch 算子: {func.__name__}")
                
                tensor_info_strs = []
                tensor_infos_for_prof = []
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        tensor_info_strs.append(f"input_{i}({list(arg.shape)}, {arg.dtype})")
                        tensor_infos_for_prof.append({'shape': list(arg.shape), 'dtype': str(arg.dtype)})
                print(f"[rocm_jit_agent] AI 分析张量签名: {', '.join(tensor_info_strs)}...")
                
                eager_us = 0.0
                compile_us = 0.0
                
                # 1. 评测原生 Eager 函数的真实耗时 (Baseline)
                try:
                    import triton.testing
                    eager_ms = triton.testing.do_bench(lambda: func(*args, **kwargs))
                    eager_us = eager_ms * 1000
                except ImportError:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    for _ in range(3): func(*args, **kwargs) # 预热
                    start_event.record()
                    for _ in range(10): func(*args, **kwargs)
                    end_event.record()
                    torch.cuda.synchronize()
                    eager_us = start_event.elapsed_time(end_event) / 10.0 * 1000
                
                print(f"[rocm_jit_agent] Torch Eager Baseline 耗时: {eager_us:.1f}us")
                
                # 2. 评测 torch.compile 作为终极目标
                try:
                    compiled_func = torch.compile(func)
                    import triton.testing
                    # Warmup compile
                    for _ in range(3): compiled_func(*args, **kwargs)
                    compile_ms = triton.testing.do_bench(lambda: compiled_func(*args, **kwargs))
                    compile_us = compile_ms * 1000
                    print(f"[rocm_jit_agent] Torch Compile Baseline 耗时: {compile_us:.1f}us")
                except Exception as e:
                    compile_us = eager_us
                    print(f"[rocm_jit_agent] 无法获取 Torch Compile 耗时，将 Eager {eager_us:.1f}us 作为优化基准.")
                
                target_us = min(eager_us, compile_us)
                
                import hashlib
                cache_dir = os.path.expanduser("~/.rocm_jit_agent_cache")
                os.makedirs(cache_dir, exist_ok=True)
                func_hash = hashlib.md5((source_code + str(target)).encode('utf-8')).hexdigest()
                cache_cpp = os.path.join(cache_dir, f"{func_hash}.cpp")
                cache_sig = os.path.join(cache_dir, f"{func_hash}_sig.cpp")
                
                if os.path.exists(cache_cpp) and os.path.exists(cache_sig):
                    print(f"\n[rocm_jit_agent] ⚡ 发现本地持久化缓存，直接加载 O(1) 零延迟介入...")
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
                        print(f"[rocm_jit_agent] ⚠️ 加载本地缓存失败，回退到重新生成: {e}")
                
                print(f"\n[rocm_jit_agent] ━━━━━━━━━ [2~4/5] 开启多轮迭代优化引擎 ━━━━━━━━")
                print(f"[rocm_jit_agent] 🔥 唤醒 Kernel Forge, 目标耗时: < {target_us:.1f}us (最高 10 轮)")
                
                best_candidate = None
                best_opt_us = float('inf')
                best_code_to_exec = ""
                best_cpp_sig = ""
                feedback = None
                generated_code = ""
                max_iters = 10
                
                try:
                    base_model_path = "models/Jan-code-4b"
                    adapter_path = "models/grpo-jan-code-4b-b26"
                    
                    if not os.path.exists(base_model_path):
                        raise FileNotFoundError(f"未找到基础模型 {base_model_path}")
                        
                    print(f"[rocm_jit_agent] 🧠 加载原版模型及 LoRA: {adapter_path}")
                    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
                    import warnings
                    warnings.filterwarnings("ignore")
                    
                    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
                    llm = AutoModelForCausalLM.from_pretrained(
                        "models/merged-Jan-code-4b", 
                        torch_dtype=torch.bfloat16, 
                        device_map="auto"
                    )
                    llm.eval()
                    
                    max_iters = 10
                    for iteration in range(1, max_iters + 1):
                        current_temp = 0.3 + (iteration - 1) * 0.15 # 逐步增加生成随机性
                        print(f"\n[rocm_jit_agent] 🔄 --- Iteration {iteration}/{max_iters} ---")
                        
                        messages = [
                            {"role": "system", "content": "You are an expert AMD GPU optimization engineer. You specialize in rewriting PyTorch code into highly optimized HIP C++ kernels."},
                        ]
                        
                        user_msg = (
                            f"Convert the following PyTorch code to a highly optimized HIP C++ kernel for AMD GPU ({target}).\n"
                            f"Requirements:\n"
                            f"1. Return ONLY valid C++ code.\n"
                            f"2. You must implement the corresponding operations correctly based on the tensor shapes.\n"
                            f"Here are the tensor signatures: {', '.join(tensor_info_strs)}\n"
                            f"3. IMPORTANT: For 1D tensors (e.g. bias, weight), you must use modulo indexing like `pid % 4096` if `pid` is larger than the 1D tensor's dimension, or broadcast correctly depending on the logic.\n"
                            f"4. You must use exactly this skeleton pattern:\n\n"
                            f"```cpp\n"
                            f"#include <torch/extension.h>\n"
                            f"#include <hip/hip_runtime.h>\n\n"
                            f"__global__ void fused_kernel(float* x_ptr, float* weight_ptr, float* bias_ptr, float* output_ptr, int n_elements) {{\n"
                            f"    int pid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                            f"    if (pid < n_elements) {{\n"
                            f"        // DO MATH HERE\n"
                            f"        // Example: float x_val = x_ptr[pid];\n"
                            f"        // Example: float weight_val = weight_ptr[pid % 4096]; // Make sure to use modulo for broadcasting 1D tensors to 2D tensors\n"
                            f"    }}\n"
                            f"}}\n\n"
                            f"torch::Tensor optimized_func(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {{\n"
                            f"    auto output = torch::empty_like(x);\n"
                            f"    int n_elements = x.numel();\n"
                            f"    int threads = 256;\n"
                            f"    int blocks = (n_elements + threads - 1) / threads;\n"
                            f"    hipLaunchKernelGGL(fused_kernel, dim3(blocks), dim3(threads), 0, 0, x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), n_elements);\n"
                            f"    return output;\n"
                            f"}}\n"
                            f"```\n\n"
                            f"Original code:\n```python\n{source_code}\n```"
                        )
                        
                        if iteration == 1 or not generated_code:
                            messages.append({"role": "user", "content": user_msg})
                        else:
                            clean_feedback = str(feedback).strip()
                            if "error" in clean_feedback.lower():
                                lines = clean_feedback.split('\n')
                                clean_feedback = "\n".join([l for l in lines if "error" in l.lower() or "line " in l.lower() or "File " in l][-5:])
                            if not clean_feedback:
                                clean_feedback = str(feedback).strip()[:300]
                                
                            messages.append({"role": "user", "content": user_msg})
                            messages.append({"role": "assistant", "content": f"```cpp\n{generated_code}\n```"})
                            messages.append({
                                "role": "user", 
                                "content": f"The kernel above failed or is too slow with this feedback:\n```\n{clean_feedback}\n```\n\nPlease fix the errors, optimize it further, and output the corrected valid C++ code. Return ONLY code."
                            })
                            
                        try:
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        except Exception:
                            parts = []
                            for m in messages:
                                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
                            prompt = "\n".join(parts) + "\n<|im_start|>assistant\n"
                            
                        prompt += "```cpp\n#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n"
                        
                        print(f"[rocm_jit_agent] 🧠 模型思考与生成中 (Temperature: {current_temp:.2f})...")
                        
                        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
                        
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
                                sys.stdout.write(f"\r\033[K[模型流式生成] 🤖: {display_text}")
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
                            
                        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        # Ensure we don't output beyond the expected stop token if any leaked
                        if "```" in generated_text:
                            generated_text = generated_text.split("```")[0]
                        generated_code = "#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n" + generated_text
                        
                        print(f"\n[rocm_jit_agent] 提取生成的代码执行沙盒编译...")
                        
                        code_to_exec = generated_code.strip()
                        if "```" in code_to_exec:
                            code_to_exec = code_to_exec.split("```")[0]
                            
                        import multiprocessing
                        import queue
                        import traceback

                        try:
                            # Use an isolated subprocess to prevent Segfaults from killing the main process
                            import tempfile
                            import subprocess
                            import json
                            
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
                                eval_script += f"        args.append(torch.randn({shape_str}, dtype={dtype_str}, device='cuda'))\n"
                                
                            eval_script += """
        out_eager = original_func(*args)
        out_opt = candidate(*args)
        if isinstance(out_eager, torch.Tensor) and isinstance(out_opt, torch.Tensor):
            mse = torch.nn.functional.mse_loss(out_eager, out_opt).item()
        else:
            mse = 0.0
            
        if mse > 1e-3:
            print(f"ERROR: Output correctness validation failed (MSE={mse:.5f}). Fix the logic bugs.")
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
        print(f"ERROR:{str(e)}\\n{traceback.format_exc()}")
        sys.exit(1)
"""
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                                f.write(eval_script)
                                eval_path = f.name
                                
                            print(f"[rocm_jit_agent] ⏳ 正在调用 ninja 与 hipcc 进行底层 C++ 编译，可能需要几十秒，请耐心等待...")
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
                                    raise RuntimeError(error_text[:1500])
                                    
                                # If success, extract the timing and load it locally
                                output_parts = [line for line in res_stdout.split("\n") if line.startswith("SUCCESS:")]
                                if not output_parts and res_stdout.strip().split("\n")[-1].startswith("SUCCESS:"):
                                    output_parts = [res_stdout.strip().split("\n")[-1]]
                                
                                success_str = output_parts[-1]
                                _, mse_str, opt_us_str, build_dir = success_str.split(":")
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
                                if os.path.exists(out_path):
                                    os.unlink(out_path)
                            
                            print(f"[rocm_jit_agent] ✅ 正确性验证通过 (MSE={mse_str}), 进行性能测速...")
                            print(f"[rocm_jit_agent] 当前生成算子耗时: {opt_us:.1f}us")
                            
                            if opt_us < best_opt_us:
                                best_opt_us = opt_us
                                best_candidate = candidate
                                best_code_to_exec = code_to_exec
                                best_cpp_sig = cpp_sig
                                
                            if opt_us <= target_us:
                                print(f"[rocm_jit_agent] 🏆 目标达成！当前速度 {opt_us:.1f}us <= 目标 {target_us:.1f}us")
                                break
                            else:
                                print(f"[rocm_jit_agent] ⚠️ 未达目标速度，启动 rocprofv3 硅片级分析...")
                                from .profiler import analyze_kernel_performance
                                prof_feedback = analyze_kernel_performance(code_to_exec, tensor_infos_for_prof)
                                print(f"[rocm_jit_agent] 📊 rocprofv3 反馈:\n{prof_feedback}")
                                
                                feedback = f"The generated kernel works correctly but takes {opt_us:.1f}us. The target is {target_us:.1f}us. Hardware profiling results:\n{prof_feedback}\nPlease optimize performance: check memory coalescing, block sizes, or unnecessary memory reads."
                                print(f"[rocm_jit_agent] ⚠️ 开启下一轮重构...")
                                
                        except subprocess.TimeoutExpired:
                            feedback = "Execution or validation error: HIP compilation or execution timed out."
                            print(f"[rocm_jit_agent] ❌ 编译或验证失败: HIP 编译或执行超时")
                        except Exception as e:
                            feedback = f"Execution or validation error:\n{str(e)}"
                            # 避免打印整屏的 traceback，截断一些
                            clean_e = str(e)
                            print(f"[rocm_jit_agent] ❌ 编译或验证失败: {clean_e[:300]}")
                            
                except Exception as e:
                    import traceback
                    print(f"[rocm_jit_agent] 推理系统发生严重错误: {e}\n{traceback.format_exc()}")

                print(f"\n[rocm_jit_agent] ━━━━━━━━━ [5/5] 测试运行与永久缓存 (Test & Cache) ━━━━")
                if best_candidate:
                    speedup = eager_us / best_opt_us if best_opt_us > 0 else 0
                    print(f"[rocm_jit_agent] ✨ 锻造成功！Torch Eager: {eager_us:.1f}us | Compile: {compile_us:.1f}us -> AI HIP Kernel: {best_opt_us:.1f}us (相对 Eager 提速 {speedup:.1f}x)")
                    print(f"[rocm_jit_agent] 💾 已将最优算子硬编码至本地持久化缓存，当前与后续所有的运行将 O(1) 零延迟介入。\n")
                    state['optimized_func'] = best_candidate
                    
                    try:
                        with open(cache_cpp, "w") as f:
                            f.write(best_code_to_exec)
                        with open(cache_sig, "w") as f:
                            f.write(best_cpp_sig)
                    except Exception as e:
                        print(f"[rocm_jit_agent] ⚠️ 写入缓存失败: {e}")
                else:
                    print(f"[rocm_jit_agent] ⚠️ {max_iters} 轮迭代算子生成均未通过或全部报错，为保证安全性，保留原生 PyTorch 算子。\n")
                    state['optimized_func'] = func
                    
                state['compiled'] = True
            
            return state['optimized_func'](*args, **kwargs)
        return wrapper
    return decorator
