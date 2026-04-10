import os
import sys
import subprocess
import tempfile
import csv
from pathlib import Path

def analyze_kernel_performance(eval_path):
    """
    Run rocprofv3 on the generated python script to extract hardware counters of the HIP kernel.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        input_txt = tmpdir_path / "rocprof_input.txt"
        input_txt.write_text("pmc: SQ_WAVES_sum SQ_INSTS_VALU_sum\npmc: GL2C_HIT_sum GL2C_MISS_sum GL2C_MC_RDREQ_sum\n")
        
        out_prefix = tmpdir_path / "out"
        
        env = os.environ.copy()
        # Ensure we run in a clean environment for profiling
        cmd = ["rocprofv3", "-i", str(input_txt), "--kernel-trace", "-o", str(out_prefix), "-f", "csv", "--", sys.executable, str(eval_path)]
        
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return "rocprofv3 profiling timed out."
            
        if res.returncode != 0:
            return f"rocprofv3 failed to run properly. Output: {res.stderr[-200:]}"
            
        metrics = {}
        # Parse counter_collection.csv
        cc_path = Path(str(out_prefix) + "_counter_collection.csv")
        if cc_path.exists():
            with open(cc_path) as f:
                for row in csv.DictReader(f):
                    kname = row.get("Kernel_Name", "")
                    if "fused_kernel" not in kname and "optimized_func" not in kname: continue
                    cname = row.get("Counter_Name", "")
                    cval = row.get("Counter_Value", "")
                    if cname and cval:
                        try: metrics[cname] = float(cval)
                        except ValueError: pass
                    
                    for col in ["VGPR_Count", "SGPR_Count", "Workgroup_Size"]:
                        v = row.get(col, "")
                        if v and col not in metrics:
                            try: metrics[col] = int(float(v))
                            except ValueError: pass
                            
        # Parse kernel_trace.csv
        kt_path = Path(str(out_prefix) + "_kernel_trace.csv")
        if kt_path.exists():
            with open(kt_path) as f:
                for row in csv.DictReader(f):
                    kname = row.get("Kernel_Name", "")
                    if "fused_kernel" not in kname and "optimized_func" not in kname: continue
                    for col in ["VGPR_Count", "SGPR_Count", "Workgroup_Size"]:
                        v = row.get(col, "")
                        if v and col not in metrics:
                            try: metrics[col] = int(float(v))
                            except ValueError: pass

        if not metrics:
            return "No profiling metrics collected (kernel might not have executed properly during profiling or name mismatch)."

        gl2c_hit = metrics.get("GL2C_HIT_sum", 0)
        gl2c_miss = metrics.get("GL2C_MISS_sum", 0)
        l2_rate = 100.0 * gl2c_hit / (gl2c_hit + gl2c_miss) if (gl2c_hit + gl2c_miss) > 0 else None
        
        vgpr = metrics.get("VGPR_Count")
        
        lines = ["[Hardware Profiling Results]"]
        if vgpr is not None:
            max_waves = min(16, 1536 // max(vgpr, 1))
            occ_pct = 100.0 * max_waves / 16
            lines.append(f"- VGPR Count: {vgpr} (est. occupancy: {occ_pct:.0f}%). " + ("Warning: Reduce VGPRs to increase occupancy!" if occ_pct < 50 else ""))
            
        if l2_rate is not None:
            lines.append(f"- L2 Cache Hit Rate: {l2_rate:.1f}%. " + ("Warning: Improve memory coalescing!" if l2_rate < 50 else ""))
            
        return "\n".join(lines)