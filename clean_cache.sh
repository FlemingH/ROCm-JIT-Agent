#!/usr/bin/env bash

# clean_cache.sh
# 用于清除 ROCm-JIT-Agent 和 Triton 在本地生成的各种算子编译缓存

echo "=============================================="
echo " 开始清除 ROCm-JIT-Agent 与 Triton 的编译缓存"
echo "=============================================="

# 1. 清除 Triton 生成的算子缓存 (Triton 会把 .so 放在这里)
TRITON_CACHE_DIR="$HOME/.triton/cache"
if [ -d "$TRITON_CACHE_DIR" ]; then
    echo "[清理] 发现 Triton 缓存目录，正在删除: $TRITON_CACHE_DIR"
    rm -rf "$TRITON_CACHE_DIR"
    echo "  -> Triton 缓存清理完成."
else
    echo "[跳过] Triton 缓存目录不存在."
fi

# 2. 清除 ROCm-JIT-Agent 硬编码保存的底层 HIP 算子缓存
JIT_AGENT_CACHE_DIR="$HOME/.rocm_jit_agent_cache"
if [ -d "$JIT_AGENT_CACHE_DIR" ]; then
    echo "[清理] 发现 ROCm-JIT-Agent 缓存目录，正在删除: $JIT_AGENT_CACHE_DIR"
    rm -rf "$JIT_AGENT_CACHE_DIR"
    echo "  -> ROCm-JIT-Agent 缓存清理完成."
else
    echo "[跳过] ROCm-JIT-Agent 缓存目录不存在."
fi

# 3. 清理当前目录可能残留的测试生成的 .csv 和 .png ( Benchmark 结果)
echo "[清理] 清理工作目录下残留的测试数据文件 (Benchmark 记录)..."
find . -maxdepth 1 -name "*_performance.csv" -type f -delete
find . -maxdepth 1 -name "*_performance.png" -type f -delete
find . -maxdepth 1 -name "results.html" -type f -delete

echo "=============================================="
echo " 所有优化过的算子与测试记录已被彻底清除。"
echo " 接下来再次运行项目时将触发全新的编译流程。"
echo "=============================================="
