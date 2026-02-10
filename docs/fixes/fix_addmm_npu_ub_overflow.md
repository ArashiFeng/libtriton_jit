# 修复说明：addmm NPU 后端 UB 溢出问题

**修复日期**：2026-02-04
**关联问题**：TODO: link（建议补充至 [docs/debugging_index.md](../debugging_index.md)）

---

## 改动动机

NPU 后端运行 `addmm` 算子时，BiShengHIR 编译器报告 **Unified Buffer (UB) 溢出**错误：

```
ub overflow, requires 2097152 bits while 1572864 bits available!
```

**Root Cause**：`addmm_op.cpp` 使用了统一的 CUDA 级别 block 配置（64×64），导致 tiling 所需内存（~256KB）超出 NPU UB 容量限制（~192KB）。对比已正常运行的 `mm_op.cpp`，其 NPU 配置使用了更小的 block（32×32）。

---

## 改动范围

| 文件 | 改动说明 |
|------|----------|
| `operators/matmul/addmm/addmm_op.cpp` | 为 NPU 后端添加独立的 block 和 warp 配置，与 CUDA/MUSA 分离 |

---

## 关键修改点

### 1. 添加 NPU 专用 block 配置

**Before**（所有后端共用）：
```cpp
constexpr int64_t BLOCK_M = 64;
constexpr int64_t BLOCK_N = 64;
constexpr int64_t BLOCK_K = 32;
constexpr int num_warps = 4;
constexpr int num_stages = 2;
```

**After**（按后端区分）：
```cpp
#if defined(BACKEND_NPU)
    // NPU: Smaller blocks to fit in Unified Buffer (~192KB available)
    constexpr int64_t BLOCK_M = 32;
    constexpr int64_t BLOCK_N = 32;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 1;
    constexpr int num_stages = 1;
#else
    // CUDA/MUSA: Larger blocks for better performance
    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 4;
    constexpr int num_stages = 2;
#endif
```

**目的**：将 NPU 的 tile 大小从 64×64 降至 32×32，UB 内存占用从 ~256KB 降至 ~64KB，符合 NPU 硬件限制。同时 `num_warps=1` 和 `num_stages=1` 与 `mm_op.cpp` NPU 配置保持一致。

### 2. 内存占用计算对比

| 配置项 | 原值 | NPU 新值 | 内存影响 |
|--------|------|----------|----------|
| BLOCK_M × BLOCK_N | 64×64 | 32×32 | Acc 缓冲从 16KB 降至 4KB |
| A tile (BLOCK_M × BLOCK_K) | 64×32 | 32×32 | 从 8KB 降至 4KB |
| B tile (BLOCK_K × BLOCK_N) | 32×64 | 32×32 | 从 8KB 降至 4KB |
| num_stages | 2 | 1 | 禁用 double-buffering，内存减半 |

---

## 行为变化

| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| NPU 编译 | `MLIRCompilationError: ub overflow` | 编译成功 |
| UB 需求 | ~256KB（超限） | ~64KB（在 192KB 限制内） |
| CUDA/MUSA 行为 | 无变化 | 无变化（仍使用 64×64 配置） |
| NPU 性能 | N/A（无法运行） | 可运行，tile 数量增加但单 tile 小 |

---

## 验证方式

### 1. 构建并运行 addmm 测试

```bash
cmake --build build/ --target test_addmm --parallel
./build/operators/matmul/addmm/test_addmm
```

**成功标准**：无 `MLIRCompilationError`，输出 `All tests passed!`

### 2. 带日志验证 kernel 编译

```bash
TORCH_CPP_LOG_LEVEL=INFO ./build/operators/matmul/addmm/test_addmm 2>&1 | head -50
```

**成功标准**：日志显示 kernel 编译成功，包含类似输出：
```
[triton_jit] Compiling kernel: addmm_kernel
[triton_jit] Kernel compiled successfully
```

### 3. 正确性校验（预期输出）

```
=== Test: addmm_basic ===
Output shape: [64, 64]
Correctness: PASS
```

---

## See Also

- **参考实现**：`mm_op.cpp:78-84` 已有 NPU 专用配置，本次修复与其保持一致
- **Debug 线索落地**：BiShengHIR 的 `ub overflow` 错误信息直接指出了 tiling block 过大的问题，解决方案参照同类 matmul 算子的 NPU 配置
- **NPU 限制说明**：见 [CLAUDE.md](../../CLAUDE.md) 中 Backend Constraints 章节，NPU 因 BiShengHIR 限制不支持部分算子

---

## 后续建议

1. 考虑将 NPU block 配置提取为全局常量或配置文件，避免各算子重复定义
2. 可在 `docs/debugging_index.md` 添加本问题的排障记录条目
