# 修复：NPU 后端 Kernel 函数名不匹配与指针运算兼容性问题

**日期**：2026-02-04
**关联问题**：[docs/debugging_index.md](../debugging_index.md) - NPU 算子测试失败
**影响范围**：`sum`、`softmax`、`apply_rotary_pos_emb`、`fused_add_rms_norm` 算子

---

## 改动动机

NPU 后端运行多个算子测试时出现两类错误：

1. **`AttributeError: module 'xxx' has no attribute 'xxx_kernel'`**：C++ 调用的 kernel 函数名与 Python 定义的实际函数名不匹配
2. **`IncompatibleTypeErrorImpl: invalid operands of type pointer<fp32> and triton.language.int64`**：NPU Triton 不支持指针的 in-place 运算（`+=`）

---

## 改动范围

| 文件路径 | 改动说明 |
|---------|---------|
| `operators/reduce/sum/sum_op.cpp` | 修正 kernel 名 `sum_kernel` → `sum_dim_kernel`，对齐参数，添加 NPU block 配置 |
| `operators/normalization/softmax/softmax_op.cpp` | 修正 kernel 名 `softmax_kernel` → `softmax_kernel_inner`，对齐参数顺序，添加 `ONE_TILE_PER_CTA` |
| `operators/fusion/apply_rotary_pos_emb/apply_rotary_pos_emb_op.cpp` | 修正 kernel 名 `apply_rotary_pos_emb_kernel` → `rotary_embedding_kernel`，拆分为 Q/K 两次 launch |
| `operators/normalization/fused_add_rms_norm/fused_add_rms_norm.py` | 避免指针 in-place 运算，改用显式 offset 计算 |

---

## 关键修改点

### 1. sum_op.cpp - Kernel 名与参数对齐

**Before:**
```cpp
const TritonJITFunction &f = TritonJITFunction::get_instance("./sum.py", "sum_kernel");
int64_t tile_m = 4;
int64_t tile_n = 512;
// ...
f(stream, num_blocks, 1, 1, num_warps, num_stages,
  permuted_self, out, non_reduction_size, reduction_size,
  tile_m, tile_n, num_stages);  // 多余参数
```

**After:**
```cpp
const TritonJITFunction &f = TritonJITFunction::get_instance("./sum.py", "sum_dim_kernel");

#if defined(BACKEND_NPU)
    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 256;  // NPU 较小 block 避免 UB 溢出
    constexpr int num_warps = 1;
    constexpr int num_stages = 1;
#else
    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 512;
    constexpr int num_warps = 8;
    constexpr int num_stages = 2;
#endif

f(stream, num_blocks, 1, 1, num_warps, num_stages,
  permuted_self, out, non_reduction_size, reduction_size,
  BLOCK_M, BLOCK_N);  // 移除多余参数
```

**目的**：Python kernel `sum_dim_kernel(inp, out, M, N, BLOCK_M, BLOCK_N)` 只有 6 个参数，C++ 调用需严格匹配。

---

### 2. fused_add_rms_norm.py - NPU 指针运算兼容性

**Before (NPU 不兼容):**
```python
pid = tl.program_id(0)
X += pid.to(tl.int64) * x_stride_r  # NPU 不支持指针 +=
R += pid.to(tl.int64) * r_stride_r

# 后续使用
x = tl.load(X + cols, mask, other=0.0)
```

**After (NPU 兼容):**
```python
pid = tl.program_id(0)
# 显式计算行偏移，避免 in-place 指针运算
x_row_offset = pid * x_stride_r
r_row_offset = pid * r_stride_r

# 在 load/store 时加上偏移
x = tl.load(X + x_row_offset + cols, mask, other=0.0)
```

**目的**：NPU Triton (triton-ascend) 不支持对指针类型执行 `+=` 运算，必须改为在 load/store 时显式加偏移。

---

### 3. apply_rotary_pos_emb_op.cpp - 拆分 Q/K Launch

**Before:**
```cpp
// 试图一次 launch 同时处理 Q 和 K
f(stream, seq_len, num_heads, 1, num_warps, num_stages,
  q_contig, k_contig, cos_contig, sin_contig, q_out, k_out, ...);
```

**After:**
```cpp
// Python kernel 一次只处理一个 tensor，需分别 launch
// Launch for Q
f(stream, grid_q_x, grid_q_y, 1, num_warps, num_stages,
  q_out, q_contig, cos_expanded, sin_expanded,
  q_contig.stride(0), q_contig.stride(1), q_contig.stride(2),
  cos_expanded.stride(0), cos_expanded.stride(2),
  num_tokens, num_heads_q, BLOCK_N, BLOCK_H, head_dim);

// Launch for K
f(stream, grid_q_x, grid_k_y, 1, num_warps, num_stages,
  k_out, k_contig, cos_expanded, sin_expanded, ...);
```

**目的**：`rotary_embedding_kernel` 签名只接收单个 state tensor，C++ 需分别对 Q 和 K 调用。

---

## 行为变化

| 项目 | 修改前 | 修改后 |
|-----|-------|-------|
| sum 测试 | `AttributeError: module 'sum' has no attribute 'sum_kernel'` | 正常执行 |
| softmax 测试 | `AttributeError: module 'softmax' has no attribute 'softmax_kernel'` | 正常执行 |
| apply_rotary 测试 | `AttributeError: ... 'apply_rotary_pos_emb_kernel'` | 正常执行 |
| fused_add_rms_norm 测试 | `IncompatibleTypeErrorImpl: invalid operands of type pointer<fp32>` | 正常编译 |

---

## 验证方式

**1. 清除缓存并重新构建**

```bash
rm -rf ~/.triton/cache/
cmake --build build/ --parallel
```

**2. 运行 sum 算子测试**

```bash
(cd build/operators/reduce/sum && ./test_sum)
```

**预期输出**：
```
=== Test: sum_basic ===
Passed: YES
Max abs error: < 1e-4
```

**3. 运行 softmax 算子测试**

```bash
(cd build/operators/normalization/softmax && ./test_softmax)
```

**预期输出**：
```
=== Test: softmax_basic ===
Passed: YES
```

**4. 运行 fused_add_rms_norm 测试**

```bash
(cd build/operators/normalization/fused_add_rms_norm && ./test_fused_add_rms_norm)
```

**预期输出**：无 `IncompatibleTypeErrorImpl` 编译错误

---

## See Also

- **Debug 文档**：本修复落地了 [NPU 算子函数名不匹配排查](../debugging_index.md) 中的结论
- **CLAUDE.md**：已更新 NPU 后端约束说明，包含 `linalg.reduce` 限制和 block size 配置指南
- **相关 PR**：建议与 matmul workspace 修复、argmax 参数修复一同合入
