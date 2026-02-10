# NPU 后端 23 算子全量支持修复

**日期**: 2026-02-04
**关联问题**: [docs/debugging_index.md](../debugging_index.md) - NPU 后端算子兼容性问题
**状态**: ✅ 已完成

---

## 改动动机

原有实现中，NPU 后端因 BiShengHIR 编译器不支持 `linalg.reduce` 操作，导致 reduce/normalization/fusion 三类共 10 个算子被跳过编译。参考 FlagGems Ascend 专用实现，采用**两阶段归约**和**在线迭代算法**重写核心 kernel，使所有 23 个算子均可在 NPU 后端编译运行。

---

## 改动范围

### 构建配置
| 文件 | 改动说明 |
|------|----------|
| `operators/CMakeLists.txt` | 移除 NPU 条件跳过，启用 reduce/normalization/fusion 三类算子 |

### Reduce 算子（4 个）
| 文件 | 改动说明 |
|------|----------|
| `operators/reduce/sum/sum.py` | 实现两阶段归约：`sum_kernel_1` 分块求和 + `sum_kernel_2` 汇总 |
| `operators/reduce/max/max.py` | 实现两阶段归约：`max_kernel_1` 分块求最大 + `max_kernel_2` 汇总 |
| `operators/reduce/argmax/argmax.py` | 实现两阶段归约：保存中间值和索引，二次归约取最终 argmax |
| `operators/reduce/topk/topk.py` | 保留原实现（使用 PyTorch fallback）|

### Normalization 算子（3 个）
| 文件 | 改动说明 |
|------|----------|
| `operators/normalization/softmax/softmax.py` | 采用在线 softmax 算法，分 inner/non_inner 两种 kernel |
| `operators/normalization/rms_norm/rms_norm.py` | 迭代累加方差，避免全局归约 |
| `operators/normalization/fused_add_rms_norm/fused_add_rms_norm.py` | 基于 FlagGems Ascend 模式，两遍扫描 |

### Fusion 算子（3 个）
| 文件 | 改动说明 |
|------|----------|
| `operators/fusion/apply_rotary_pos_emb/apply_rotary_pos_emb.py` | 基于 FlagGems rotary_embedding.py 重构 |
| `operators/fusion/rwkv_ka_fusion/rwkv_ka_fusion.py` | 添加分块迭代，设备无关处理 |
| `operators/fusion/rwkv_mm_sparsity/rwkv_mm_sparsity.py` | 调整 BLOCK 大小为 32，适配 NPU |

---

## 关键修改点

### 1. CMakeLists.txt - 启用所有算子

```diff
-if(NOT BACKEND STREQUAL "NPU")
-    add_subdirectory(reduce)
-    add_subdirectory(normalization)
-    add_subdirectory(fusion)
-else()
-    message(STATUS "[NPU] Skipping reduce operators...")
-endif()
+# NPU backend: Enable all operators for validation
+add_subdirectory(reduce)
+add_subdirectory(normalization)
+add_subdirectory(fusion)
```

**目的**: 移除 NPU 后端的条件编译限制，使 reduce/normalization/fusion 算子参与构建。

### 2. sum.py - 两阶段归约策略

```python
# Before: 单 kernel 全局归约（NPU 不支持）
@triton.jit
def sum_kernel(...):
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in tl.range(0, N, BLOCK_N, STAGE):
        ...
    out = tl.sum(acc, axis=1)  # ❌ 编译失败：linalg.reduce

# After: 两阶段归约（NPU 兼容）
@triton.jit
def sum_kernel_1(inp, mid, M, BLOCK_SIZE):
    """第一阶段：分块求和"""
    block_sum = tl.sum(inp_val, axis=0)
    tl.store(mid_ptr, block_sum)

@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID):
    """第二阶段：汇总分块结果"""
    final_sum = tl.sum(mid_val, axis=0)
    tl.store(out, final_sum)
```

**目的**: 将全局归约拆分为两阶段，每阶段只在 block 内归约，规避 BiShengHIR 的 `linalg.reduce` 限制。

### 3. softmax.py - 在线迭代算法

```python
# Before: 单次加载全行归约
row_max = tl.max(row, axis=0)
numerator = tl.exp(row - row_max)
denominator = tl.sum(numerator, axis=0)  # ❌ 全局 sum

# After: 在线 softmax（分块迭代）
for start_n in range(0, N, TILE_N):
    inp = tl.load(input_ptr + n_offsets, mask=mask, other=float("-inf"))
    m_new = tl.maximum(m, inp)
    # 在线更新：z = z * exp(m - m_new) + exp(inp - m_new)
    z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
    m = m_new
```

**目的**: 采用数值稳定的在线 softmax 算法，分块迭代计算 max 和 sum(exp)，避免一次性全局归约。

---

## 行为变化

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| **NPU reduce 编译** | ❌ 跳过，CMake 输出 "[NPU] Skipping reduce operators" | ✅ 正常编译 |
| **sum/max/argmax 执行** | N/A（未编译） | 两阶段 kernel 顺序执行 |
| **softmax 内存** | 单次加载全行 | 分块迭代，TILE_N ≤ 4096 |
| **Grid 大小** | 无限制 | NPU 上限 4096 blocks |
| **BLOCK_SIZE** | 最大 next_power_of_2(N) | NPU 上限 8192 |

---

## 验证方式

### 1. 构建验证
```bash
cmake -S . -B build/ -DPython_ROOT="$(which python)/../.." -DBACKEND=NPU
cmake --build build/ --parallel
```

**成功标准**:
- 无 "[NPU] Skipping" 输出
- 所有 23 个算子目标编译成功

### 2. 单算子测试
```bash
# Reduce 算子
TORCH_CPP_LOG_LEVEL=INFO ./build/operators/reduce/sum/test_sum
TORCH_CPP_LOG_LEVEL=INFO ./build/operators/reduce/max/test_max

# Normalization 算子
TORCH_CPP_LOG_LEVEL=INFO ./build/operators/normalization/softmax/test_softmax
TORCH_CPP_LOG_LEVEL=INFO ./build/operators/normalization/rms_norm/test_rms_norm
```

**成功标准**:
- 输出 "Results match: YES"
- Max difference < 0.01（float16 精度）

### 3. 全量测试
```bash
python tests/run_all_tests.py --backend NPU --build-dir build
```

**成功标准**: 23/23 算子通过

---

## See Also

- **Debug 线索落地**: [docs/debugging_index.md](../debugging_index.md) 中 "NPU linalg.reduce 不支持" 问题，本次通过 FlagGems Ascend 两阶段策略解决
- **FlagGems 参考**: `/data/baai_user_home/chwork/FlagGems/src/flag_gems/runtime/backend/_ascend/ops/`
- **NPU 兼容性矩阵**: [docs/npu_compatibility.md](./npu_compatibility.md)

---

## 附录：FlagGems Ascend 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| AIV 核数 | 40 | 910B3/B4 |
| BLOCK_SIZE 上限 | 8192 | 单 block 最大元素 |
| Grid 上限 | 4096 | 单维度最大 block 数 |
| num_warps | 1-4 | NPU 无 warp 概念，设为 1 |
