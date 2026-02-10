# Debugging Index

本文档记录 triton_jit 项目开发过程中遇到的典型问题及排障复盘，便于后续开发参考。

---

## [2026-02-04] NPU 后端 mm_kernel aicore exception 排障复盘

### 现象
Python 直接调用 `python examples/matmul/mm.py` 正常（float16/float32 均通过），但 C++ 调用 `build/examples/matmul/test_mm_example` 触发 `aicore exception`，报错 `The DDR address of the MTE instruction is out of range`，最终 `ACL stream synchronize failed, error code: 507057`。

### 定位过程

**Step 1：排除数据类型差异**
在 `mm.py` 中增加 float32 测试，验证 Python 端是否受 dtype 影响：
```python
# 测试与 C++ 相同的配置
test_mm(torch.float32, "float32", block_m=32, block_n=32, block_k=32, group_m=4,
        matrix_m=512, matrix_k=256, matrix_n=512)
```
结果：Python 端 float32 + 相同 block size + 相同矩阵尺寸均正常 → 排除 dtype/block size 问题。

**Step 2：对比 kernel 元数据**
查看 Triton 编译产物：
```bash
find ~/.triton -name "*.json" -path "*mm_kernel*" | head -5 | xargs cat
```
发现关键字段：
```json
{
  "workspace_size": 12288,
  "arg_layout": [{"type": "ptr", "dtype": "fp32"}, ...]
}
```
→ kernel 声明需要 12KB workspace。

**Step 3：定位 Python 驱动的 workspace 处理**
```bash
grep -rn "workspace" /root/miniconda3/envs/triton/lib/python3.10/site-packages/triton/backends/ascend/driver.py
```
发现 Python 端在 `driver.py:577-582` 有分配逻辑：
```cpp
uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
ret = rtMalloc(reinterpret_cast<void **>(&workspace_addr),
               totalWorkSpaceSize, RT_MEMORY_HBM, ModuleId);
```

**Step 4：检查 C++ 端对应实现**
查看 `include/triton_jit/backends/npu_backend.h:319`：
```cpp
arg_buffer.set_system_args(ffts_addr, nullptr, nullptr);  // workspace 直接传 nullptr!
```
→ 确认 C++ 未分配 workspace。

### 关键线索
1. `"workspace_size": 12288` — kernel 需要 workspace，C++ 未提供
2. `set_system_args(..., nullptr, nullptr)` — 直接证据
3. `MTE instruction is out of range, fixp_error0: 0x3000047` — MTE 访问空指针越界

### 根因
C++ NPU 后端 `launch_kernel` 未读取 `workspace_size`，未调用 `rtMalloc` 分配 workspace 内存，导致 kernel 运行时 MTE 指令访问空指针。

### 修复方案

**修改点**：`include/triton_jit/backends/npu_backend.h`

1. `NpuKernelMetadata` 增加 `workspace_size` 字段：
```cpp
struct NpuKernelMetadata {
    unsigned int shared;
    std::string mix_mode;
    std::vector<NpuArgInfo> arg_layout;
    size_t workspace_size = 0;  // 新增
};
```

2. `load_kernel` 时从 JSON 解析 `workspace_size`：
```cpp
if (meta_data.contains("workspace_size")) {
    metadata.workspace_size = meta_data["workspace_size"].get<size_t>();
}
```

3. `launch_kernel` 中分配 workspace：
```cpp
void* workspace_addr = nullptr;
if (metadata->workspace_size > 0) {
    size_t total_workspace = metadata->workspace_size * blockNum;
    rtError_t ret = rtMalloc(&workspace_addr, total_workspace, RT_MEMORY_HBM, 0);
    if (ret != RT_ERROR_NONE) {
        throw std::runtime_error("rtMalloc workspace failed");
    }
}
arg_buffer.set_system_args(ffts_addr, nullptr, workspace_addr);
```

### 验证命令
```bash
# 重新编译
cmake --build build/ --target test_mm_example --parallel

# 运行测试，预期无 aicore exception 且结果正确
build/examples/matmul/test_mm_example

# 对比验证：Python 与 C++ 结果一致性
python examples/matmul/mm.py  # 作为 baseline
```

### 状态
- [x] 已修复（workspace 分配部分）

---

## [2026-02-04] NPU 后端 mm_kernel 结果错误（global scratch pointer 多余参数）

### 现象
workspace 分配问题修复后，mm_kernel 不再崩溃，但计算结果错误：
```
Result[0,0]: -0
Expected[0,0]: 59.5625
Max difference: 79.75
```
日志显示参数数量不匹配：
```
Parsed signature '...' -> 12 runtime args
NPU args debug: num_args=13
```

### 定位过程

**Step 1：分析日志中的参数数量差异**
```bash
TORCH_CPP_LOG_LEVEL=INFO ./test_mm_example 2>&1 | grep -E "(runtime args|num_args)"
```
输出：
```
Parsed signature '*fp16,*fp16,*fp16,i64,i64,i64,i64,i64,i64,i64,i64,i64,32,32,32,4' -> 12 runtime args
NPU args debug: num_args=13
```
→ signature 解析 12 个参数，但实际传递了 13 个。

**Step 2：追踪多余参数来源**
```bash
grep -n "append_global_scratch" include/triton_jit/triton_jit_function_impl.h
```
定位到第 292 行调用了 `append_global_scratch()`。

检查该函数实现 (`triton_jit_function_impl.h:222-225`)：
```cpp
void append_global_scratch() {
    void* global_scratch = nullptr;
    this->buf.push_arg(global_scratch);  // 添加第 13 个参数
}
```
→ C++ 端无条件添加 global scratch pointer。

**Step 3：验证 Python 驱动是否传递 global scratch**
```bash
grep -n "global_scratch\|scratch" /root/miniconda3/envs/triton/lib/python3.10/site-packages/triton/backends/ascend/driver.py
```
→ Python NPU 驱动不传递 global scratch，参数布局为：
- 3 个系统参数 (ffts, syncBlockLock, workspace)
- 用户参数（signature 中非 constexpr 的参数）
- grid 维度

**Step 4：确认 npu_backend.h 的兼容性处理**
查看 `npu_backend.h:306-314`：
```cpp
if (num_args == layout.size() + 1) {
    // Triton 3.3+ appends a global scratch pointer not present in the signature.
    layout.push_back(NpuArgInfo{NpuArgType::POINTER, sizeof(void*)});
}
```
→ 该逻辑试图兼容 Triton 3.3+ global scratch，但 NPU 驱动根本不使用它。

### 关键线索
1. `Parsed signature -> 12 runtime args` vs `num_args=13` — 参数数量不匹配
2. `append_global_scratch()` 添加 `nullptr` — C++ 端多传一个参数
3. Python `driver.py` 不包含 global scratch — NPU 驱动不需要此参数

### 根因
C++ 端 `triton_jit_function_impl.h` 中 `append_global_scratch()` 对所有后端无条件添加 global scratch pointer，但 NPU 驱动不使用 global scratch（workspace 通过 `set_system_args` 单独传递）。多余的第 13 个参数导致参数布局错位，kernel 读取错误数据。

### 修复方案

**修改点 1**：`include/triton_jit/triton_jit_function_impl.h:291-295`

为 NPU 后端禁用 global scratch：
```cpp
#if !defined(BACKEND_NPU)
        // global scratch: introduced in triton 3.3
        // NPU backend does not use global scratch (handled differently via workspace)
        handler.append_global_scratch();
#endif
```

**修改点 2**：`include/triton_jit/backends/npu_backend.h:303-314`

移除对 Triton 3.3+ global scratch 的特殊处理：
```cpp
// NPU does not use global scratch pointer (workspace is handled separately)
// Just verify arg count matches
if (num_args != 0 && num_args != layout.size()) {
    throw std::runtime_error(fmt::format(
        "launch_kernel: arg count mismatch (layout={}, args={})",
        layout.size(), num_args));
}
```

### 验证命令
```bash
# 重新编译
cmake --build build/ --target test_mm_example --parallel

# 运行测试，预期结果正确
TORCH_CPP_LOG_LEVEL=INFO ./build/examples/matmul/test_mm_example

# 验证参数数量一致
TORCH_CPP_LOG_LEVEL=INFO ./build/examples/matmul/test_mm_example 2>&1 | grep -E "(runtime args|num_args)"
# 预期输出：12 runtime args, num_args=12
```

### 状态
- [x] 已修复

---

## [2026-02-04] NPU 后端 reduce/normalization/fusion 算子编译跳过问题

### 现象
CMake 配置 NPU 后端时，reduce（sum/max/argmax/topk）、normalization（softmax/rms_norm/fused_add_rms_norm）、fusion（apply_rotary_pos_emb/rwkv_*）共 10 个算子被跳过编译，输出：
```
[NPU] Skipping reduce operators (linalg.reduce not supported)
[NPU] Skipping normalization operators (uses reduce internally)
[NPU] Skipping fusion operators (may use reduce operations)
```

### 定位过程

**Step 1：确认跳过原因**
查看 `operators/CMakeLists.txt:136-149`：
```cmake
if(NOT BACKEND STREQUAL "NPU")
    add_subdirectory(reduce)
    add_subdirectory(normalization)
    add_subdirectory(fusion)
else()
    message(STATUS "[NPU] Skipping reduce operators...")
endif()
```
→ 构建系统硬编码跳过 NPU 后端的这些算子。

**Step 2：分析原始 kernel 实现**
检查 `operators/reduce/sum/sum.py` 中的 `tl.sum(acc, axis=1)` 调用：
→ BiShengHIR 编译器不支持 `linalg.reduce` IR，导致编译失败。

**Step 3：参考 FlagGems Ascend 实现**
查看 `/data/baai_user_home/chwork/FlagGems/src/flag_gems/runtime/backend/_ascend/ops/`：
```python
# FlagGems Ascend 采用两阶段归约
@triton.jit
def amax_kernel_1(inp, mid, M, BLOCK_SIZE):
    max_val = tl.max(inp_val, axis=0)  # block 内归约
    tl.store(mid_ptr, max_val)

@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID):
    max_val = tl.max(mid_val, axis=0)  # 汇总阶段
    tl.store(out, max_val)
```
→ FlagGems 通过分块归约规避全局 `linalg.reduce`。

### 关键线索
1. `if(NOT BACKEND STREQUAL "NPU")` — CMake 硬编码跳过
2. `tl.sum(acc, axis=1)` — 编译为 `linalg.reduce`，NPU 不支持
3. FlagGems `amax_kernel_1` + `amax_kernel_2` — 两阶段策略可行

### 根因
原始 reduce/normalization kernel 使用全局归约操作（`tl.sum`/`tl.max` 全轴），BiShengHIR 编译器不支持对应的 `linalg.reduce` IR。构建系统直接跳过这些算子而非修改实现。

### 修复方案
参考 FlagGems Ascend 实现，采用以下策略：

1. **CMakeLists.txt**: 移除 NPU 条件跳过
2. **Reduce 算子**: 实现两阶段归约（分块 → 汇总）
3. **Normalization 算子**: 采用在线迭代算法（分块累加 → 最终归约）
4. **Fusion 算子**: 调整 block size，添加设备无关处理

详细修改见：[docs/fixing/npu_23ops_full_support.md](./fixing/npu_23ops_full_support.md)

### 验证命令
```bash
# 验证不再跳过算子
cmake -S . -B build/ -DBACKEND=NPU 2>&1 | grep -E "Skipping|add_subdirectory"
# 预期：无 "Skipping" 输出

# 全量测试
python tests/run_all_tests.py --backend NPU --build-dir build
# 预期：23/23 算子通过
```

### 状态
- [x] 已修复

---

## [2026-02-04] NPU 后端 addmm UB 溢出 + argmax linalg.reduce 不支持排障

### 现象

1. **addmm 测试**：BiShengHIR 编译失败，报 UB 溢出：
```
ub overflow, requires 2097152 bits while 1572864 bits available!
(possible reason: tiling basic block is too large...)
```

2. **argmax 测试**：编译失败，报 `linalg.reduce` 非法：
```
error: failed to legalize operation 'linalg.reduce' that was explicitly marked illegal
[ConvertLinalgRToBinary] encounters error: Failed to run BiShengHIR pipeline
```

### 定位过程

**Step 1：addmm UB 溢出分析**

对比 `mm_op.cpp` 与 `addmm_op.cpp` 的 block 配置：
```bash
grep -A5 "BLOCK_M = " operators/matmul/mm/mm_op.cpp
grep -A5 "BLOCK_M = " operators/matmul/addmm/addmm_op.cpp
```

发现 `mm_op.cpp` 有 NPU 专用配置（32×32），而 `addmm_op.cpp` 所有后端共用 64×64：
```cpp
// addmm_op.cpp (修复前) - 无 NPU 区分
constexpr int64_t BLOCK_M = 64;
constexpr int64_t BLOCK_N = 64;
```

**Step 2：argmax 函数名与参数不匹配**

首次运行报 `AttributeError: no attribute 'argmax_kernel'`：
```bash
grep "@triton.jit" -A1 operators/reduce/argmax/argmax.py
```
→ 实际函数名是 `argmax_dim_kernel`，非 `argmax_kernel`。

修复后仍报参数不匹配：
```
AssertionError: number of argument mismatch: Actual(6), Function Definition(7)
```

对比 Python 定义与 C++ 调用：
```python
# argmax.py 定义 7 个参数
def argmax_dim_kernel(inp, out_index, M, N, K, BLOCK_M, BLOCK_N):
```
```cpp
// argmax_op.cpp 调用只传 6 个（缺 K）
f(stream, num_blocks, 1, 1, num_warps, num_stages,
  permuted.view({M, N}), out, M, N, BLOCK_M, BLOCK_N);  // 缺少 K!
```

**Step 3：linalg.reduce 根因实锤**

在 BiShengHIR 二进制中搜索证据：
```bash
strings /usr/local/Ascend/cann-8.5.0/bin/bishengir-compile | grep "linalg.reduce"
```
输出：
```
linalg.reduce
failed to create linalg.reduce operation for reduction
```

```bash
strings /usr/local/Ascend/cann-8.5.0/bin/bishengir-compile | grep "backend illegal"
```
输出：
```
found an op that was marked as backend illegal
' that was explicitly marked illegal
```

→ **实锤**：BiShengHIR 编译器内部硬编码了"非法操作列表"，`linalg.reduce` 在此列表中。

### 关键线索

1. `ub overflow, requires 2097152 bits while 1572864 bits available` — BLOCK 64×64 超出 NPU UB 容量 (~192KB)
2. `argmax_dim_kernel` vs `argmax_kernel` — 函数名不匹配
3. `Actual(6), Function Definition(7)` — 缺少 K 参数
4. **`found an op that was marked as backend illegal`** — BiShengHIR 明确标记 `linalg.reduce` 非法
5. `tl.max()`, `tl.argmax()`, `tl.sum()` → 编译为 `linalg.reduce` MLIR 操作

### 根因

1. **addmm UB 溢出**：block 配置 64×64 超出 NPU Unified Buffer 限制，需降为 32×32
2. **argmax 编译失败**：BiShengHIR 编译器内部将 `linalg.reduce` 硬编码为非法操作（设计限制，非 bug）

**编译流程**：
```
Triton Python (tl.max/tl.argmax/tl.sum)
       ↓
triton-adapter-opt --triton-to-linalg  ← 转换为 linalg.reduce
       ↓
bishengir-compile  ← 在这里标记 linalg.reduce 为 illegal 并拒绝
       ↓
MLIRCompilationError: failed to legalize operation 'linalg.reduce'
```

### 修复方案

**修复 1：addmm UB 溢出** (`operators/matmul/addmm/addmm_op.cpp`)
```cpp
#if defined(BACKEND_NPU)
    constexpr int64_t BLOCK_M = 32;
    constexpr int64_t BLOCK_N = 32;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 1;
    constexpr int num_stages = 1;
#else
    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 4;
    constexpr int num_stages = 2;
#endif
```

**修复 2：argmax 函数名与参数** (`operators/reduce/argmax/argmax_op.cpp`)
```cpp
// 修复函数名
const TritonJITFunction& f = TritonJITFunction::get_instance(
    std::string("argmax.py"), "argmax_dim_kernel");  // 原: argmax_kernel

// 添加缺失的 K 参数
constexpr int64_t K = 1;
f(stream, num_blocks, 1, 1, num_warps, num_stages,
  permuted.view({M, N}), out, M, N, K, BLOCK_M, BLOCK_N);  // 新增 K
```

**修复 3：argmax linalg.reduce 问题** — **NPU 后端设计限制，无法修复**

BiShengHIR 不支持 `linalg.reduce`，可选方案：
- 方案 A：NPU 上使用 PyTorch 原生 `torch.argmax()` 替代 Triton 实现
- 方案 B：参考 FlagGems 实现两阶段归约（需重写 kernel，规避 `linalg.reduce`）

### 验证命令

```bash
# 验证 addmm（预期通过）
cmake --build build/ --target test_addmm --parallel
(cd build/operators/matmul/addmm && ./test_addmm)

# 验证 argmax linalg.reduce 限制（预期失败）
cmake --build build/ --target test_argmax --parallel
(cd build/operators/reduce/argmax && ./test_argmax)
# 预期错误: failed to legalize operation 'linalg.reduce'

# 实锤命令：确认 BiShengHIR 标记 linalg.reduce 为非法
strings /usr/local/Ascend/cann-8.5.0/bin/bishengir-compile | grep "linalg.reduce"
strings /usr/local/Ascend/cann-8.5.0/bin/bishengir-compile | grep "backend illegal"
```

### 状态
- [x] addmm UB 溢出已修复
- [x] argmax 函数名/参数已修复
- [ ] argmax linalg.reduce — NPU 后端设计限制，Triton reduce 类算子无法在 NPU 上运行

---

## [2026-02-04] NPU 后端 fused_add_rms_norm 参数不匹配 + UB 溢出排障

### 现象

运行 `test_fused_add_rms_norm` 时出现两个问题：

1. **编译时类型错误**：
```
IncompatibleTypeErrorImpl('invalid operands of type pointer<fp32> and triton.language.int32')
```
错误指向 kernel 第 23 行 `x_row_offset = pid * x_stride_r`。

2. **基础测试通过后，benchmark 测试 UB 溢出**：
```
ub overflow, requires 3145728 bits while 1572864 bits available!
(possible reason: tiling basic block is too large...)
```

### 定位过程

**Step 1：分析类型错误来源**

错误提示 `pointer<fp32>` 与 `int32` 不兼容，说明某个参数类型被错误解析：
```bash
# 查看 Python kernel 参数定义
head -30 operators/normalization/fused_add_rms_norm/fused_add_rms_norm.py
```

Python kernel 参数顺序：
```python
def fused_add_rms_norm_kernel(
    X,              # 1. ptr
    R,              # 2. ptr
    W,              # 3. ptr
    x_stride_r,     # 4. int  ← 错误发生位置
    x_stride_c,     # 5. int
    r_stride_r,     # 6. int
    r_stride_c,     # 7. int
    N,              # 8. int
    eps,            # 9. float
    BLOCK_SIZE,     # 10. constexpr
)
```

**Step 2：对比 C++ 调用端参数**
```bash
grep -A20 "f(stream" operators/normalization/fused_add_rms_norm/fused_add_rms_norm_op.cpp
```

发现 C++ 代码传递了额外的 `output` 和 `residual_out` tensor：
```cpp
f(stream, n_rows, 1, 1, num_warps, num_stages,
  x_flat, res_flat, weight,
  output,              // ← 这被当成了 x_stride_r (指针→整数类型错误!)
  residual_out,        // ← 这被当成了 x_stride_c
  x_flat.stride(0), output.stride(0),
  hidden_size, static_cast<float>(eps),
  BLOCK_SIZE);
```

→ **Python kernel 是 in-place 操作**，不需要 `output`/`residual_out`。C++ 多传了 2 个 tensor 指针，导致后续参数偏移，Triton 把指针解释成整数。

**Step 3：分析 UB 溢出**

修复参数后，基础测试通过，但 benchmark（hidden_size=4096）失败：
```bash
# 查看 BLOCK_SIZE 计算逻辑
grep -A5 "BLOCK_SIZE" operators/normalization/fused_add_rms_norm/fused_add_rms_norm_op.cpp
```

发现 BLOCK_SIZE 无上限：
```cpp
int64_t BLOCK_SIZE = 1;
while (BLOCK_SIZE < hidden_size) BLOCK_SIZE *= 2;
// hidden_size=4096 → BLOCK_SIZE=4096
```

UB 内存计算：`3145728 bits = 384KB`，超过 NPU UB 上限 `192KB`。

### 关键线索

1. `invalid operands of type pointer<fp32> and triton.language.int32` — 指针被当作整数
2. `output, residual_out` 作为 kernel 参数传入 — C++ 与 Python 参数不匹配
3. **Python kernel 是 in-place**：`tl.store(R + ..., x)` 和 `tl.store(X + ..., y)` — 不需要额外输出
4. `requires 3145728 bits while 1572864 bits available` — BLOCK_SIZE=4096 超出 UB

### 根因

1. **参数不匹配**：C++ 端假设 kernel 需要独立的 output tensor，传递了 `output` 和 `residual_out` 参数，但 Python kernel 采用 in-place 设计，直接修改输入的 X 和 R。多余参数导致类型错位。

2. **UB 溢出**：BLOCK_SIZE 直接取 next_power_of_2(hidden_size)，对于大 hidden_size (4096) 超出 NPU 192KB UB 限制。

### 修复方案

**修复 1：参数对齐** (`fused_add_rms_norm_op.cpp`)

移除 `output`/`residual_out` 参数，使用 clone + in-place 模式：
```cpp
// 修复前：创建独立输出 tensor 并传入 kernel
at::Tensor output = at::empty_like(x_flat);
at::Tensor residual_out = at::empty_like(x_flat);
f(..., x_flat, res_flat, weight, output, residual_out, ...);

// 修复后：clone 输入 tensor，让 kernel in-place 修改
at::Tensor x_flat = input.view({n_rows, hidden_size}).contiguous().clone();
at::Tensor res_flat = residual.view({n_rows, hidden_size}).contiguous().clone();

// 参数与 Python kernel 完全对齐
f(stream, n_rows, 1, 1, num_warps, num_stages,
  x_flat,                        // X: in-place 修改
  res_flat,                      // R: in-place 修改
  weight,                        // W
  x_flat.stride(0),              // x_stride_r
  x_flat.stride(1),              // x_stride_c
  res_flat.stride(0),            // r_stride_r
  res_flat.stride(1),            // r_stride_c
  hidden_size,                   // N
  static_cast<float>(eps),       // eps
  BLOCK_SIZE);                   // BLOCK_SIZE

return std::make_tuple(x_flat.view(orig_shape), res_flat.view(orig_shape));
```

**修复 2：限制 BLOCK_SIZE** (`fused_add_rms_norm_op.cpp`)
```cpp
int64_t BLOCK_SIZE = 1;
while (BLOCK_SIZE < hidden_size) BLOCK_SIZE *= 2;

#if defined(BACKEND_NPU)
    // NPU UB ~192KB，限制 BLOCK_SIZE 避免溢出
    // 1024 元素 * 4 bytes * 3 数组 * 2 pass ≈ 24KB 安全
    constexpr int64_t NPU_MAX_BLOCK_SIZE = 1024;
    BLOCK_SIZE = std::min(BLOCK_SIZE, NPU_MAX_BLOCK_SIZE);
    constexpr int num_warps = 1;
#else
    constexpr int num_warps = 4;
#endif
```

### 验证命令

```bash
# 重新编译
cmake --build build/ --target test_fused_add_rms_norm --parallel

# 运行测试（从构建目录）
(cd build/operators/normalization/fused_add_rms_norm && ./test_fused_add_rms_norm)

# 预期输出
# [PASS] fused_add_rms_norm_basic
# [PASS] fused_add_rms_norm_with_weight
# [PASS] fused_add_rms_norm_benchmark  ← 之前失败

# 验证参数布局
TORCH_CPP_LOG_LEVEL=INFO (cd build/operators/normalization/fused_add_rms_norm && ./test_fused_add_rms_norm) 2>&1 | grep "arg_layout"
# 预期：9 runtime args（3 ptr + 4 stride + N + eps）
```

### 状态
- [x] 参数不匹配已修复
- [x] UB 溢出已修复（限制 BLOCK_SIZE ≤ 1024）

---

## 问题模板

```markdown
## [日期] 问题标题

### 现象
描述问题表现

### 定位过程
排查步骤与命令

### 关键线索
日志/代码中的关键证据

### 根因
最终确认的原因

### 修复方案
具体修改点与代码

### 验证命令
如何验证修复有效

### 状态
- [ ] 待修复 / [x] 已修复
```
