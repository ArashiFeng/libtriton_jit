# NPU 后端 mm_kernel Workspace 分配修复

**修复日期**：2026-02-04
**关联问题**：[docs/debugging_index.md - NPU 后端 mm_kernel aicore exception](../debugging_index.md)

---

## 改动动机

C++ NPU 后端 `launch_kernel` 未解析 kernel 元数据中的 `workspace_size` 字段，导致 `set_system_args(ffts_addr, nullptr, nullptr)` 中 workspace 直接传 `nullptr`。当 kernel（如 mm_kernel）声明需要 workspace（`workspace_size: 12288`）时，MTE 指令访问空指针触发 aicore exception。

---

## 改动范围

| 文件 | 改动说明 |
|------|----------|
| `include/triton_jit/backends/npu_backend.h` | 1) `NpuKernelMetadata` 增加 `workspace_size` 字段；2) `load_kernel` 解析 JSON 中的 `workspace_size`；3) `launch_kernel` 签名增加 `workspace_size` 参数并实现 `rtMalloc` 分配逻辑 |
| `include/triton_jit/triton_kernel_impl.h` | 从 metadata 提取 `workspace_size` 并传递给 `Backend::launch_kernel` |

---

## 关键修改点

### 1. NpuKernelMetadata 增加 workspace_size 字段

```diff
 struct NpuKernelMetadata {
     unsigned int shared;
     std::string mix_mode;
     std::vector<NpuArgInfo> arg_layout;
+    size_t workspace_size = 0;           // Per-block workspace size in bytes
     ...
 };
```

**目的**：存储从 JSON 元数据解析出的 per-block workspace 大小，供 `launch_kernel` 使用。

### 2. launch_kernel 中分配 workspace 并传递给 set_system_args

```diff
-        // 1. Set system arguments
-        arg_buffer.set_system_args(ffts_addr, nullptr, nullptr);
+        // 1. Allocate workspace if needed
+        void* workspace_addr = nullptr;
+        if (workspace_size > 0) {
+            size_t total_workspace = workspace_size * blockNum;
+            rtError_t ws_ret = rtMalloc(&workspace_addr, total_workspace, RT_MEMORY_HBM, 0);
+            if (ws_ret != RT_ERROR_NONE) {
+                throw std::runtime_error(fmt::format(
+                    "rtMalloc workspace failed: {}, requested {} bytes",
+                    static_cast<int>(ws_ret), total_workspace));
+            }
+            LOG(INFO) << fmt::format(
+                "NPU workspace allocated: {} bytes ({} per block x {} blocks)",
+                total_workspace, workspace_size, blockNum);
+        }
+
+        // 2. Set system arguments (ffts, sync_lock, workspace)
+        arg_buffer.set_system_args(ffts_addr, nullptr, workspace_addr);
```

**目的**：根据 `workspace_size * blockNum` 计算总 workspace 大小，调用 `rtMalloc` 在 HBM 上分配内存，然后将有效地址传入 `set_system_args` 的第三个参数。

### 3. load_kernel 解析 workspace_size

```diff
+            // Parse workspace_size if present
+            if (meta_data.contains("workspace_size")) {
+                metadata.workspace_size = meta_data["workspace_size"].get<size_t>();
+                LOG(INFO) << fmt::format("Loaded workspace_size={} from metadata", metadata.workspace_size);
+            }
```

**目的**：从 Triton 编译产物的 JSON 元数据中读取 `workspace_size`，缓存到 `NpuKernelMetadata`。

---

## 行为变化

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| workspace 分配 | 始终传 `nullptr` | 根据 `workspace_size > 0` 动态分配 HBM 内存 |
| mm_kernel 执行 | 触发 `aicore exception`，错误码 507057 | 正常执行，结果与 Python 一致 |
| 日志输出 | 无 workspace 相关信息 | 输出 `NPU workspace allocated: ...` 和 `workspace=...B` |

---

## 验证方式

### 命令 1：编译并运行 test_mm_example

```bash
cmake --build build/ --target test_mm_example --parallel && \
build/examples/matmul/test_mm_example
```

**成功标准**：
- 无 `aicore exception` 或 `ACL stream synchronize failed` 错误
- 输出包含 `NPU workspace allocated: 12288 bytes` 或类似日志
- 测试通过，输出 `PASSED` 或 `max_diff` 在可接受范围内

### 命令 2：对比 Python baseline

```bash
# Python baseline（应已通过）
python examples/matmul/mm.py

# C++ 测试（修复后应与 Python 结果一致）
TORCH_CPP_LOG_LEVEL=INFO build/examples/matmul/test_mm_example 2>&1 | grep -E "(workspace|PASSED|max_diff)"
```

**成功标准**：
- 日志中出现 `Loaded workspace_size=12288 from metadata`
- C++ 与 Python 计算结果的 `max_diff` 数值一致

---

## See Also

本修复落地了 [docs/debugging_index.md](../debugging_index.md) 中 **"[2026-02-04] NPU 后端 mm_kernel aicore exception"** 的以下结论：

- **关键线索 #1**：`"workspace_size": 12288` — 在 `load_kernel` 中解析此字段
- **关键线索 #2**：`set_system_args(..., nullptr, nullptr)` — 修改为传入有效 `workspace_addr`
- **修复方案**：完整实现了文档中列出的 3 个修改点

---

## 已知限制

当前实现存在 workspace 内存泄漏问题（kernel 执行为异步，无法在 `launch_kernel` 返回后立即释放）。后续需实现 workspace pool 管理机制。参见代码中 `TODO: implement workspace pool to avoid memory leak`。
