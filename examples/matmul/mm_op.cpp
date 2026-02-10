// ==============================================================================
// mm_op.cpp - Multi-backend Triton JIT Matrix Multiplication Example
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "mm_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

// ==============================================================================
//                         BACKEND DETECTION & HEADERS
// ==============================================================================

#if defined(BACKEND_NPU)
    // ----------------------------- NPU Backend -----------------------------
    #if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
        #include "torch_npu/csrc/core/npu/NPUStream.h"
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
        #warning "torch_npu headers not found, NPU stream support disabled"
    #endif

#elif defined(BACKEND_MUSA)
    // ----------------------------- MUSA Backend ----------------------------
    #include <musa_runtime.h>

#else
    // ----------------------- CUDA / IX Backend (Default) -------------------
    #include "c10/cuda/CUDAStream.h"

#endif

// ==============================================================================
//                         BACKEND-SPECIFIC TYPES & UTILITIES
// ==============================================================================

namespace {

// ------------------------------ Stream Types ---------------------------------

#if defined(BACKEND_NPU)
    using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
    using RawStream = musaStream_t;
#else
    using RawStream = CUstream;
#endif

// ----------------------------- Stream Getter ---------------------------------

inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& tensor) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif

#elif defined(BACKEND_MUSA)
    return nullptr;  // Default MUSA stream

#else  // CUDA / IX
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

// ==============================================================================
//                         KERNEL IMPLEMENTATION
// ==============================================================================

namespace my_ops {
using namespace triton_jit;

at::Tensor mm(const at::Tensor& a, const at::Tensor& b) {
    // ------------------------- Input Validation ------------------------------
    TORCH_CHECK(a.dim() == 2, "Expected 2D tensor for a");
    TORCH_CHECK(b.dim() == 2, "Expected 2D tensor for b");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions must match");

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    // ------------------------- Input Preparation -----------------------------
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();

    // ------------------------- Output Allocation -----------------------------
#if defined(BACKEND_MUSA)
    void* c_ptr = nullptr;
    size_t c_bytes = M * N * at::elementSize(a.scalar_type());
    musaError_t err = musaMalloc(&c_ptr, c_bytes);
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }
    auto opts = at::TensorOptions().dtype(a.scalar_type()).device(a.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor c = at::from_blob(c_ptr, {M, N}, deleter, opts);
#else
    at::Tensor c = at::empty({M, N}, a.options());
#endif

    // ------------------------- Kernel Parameters -----------------------------
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("mm.py"), "mm_kernel");

#if defined(BACKEND_NPU)
    // NPU-specific block sizes (smaller due to BiShengHIR constraints)
    constexpr int64_t BLOCK_M = 32;
    constexpr int64_t BLOCK_N = 32;
    constexpr int64_t BLOCK_K = 32;
    constexpr int64_t GROUP_M = 4;
    constexpr int num_warps = 1;
    constexpr int num_stages = 1;
#else
    // CUDA/MUSA/IX: Optimized block sizes
    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 32;
    constexpr int64_t GROUP_M = 8;
    constexpr int num_warps = 4;
    constexpr int num_stages = 2;
#endif

    // Grid calculation
    int64_t grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int64_t grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    unsigned int num_blocks = grid_m * grid_n;

    // ------------------------- Kernel Launch ---------------------------------
    c10::DeviceGuard guard(a.device());
    RawStream stream = get_device_stream(a);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      a_contig, b_contig, c,
      M, N, K,
      a_contig.stride(0), a_contig.stride(1),
      b_contig.stride(0), b_contig.stride(1),
      c.stride(0), c.stride(1),
      BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);

    return c;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(mm_example_ops, m) {
    m.def("mm(Tensor self, Tensor other) -> Tensor");
}

// Backend-specific dispatch key registration
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(mm_example_ops, PrivateUse1, m) {
        m.impl("mm", TORCH_FN(mm));
    }
#else
    TORCH_LIBRARY_IMPL(mm_example_ops, CUDA, m) {
        m.impl("mm", TORCH_FN(mm));
    }
#endif

}  // namespace my_ops
