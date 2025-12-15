// ==============================================================================
// add_op.cpp - Multi-backend Triton JIT Add Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "add_op.h"
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
    // TODO: Add MUSA stream headers when available
    #define HAS_TORCH_MUSA 0
    #warning "MUSA backend stream support not yet implemented"

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
    using RawStream = void*;  // TODO: Replace with actual MUSA stream type
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
    // TODO: Implement MUSA stream getter
    return nullptr;

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

at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
    // ------------------------- Input Preparation -----------------------------
    auto res = torch::broadcast_tensors({a_, b_});
    res[0] = res[0].contiguous();
    res[1] = res[1].contiguous();
    const at::Tensor &a = res[0];
    const at::Tensor &b = res[1];

    // ------------------------- Output Allocation -----------------------------
    at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    at::Tensor out = at::empty(a.sizes(),
        at::TensorOptions().dtype(out_dtype).device(a.device()));

    // ------------------------- Kernel Parameters -----------------------------
    const TritonJITFunction &f = TritonJITFunction::get_instance(
        std::string("add.py"), "binary_pointwise_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    // ------------------------- Kernel Launch ---------------------------------
    c10::DeviceGuard guard(out.device());
    RawStream stream = get_device_stream(a);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);

    return out;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(my_ops, m) {
    m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

// Backend-specific dispatch key registration
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    // NPU and MUSA use PrivateUse1 dispatch key
    TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
        m.impl("add_tensor", TORCH_FN(add_tensor));
    }
#else
    // CUDA and IX use CUDA dispatch key
    TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
        m.impl("add_tensor", TORCH_FN(add_tensor));
    }
#endif

}  // namespace my_ops
