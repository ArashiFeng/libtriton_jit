// ==============================================================================
// fused_add_rms_norm_op.cpp - Multi-backend Fused Add + RMS Normalization
// ==============================================================================

#include "fused_add_rms_norm_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#if defined(BACKEND_NPU)
    #if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
        #include "torch_npu/csrc/core/npu/NPUStream.h"
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif
#elif defined(BACKEND_MUSA)
    #include <musa_runtime.h>
#else
    #include "c10/cuda/CUDAStream.h"
#endif

namespace {

#if defined(BACKEND_NPU)
    using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
    using RawStream = musaStream_t;
#else
    using RawStream = CUstream;
#endif

inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& tensor) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif
#elif defined(BACKEND_MUSA)
    return nullptr;
#else
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

namespace my_ops {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm(
    const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& weight, double eps) {

    TORCH_CHECK(input.sizes() == residual.sizes(), "Input and residual must have same shape");
    TORCH_CHECK(input.size(-1) == weight.size(0), "Hidden dim must match weight size");

    auto orig_shape = input.sizes().vec();
    int64_t hidden_size = input.size(-1);
    int64_t n_rows = input.numel() / hidden_size;

    // The kernel operates in-place, so we need mutable copies
    at::Tensor x_flat = input.view({n_rows, hidden_size}).contiguous().clone();
    at::Tensor res_flat = residual.view({n_rows, hidden_size}).contiguous().clone();

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fused_add_rms_norm.py"), "fused_add_rms_norm_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < hidden_size) BLOCK_SIZE *= 2;

#if defined(BACKEND_NPU)
    // NPU UB is ~192KB, limit BLOCK_SIZE to avoid overflow
    // With fp32: 1024 elements * 4 bytes * 3 arrays (x, r, w) * 2 passes ≈ 24KB safe margin
    constexpr int64_t NPU_MAX_BLOCK_SIZE = 1024;
    BLOCK_SIZE = std::min(BLOCK_SIZE, NPU_MAX_BLOCK_SIZE);
    constexpr int num_warps = 1;
#else
    constexpr int num_warps = 4;
#endif
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(input.device());
    RawStream stream = get_device_stream(input);

    // Kernel signature:
    // fused_add_rms_norm_kernel(X, R, W, x_stride_r, x_stride_c, r_stride_r, r_stride_c, N, eps, BLOCK_SIZE)
    // The kernel modifies X and R in-place
    f(stream, n_rows, 1, 1, num_warps, num_stages,
      x_flat,                        // X: input (modified in-place)
      res_flat,                      // R: residual (modified in-place)
      weight,                        // W: weight
      x_flat.stride(0),              // x_stride_r
      x_flat.stride(1),              // x_stride_c
      res_flat.stride(0),            // r_stride_r
      res_flat.stride(1),            // r_stride_c
      hidden_size,                   // N
      static_cast<float>(eps),       // eps
      BLOCK_SIZE);                   // BLOCK_SIZE (constexpr)

    return std::make_tuple(x_flat.view(orig_shape), res_flat.view(orig_shape));
}

TORCH_LIBRARY(fused_add_rms_norm_ops, m) {
    m.def("fused_add_rms_norm(Tensor input, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(fused_add_rms_norm_ops, PrivateUse1, m) {
        m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
    }
#else
    TORCH_LIBRARY_IMPL(fused_add_rms_norm_ops, CUDA, m) {
        m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
    }
#endif

}  // namespace my_ops
