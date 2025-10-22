#include "gemm_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include <iostream>

#if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
#include "torch_npu/csrc/core/npu/NPUStream.h"
#define HAS_TORCH_NPU 1
#else
#define HAS_TORCH_NPU 0
#endif

namespace my_ops {
using namespace triton_jit;

at::Tensor matmul(const at::Tensor &a_, const at::Tensor &b_) {
    
    const at::Tensor a = a_.contiguous();
    const at::Tensor b = b_.contiguous();

    // Get matrix dimensions
    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);
    
    // Kernel launch parameters
    const int64_t BLOCK_SIZE_M = 128;
    const int64_t BLOCK_SIZE_N = 128;
    const int64_t BLOCK_SIZE_K = 32;
    const int64_t GROUP_SIZE_M = 8;
    const int num_warps = 4;
    const int num_stages = 1;
    // Create output tensor
    at::Tensor c = at::empty({M, N}, 
                             at::TensorOptions()
                                 .dtype(torch::kFloat16)
                                 .device(a.device()));
    
    // Get Triton JIT function
    
    const TritonJITFunction &f =
        TritonJITFunction::get_instance(std::string("gemm.py"), "matmul_kernel");
    
    // Calculate 2D grid dimensions
    const unsigned int grid_m = M / BLOCK_SIZE_M;
    const unsigned int grid_n = N / BLOCK_SIZE_N;
    
    // Get stream
    aclrtStream stream = nullptr;
#if HAS_TORCH_NPU
    stream = c10_npu::getCurrentNPUStream().stream(true);
#else
    std::cout << "[GEMM DEBUG] NPU not available, using default stream" << std::endl;
#endif

    // Execute kernel with 2D grid
    c10::DeviceGuard guard(c.device());
    
    f(stream, grid_m, grid_n, 1, num_warps, num_stages,
      a, b, c,
      M, N, K,
      BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M);
    
#if HAS_TORCH_NPU
    if (stream != nullptr) {
        aclrtSynchronizeStream(stream);
    }
#else
    std::cout << "[GEMM DEBUG] No stream to synchronize (CPU)" << std::endl;
#endif
    
    return c;
}

TORCH_LIBRARY(my_ops, m) {
    m.def("matmul(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
    m.impl("matmul", TORCH_FN(matmul));
}

TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
    m.impl("matmul", TORCH_FN(matmul));
}

}  // namespace my_ops


