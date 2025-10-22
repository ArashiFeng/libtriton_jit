#include "gemm_op.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include "torch/torch.h"
#include "torch/script.h"

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include <torch_npu/torch_npu.h>
#include "triton_jit/triton_jit_function.h"

// Timing helper functions
using time_point_t = std::chrono::high_resolution_clock::time_point;

inline time_point_t now() {
  return std::chrono::high_resolution_clock::now();
}

inline double elapsed_ms(time_point_t start, time_point_t end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== Testing Triton GEMM Operation ===" << std::endl;

    // Disable auto-load of torch device backend
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

    // Get device ID from environment variable or use default
    int32_t deviceId = 1;  // 默认设备 0
    const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
        std::cout << "Using NPU device from environment variable: " << deviceId << std::endl;
    } else {
        std::cout << "NPU_DEVICE_ID not set, using default device: " << deviceId << std::endl;
    }

    // Initialize ACL and set device
    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed with error code: " << ret << std::endl;
        return -1;
    }
    std::cout << "ACL device " << deviceId << " set successfully" << std::endl;

    // Initialize NPU
    std::string npu_device_str = "npu:" + std::to_string(deviceId);
    torch_npu::init_npu(npu_device_str);
    auto device = at::Device(npu_device_str);
    std::cout << "NPU initialized: " << device << std::endl;


    // Set random seed for reproducibility
    torch::manual_seed(42);

    // Test configuration
    // M, N 是 128 的整数倍，K 是 32 的整数倍
    const int64_t M = 128;
    const int64_t K = 32;
    const int64_t N = 128;

    std::cout << "\nMatrix dimensions: A(" << M << "x" << K << ") x B("
              << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;

    // Create input tensors on CPU first, then transfer to NPU
    // This avoids the aclnnInplaceNormal error when creating random tensors directly on NPU
    std::cout << "\nCreating input tensors..." << std::endl;
    at::Tensor a = torch::ones({M, K}, torch::kFloat16).to(device);
    at::Tensor b = torch::ones({K, N}, torch::kFloat16).to(device);

    // Print sample values from input tensors
    auto a_cpu = a.to(at::kCPU).contiguous();
    auto b_cpu = b.to(at::kCPU).contiguous();
    auto a_ptr = a_cpu.data_ptr<at::Half>();
    auto b_ptr = b_cpu.data_ptr<at::Half>();

    std::cout << "A[0, 0:10]: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<float>(a_ptr[i]) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nB[0, 0:10]: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<float>(b_ptr[i]) << " ";
    }
    std::cout << std::endl;

    // Perform Triton matmul
    std::cout << "\n=== Performing Triton GEMM ===" << std::endl;

    // ========== 测量点 1: 首次运行（包含 JIT 编译和加载） ==========
    std::cout << "\n=== FIRST RUN (includes JIT compilation + loading) ===" << std::endl;
    
    at::Tensor c_triton;
    auto t_first_start = now();
    try {
        c_triton = my_ops::matmul(a, b);
        aclrtSynchronizeDevice();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    auto t_first_end = now();

    double t_first = elapsed_ms(t_first_start, t_first_end);
    std::cout << "[TIMING] First run total: " << t_first << " ms" << std::endl;
    
    // ========== 测量点 2: 第二次运行（使用缓存） ==========
    std::cout << "\n=== SECOND RUN (uses cached kernel) ===" << std::endl;

    auto t_second_start = now();    
    c_triton = my_ops::matmul(a, b);
    aclrtSynchronizeDevice();
    auto t_second_end = now();

    double t_second = elapsed_ms(t_second_start, t_second_end);
    std::cout << "[TIMING] Second run total: " << t_second << " ms" << std::endl;

    // ========== 测量点 3: 性能基准测试 ==========
    std::cout << "\n=== BENCHMARK (100 iterations) ===" << std::endl;
    
    const int num_iterations = 100;
    auto t_benchmark_start = now();
    for (int i = 0; i < num_iterations; ++i) {
        c_triton = my_ops::matmul(a, b);
    }
    aclrtSynchronizeDevice();
    auto t_benchmark_end = now();
    
    double t_total_benchmark = elapsed_ms(t_benchmark_start, t_benchmark_end);
    double t_avg = t_total_benchmark / num_iterations;
    
    std::cout << "[TIMING] Total time (" << num_iterations << " iters): " 
              << t_total_benchmark << " ms" << std::endl;
    std::cout << "[TIMING] Average kernel time: " << t_avg << " ms" << std::endl;


    // ========== 结果验证 ==========
    auto c_cpu = c_triton.to(at::kCPU).contiguous();
    auto p = c_cpu.data_ptr<at::Half>();
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "C[0, 0:5]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << static_cast<float>(p[i]) << " ";
    }
    std::cout << std::endl;

    // Cleanup ACL resources
    ret = aclrtSynchronizeDevice();
    ret = aclrtResetDevice(deviceId);
    ret = aclFinalize();
   
    std::cout << "ACL resources cleaned up successfully" << std::endl;

    return 0;
}
