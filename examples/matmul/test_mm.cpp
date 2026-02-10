// ==============================================================================
// test_mm.cpp - Multi-backend Triton JIT Matrix Multiplication Test
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "mm_op.h"
#include "torch/torch.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

// ==============================================================================
//                         BACKEND DETECTION & HEADERS
// ==============================================================================

#if defined(BACKEND_NPU)
    // ----------------------------- NPU Backend -----------------------------
    #include "acl/acl.h"
    #include "acl/acl_rt.h"
    #if __has_include("torch_npu/torch_npu.h")
        #include <torch_npu/torch_npu.h>
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif

#elif defined(BACKEND_MUSA)
    // ----------------------------- MUSA Backend ----------------------------
    #include "musa.h"
    #include "musa_runtime.h"
    #include "pybind11/embed.h"

#else
    // ----------------------- CUDA / IX Backend (Default) -------------------
    #include "c10/cuda/CUDAFunctions.h"

#endif

// ==============================================================================
//                         BACKEND-SPECIFIC UTILITIES
// ==============================================================================

namespace {

// ----------------------------- Device Synchronize ----------------------------

inline void device_synchronize() {
#if defined(BACKEND_NPU)
    aclrtSynchronizeDevice();
#elif defined(BACKEND_MUSA)
    musaDeviceSynchronize();
#else
    c10::cuda::device_synchronize();
#endif
}

// ----------------------------- Device Initialization -------------------------

#if defined(BACKEND_NPU)

int init_npu_device(at::Device& device) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

    int32_t deviceId = 1;
    const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
        std::cout << "Using NPU device from env: " << deviceId << std::endl;
    } else {
        std::cout << "NPU_DEVICE_ID not set, using default: " << deviceId << std::endl;
    }

    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed: " << ret << std::endl;
        return -1;
    }

    #if HAS_TORCH_NPU
        std::string npu_device_str = "npu:" + std::to_string(deviceId);
        torch_npu::init_npu(npu_device_str);
        device = at::Device(npu_device_str);
        std::cout << "NPU initialized: " << device << std::endl;
        return 0;
    #else
        std::cerr << "torch_npu not available" << std::endl;
        return -1;
    #endif
}

void finalize_npu_device() {
    #if HAS_TORCH_NPU
        int32_t deviceId = 0;
        const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
        if (deviceEnv != nullptr) {
            deviceId = std::atoi(deviceEnv);
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    #endif
}

#elif defined(BACKEND_MUSA)

int init_musa_device(at::Device& device) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

    int32_t deviceId = 0;
    const char* deviceEnv = std::getenv("MUSA_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
        std::cout << "Using MUSA device from env: " << deviceId << std::endl;
    } else {
        std::cout << "MUSA_DEVICE_ID not set, using default: " << deviceId << std::endl;
    }

    musaError_t ret = musaSetDevice(deviceId);
    if (ret != musaSuccess) {
        std::cerr << "musaSetDevice failed: " << musaGetErrorString(ret) << std::endl;
        return -1;
    }

    namespace py = pybind11;
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    try {
        py::gil_scoped_acquire gil;
        py::module_::import("torch_musa");
    } catch (const py::error_already_set& e) {
        std::cerr << "Failed to import torch_musa: " << e.what() << std::endl;
        return -1;
    }

    device = at::Device(at::DeviceType::PrivateUse1, deviceId);
    std::cout << "MUSA device initialized: " << device << std::endl;

    return 0;
}

void finalize_musa_device() {
    musaDeviceReset();
}

#else  // CUDA / IX

int init_cuda_device(at::Device& device) {
    device = at::Device(at::kCUDA);
    std::cout << "CUDA device initialized" << std::endl;
    return 0;
}

void finalize_cuda_device() {
    // CUDA cleanup is automatic
}

#endif

// ----------------------------- Tensor Creation -------------------------------

inline at::Tensor create_random_matrix(int64_t rows, int64_t cols, const at::Device& device) {
#if defined(BACKEND_NPU)
    // NPU: Use float16 for better kernel compatibility (matching Python test)
    return at::rand({rows, cols}, at::TensorOptions().dtype(at::kHalf).device(device));

#elif defined(BACKEND_MUSA)
    std::vector<float> h_data(rows * cols);
    for (int64_t i = 0; i < rows * cols; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    void* d_ptr = nullptr;
    musaError_t err = musaMalloc(&d_ptr, rows * cols * sizeof(float));
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }

    err = musaMemcpy(d_ptr, h_data.data(), rows * cols * sizeof(float), musaMemcpyHostToDevice);
    if (err != musaSuccess) {
        musaFree(d_ptr);
        throw std::runtime_error("musaMemcpy failed: " + std::string(musaGetErrorString(err)));
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(device);
    auto deleter = [](void* ptr) { musaFree(ptr); };
    return at::from_blob(d_ptr, {rows, cols}, deleter, options);

#else
    return at::rand({rows, cols}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
#endif
}

}  // anonymous namespace

// ==============================================================================
//                                   MAIN
// ==============================================================================

int main() {
    constexpr int64_t M = 512;
    constexpr int64_t K = 256;
    constexpr int64_t N = 512;
    constexpr int WARMUP_ITERS = 5;
    constexpr int BENCH_ITERS = 10;

    std::cout << "=======================================" << std::endl;
    std::cout << "  Triton JIT MM Example Test          " << std::endl;
    std::cout << "=======================================" << std::endl;

    // ======================== Device Initialization ==========================
    at::Device device(at::kCPU);

#if defined(BACKEND_NPU)
    std::cout << "Backend: NPU" << std::endl;
    if (init_npu_device(device) != 0) return -1;
#elif defined(BACKEND_MUSA)
    std::cout << "Backend: MUSA" << std::endl;
    if (init_musa_device(device) != 0) return -1;
#else
    std::cout << "Backend: CUDA" << std::endl;
    if (init_cuda_device(device) != 0) return -1;
#endif

    // ======================== Create Test Matrices ===========================
    at::Tensor a = create_random_matrix(M, K, device);
    at::Tensor b = create_random_matrix(K, N, device);

    std::cout << "\n=== Input Matrix Info ===" << std::endl;
    std::cout << "A: (" << M << ", " << K << ")" << std::endl;
    std::cout << "B: (" << K << ", " << N << ")" << std::endl;
    std::cout << "Device: " << a.device() << std::endl;

    // ======================== Compute ========================================
    std::cout << "\n=== Executing Matrix Multiplication ===" << std::endl;

    at::Tensor result = my_ops::mm(a, b);
    device_synchronize();

    // ======================== Result Verification ============================
    std::cout << "\n=== Results ===" << std::endl;

#if defined(BACKEND_MUSA)
    // MUSA: Manual verification with CPU computation
    std::vector<float> a_cpu(M * K), b_cpu(K * N), result_cpu(M * N);
    musaMemcpy(a_cpu.data(), a.data_ptr<float>(), M * K * sizeof(float), musaMemcpyDeviceToHost);
    musaMemcpy(b_cpu.data(), b.data_ptr<float>(), K * N * sizeof(float), musaMemcpyDeviceToHost);
    musaMemcpy(result_cpu.data(), result.data_ptr<float>(), M * N * sizeof(float), musaMemcpyDeviceToHost);

    // Compute reference on CPU
    std::vector<float> expected_cpu(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_cpu[i * K + k] * b_cpu[k * N + j];
            }
            expected_cpu[i * N + j] = sum;
        }
    }

    // Check results
    bool is_close = true;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        float diff = std::abs(result_cpu[i] - expected_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3f * std::abs(expected_cpu[i]) + 1e-3f) {
            is_close = false;
        }
    }

    std::cout << "Result[0,0]: " << result_cpu[0] << std::endl;
    std::cout << "Expected[0,0]: " << expected_cpu[0] << std::endl;
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;

#elif defined(BACKEND_NPU)
    // NPU: Use CPU computation as reference
    at::Tensor a_cpu = a.cpu();
    at::Tensor b_cpu = b.cpu();
    at::Tensor result_cpu = result.cpu();
    at::Tensor expected = at::mm(a_cpu, b_cpu);

    std::cout << "Result[0,0]: " << result_cpu[0][0].item<float>() << std::endl;
    std::cout << "Expected[0,0]: " << expected[0][0].item<float>() << std::endl;

    bool is_close = at::allclose(result_cpu, expected, 1e-3, 1e-3);
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    if (!is_close) {
        auto diff = at::abs(result_cpu - expected);
        std::cout << "Max difference: " << at::max(diff).item<float>() << std::endl;
    }

#else
    // CUDA/IX: Use at::mm as reference
    at::Tensor expected = at::mm(a, b);
    at::Tensor result_cpu = result.cpu();
    at::Tensor expected_cpu = expected.cpu();

    std::cout << "Result[0,0]: " << result_cpu[0][0].item<float>() << std::endl;
    std::cout << "Expected[0,0]: " << expected_cpu[0][0].item<float>() << std::endl;

    bool is_close = at::allclose(result, expected, 1e-3, 1e-3);
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    if (!is_close) {
        auto diff = at::abs(result_cpu - expected_cpu);
        std::cout << "Max difference: " << at::max(diff).item<float>() << std::endl;
    }
#endif

    // ======================== Performance Benchmark ==========================
    std::cout << "\n=== Performance Benchmark ===" << std::endl;

    // Warm-up
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        auto tmp = my_ops::mm(a, b);
    }
    device_synchronize();

    // Benchmark
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto tmp = my_ops::mm(a, b);
    }
    device_synchronize();
    std::cout << "my_ops::mm benchmark completed (" << BENCH_ITERS << " iters)" << std::endl;

    // ======================== Cleanup ========================================
#if defined(BACKEND_NPU)
    finalize_npu_device();
#elif defined(BACKEND_MUSA)
    finalize_musa_device();
#else
    finalize_cuda_device();
#endif

    std::cout << "\nProgram completed successfully!" << std::endl;
    return is_close ? 0 : 1;
}
