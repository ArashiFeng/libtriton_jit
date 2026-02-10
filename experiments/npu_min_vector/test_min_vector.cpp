// ==============================================================================
// test_min_vector.cpp - Test NPU minimum vector length for Triton kernels
// Purpose: Find the critical n value where Triton NPU kernel starts working
// ==============================================================================

#include "add_op.h"
#include "torch/torch.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

// Backend-specific includes
#if defined(BACKEND_NPU)
    #include "acl/acl.h"
    #include "acl/acl_rt.h"
    #if __has_include("torch_npu/torch_npu.h")
        #include <torch_npu/torch_npu.h>
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif
#else
    #error "This test is only for NPU backend"
#endif

namespace {

inline void device_synchronize() {
    aclrtSynchronizeDevice();
}

int init_npu_device(at::Device& device) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

    int32_t deviceId = 0;
    const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
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

bool test_add_with_size(int64_t n, const at::Device& device) {
    // Create tensors with known values
    at::Tensor a = at::ones({n}, at::TensorOptions().dtype(at::kFloat).device(device));
    at::Tensor b = at::ones({n}, at::TensorOptions().dtype(at::kFloat).device(device)) * 2.0f;
    device_synchronize();

    // Expected: 1.0 + 2.0 = 3.0 for all elements
    at::Tensor result = my_ops::add_tensor(a, b);
    device_synchronize();

    at::Tensor expected = at::add(a, b);
    device_synchronize();

    // Compare on CPU
    at::Tensor result_cpu = result.cpu();
    at::Tensor expected_cpu = expected.cpu();

    float* result_ptr = result_cpu.data_ptr<float>();
    float* expected_ptr = expected_cpu.data_ptr<float>();

    float max_diff = 0.0f;
    int64_t num_mismatches = 0;
    for (int64_t i = 0; i < n; ++i) {
        float diff = std::abs(result_ptr[i] - expected_ptr[i]);
        if (diff > 1e-5f) {
            num_mismatches++;
        }
        max_diff = std::max(max_diff, diff);
    }

    bool passed = (num_mismatches == 0);
    return passed;
}

}  // anonymous namespace

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "  NPU Minimum Vector Length Test" << std::endl;
    std::cout << "=============================================" << std::endl;

    at::Device device(at::kCPU);
    if (init_npu_device(device) != 0) {
        return -1;
    }

    // Test different n values to find the critical point
    std::vector<int64_t> test_sizes = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    };

    std::cout << "\n=== Testing different tensor sizes ===" << std::endl;
    std::cout << "BLOCK_N = 1024 (from add.py kernel)" << std::endl;
    std::cout << std::endl;

    int64_t first_pass = -1;
    int64_t last_fail = -1;

    for (int64_t n : test_sizes) {
        bool passed = test_add_with_size(n, device);
        std::cout << "n = " << n << ": " << (passed ? "PASS" : "FAIL") << std::endl;

        if (!passed) {
            last_fail = n;
        } else if (first_pass == -1) {
            first_pass = n;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    if (last_fail > 0) {
        std::cout << "Last failing n:  " << last_fail << std::endl;
    }
    if (first_pass > 0) {
        std::cout << "First passing n: " << first_pass << std::endl;
        std::cout << "\nConclusion: NPU Triton kernel requires n >= " << first_pass << std::endl;
    } else {
        std::cout << "All tests failed!" << std::endl;
    }

    // Additional fine-grained test around the critical point
    if (last_fail > 0 && first_pass > 0 && last_fail < first_pass) {
        std::cout << "\n=== Fine-grained test between " << last_fail << " and " << first_pass << " ===" << std::endl;
        for (int64_t n = last_fail; n <= first_pass; ++n) {
            bool passed = test_add_with_size(n, device);
            if (passed) {
                std::cout << "Exact minimum n: " << n << std::endl;
                break;
            }
        }
    }

    std::cout << "\nTest completed." << std::endl;
    return 0;
}
