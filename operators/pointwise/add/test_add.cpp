// ==============================================================================
// test_add.cpp - Multi-backend Triton JIT Add Operation Test
// ==============================================================================

#include "add_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_add_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: add_basic ===" << std::endl;

    constexpr int64_t SIZE = 128 * 1024;

    at::Tensor a = tf.rand({SIZE});
    at::Tensor b = tf.rand({SIZE});
    dm.synchronize();

    at::Tensor result = my_ops::add_tensor(a, b);
    dm.synchronize();

    at::Tensor expected = at::add(a, b);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "add_basic correctness check failed");
    return 0;
}

int test_add_2d(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: add_2d ===" << std::endl;

    at::Tensor a = tf.rand({64, 128});
    at::Tensor b = tf.rand({64, 128});
    dm.synchronize();

    at::Tensor result = my_ops::add_tensor(a, b);
    dm.synchronize();

    at::Tensor expected = at::add(a, b);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "add_2d correctness check failed");
    return 0;
}

int test_add_broadcast(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: add_broadcast ===" << std::endl;

    at::Tensor a = tf.rand({32, 64});
    at::Tensor b = tf.rand({64});  // Will broadcast
    dm.synchronize();

    at::Tensor result = my_ops::add_tensor(a, b);
    dm.synchronize();

    at::Tensor expected = at::add(a, b);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "add_broadcast correctness check failed");
    return 0;
}

int test_add_shapes(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: add_shapes ===" << std::endl;

    std::vector<std::vector<int64_t>> shapes = {
        {1}, {1024}, {1024, 1024}, {32, 64, 128}
    };

    for (const auto& shape : shapes) {
        at::Tensor a = tf.rand(shape);
        at::Tensor b = tf.rand(shape);
        dm.synchronize();

        at::Tensor result = my_ops::add_tensor(a, b);
        dm.synchronize();

        at::Tensor expected = at::add(a, b);
        dm.synchronize();

        CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);

        std::cout << "Shape [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        std::cout << "]: " << (cr.passed ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(cr.passed, "add_shapes failed for a shape");
    }

    return 0;
}

int test_add_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: add ===" << std::endl;

    constexpr int64_t SIZE = 1024 * 1024 * 16;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor a = tf.rand({SIZE});
    at::Tensor b = tf.rand({SIZE});
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::add_tensor(a, b); },
        [&]() { dm.synchronize(); }
    );

    // Read a + b, write result
    int64_t bytes = SIZE * sizeof(float) * 3;
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Size: " << SIZE << " elements" << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "=======================================" << std::endl;
    std::cout << "  Triton JIT Add Operator Test Suite  " << std::endl;
    std::cout << "=======================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_add_basic(dm, tf));
    RUN_TEST(test_add_2d(dm, tf));
    RUN_TEST(test_add_broadcast(dm, tf));
    RUN_TEST(test_add_shapes(dm, tf));
    RUN_TEST(test_add_benchmark(dm, tf));

    std::cout << "\n=======================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "=======================================" << std::endl;

    return 0;
}
