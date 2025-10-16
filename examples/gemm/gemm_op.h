#pragma once

#include <iostream>
#include "torch/torch.h"

namespace my_ops {

/**
 * @brief Performs matrix multiplication C = A x B using Triton JIT kernel
 * @param a Input matrix A with shape (M, K)
 * @param b Input matrix B with shape (K, N)
 * @return Output matrix C with shape (M, N)
 */
at::Tensor matmul(const at::Tensor &a, const at::Tensor &b);

}  // namespace my_ops


