#pragma once

#include "triton_jit/backend_config.h"
#include "triton_jit/triton_jit_function_impl.h"

namespace triton_jit {

// Compile-time checks
static_assert(std::is_move_constructible_v<TritonJITFunction>,
              "TritonJITFunction must be move constructible");

static_assert(std::is_move_constructible_v<TritonKernel>,
              "TritonKernel must be move constructible");

} // namespace triton_jit
