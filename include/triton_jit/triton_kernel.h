#pragma once

#include "triton_jit/backend_config.h"
#include "triton_jit/triton_kernel_impl.h"

namespace triton_jit {

// For backward compatibility, ensure TritonKernel is move constructible
static_assert(std::is_move_constructible_v<TritonKernel>,
              "TritonKernel must be move constructible");

} // namespace triton_jit
