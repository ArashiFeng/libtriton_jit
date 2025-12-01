#pragma once

#include "triton_jit/backend_config.h"
#include "triton_jit/triton_jit_function_impl.h"

namespace triton_jit {

// ============================================================
// Types exported via backend_config.h and triton_jit_function_impl.h:
// - TritonJITFunction (type alias)
// - TritonKernel (type alias)
// - ArgType (enum)
// - StaticSignature (struct)
// - ParameterBuffer (struct)
// - ArgHandle (struct)
// - join_sig() (function)
// - get_next_multiple_of() (function template)
// ============================================================

// Compile-time checks
static_assert(std::is_move_constructible_v<TritonJITFunction>,
              "TritonJITFunction must be move constructible");

static_assert(std::is_move_constructible_v<TritonKernel>,
              "TritonKernel must be move constructible");

} // namespace triton_jit
