#pragma once

#if defined(BACKEND_IX)
    #include "triton_jit/backends/ix_backend.h"
#else
    #include "triton_jit/backends/cuda_backend.h"
#endif

namespace triton_jit {

// Forward declarations of template classes
template<BackendPolicy Backend>
class TritonKernelImpl;

template<BackendPolicy Backend>
class TritonJITFunctionImpl;


#if defined(BACKEND_CUDA)
    /// Default backend for CUDA
    using DefaultBackend = CudaBackend;

#elif defined(BACKEND_IX)
    /// Default backend for IX (Tianshu)
    using DefaultBackend = IxBackend;

#elif defined(BACKEND_NPU)
    /// Default backend for NPU (Ascend)
    // Future implementation
    using DefaultBackend = NpuBackend;
    #error "NPU Backend not yet implemented. Use BACKEND_CUDA for now."

#else
    // Default to CUDA if no backend specified
    #warning "No backend specified, defaulting to CUDA. Use -DBACKEND=CUDA explicitly."
    using DefaultBackend = CudaBackend;

#endif


using TritonKernel = TritonKernelImpl<DefaultBackend>;
using TritonJITFunction = TritonJITFunctionImpl<DefaultBackend>;
using DefaultStreamType = DefaultBackend::StreamType;
using DefaultContextType = DefaultBackend::ContextType;
using DefaultKernelHandle = DefaultBackend::KernelHandle;

} // namespace triton_jit
