

#include "axpy_op.h"
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor axpy(const at::Tensor &x, const at::Tensor &y, const c10::Scalar &alpha) {
  auto res = torch::broadcast_tensors({x, y});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &xx = res[0];
  const at::Tensor &yy = res[1];

  // TODO: consider weak-type of alpha here
  at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.scalar_type());
  at::Tensor out = at::empty(xx.sizes(), at::TensorOptions().dtype(out_dtype).device(x.device()));

  const TritonJITFunction &f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream, num_blocks, 1, 1, num_warps, num_stages, x, y, out, alpha, n, tile_size);
  return out;
}

at::Tensor axpy2(const at::Tensor &x, const at::Tensor &y, const std::optional<c10::Scalar> &alpha) {
  auto res = torch::broadcast_tensors({x, y});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &xx = res[0];
  const at::Tensor &yy = res[1];

  // TODO: consider weak-type of alpha here
  at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.scalar_type());
  at::Tensor out = at::empty(xx.sizes(), at::TensorOptions().dtype(out_dtype).device(x.device()));

  const TritonJITFunction &f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy2_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream, num_blocks, 1, 1, num_warps, num_stages, x, y, out, alpha, n, tile_size);
  return out;
}

at::Tensor axpy3(const at::Tensor &x,
                 const std::optional<at::Tensor> &y,
                 const std::optional<c10::Scalar> &alpha) {
  at::Tensor out = [&]() {
    if (!y.has_value()) {
      return at::empty_like(x);
    } else {
      auto res = torch::broadcast_tensors({x, y.value()});
      res[0] = res[0].contiguous();
      res[1] = res[1].contiguous();
      const at::Tensor &xx = res[0];
      const at::Tensor &yy = res[1];

      // TODO: consider weak-type of alpha here
      at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.value().scalar_type());
      return at::empty(xx.sizes(), at::TensorOptions().dtype(out_dtype).device(x.device()));
    }
  }();
  const TritonJITFunction &f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy3_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream, num_blocks, 1, 1, num_warps, num_stages, x, y, out, alpha, n, tile_size);
  return out;
}

at::Tensor axpy3_manual(const at::Tensor &x,
                        const std::optional<at::Tensor> &y,
                        const std::optional<c10::Scalar> &alpha) {
  at::Tensor out = [&]() {
    if (!y.has_value()) {
      return at::empty_like(x);
    } else {
      auto res = torch::broadcast_tensors({x, y.value()});
      res[0] = res[0].contiguous();
      res[1] = res[1].contiguous();
      const at::Tensor &xx = res[0];
      const at::Tensor &yy = res[1];

      // TODO: consider weak-type of alpha here
      at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.value().scalar_type());
      return at::empty(xx.sizes(), at::TensorOptions().dtype(out_dtype).device(x.device()));
    }
  }();
  const TritonJITFunction &f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy3_kernel");

  ParameterBuffer buffer;
  const int num_args = 3;
  buffer.reserve(num_args);
  c10::SmallVector<std::string> signature;
  signature.reserve(num_args);
  ArgHandle handler = {f.get_static_sig(), buffer, signature, 0};

  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();

  // add each arg manually
  handler.handle_arg(x);
  handler.handle_arg(y);
  handler.handle_arg(out);
  handler.handle_arg(alpha);
  handler.handle_arg(n);
  handler.handle_arg(tile_size);

  void *global_scratch = nullptr;
  // data_pointers.push_back(global_scratch);
  // kernel_args.push_back(&(data_pointers.back()));
  buffer.push_arg(global_scratch);
  std::string full_signature;
  for (int i = 0; i < signature.size(); i++) {
    if (i == 0) {
      full_signature += signature[i];
    } else {
      full_signature += ",";
      full_signature += signature[i];
    }
  }

  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;
  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  // TODO: use torch backend-agnostic device APIs
  ensure_cuda_context();
  CUdevice device_index;
  checkCudaErrors(cuCtxGetDevice(&device_index));
  const TritonKernel &kernel = f.get_kernel(full_signature, num_warps, num_stages, device_index);
  kernel.launch(num_blocks, 1, 1, num_warps, stream, buffer.get_ptrs());
  return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("axpy(Tensor self, Tensor other, Scalar alpha) -> Tensor");
  m.def("axpy2(Tensor self, Tensor other, Scalar? alpha) -> Tensor");
  m.def("axpy3(Tensor self, Tensor? other, Scalar? alpha) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("axpy", TORCH_FN(axpy));
  m.impl("axpy2", TORCH_FN(axpy2));
  m.impl("axpy3", TORCH_FN(axpy3));
}
}  // namespace my_ops
