

#include "add_op.h"
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string("add.py"), "binary_pointwise_kernel");

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
  f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
  return out;
}

at::Tensor add_tensor_manual_arg_handle(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string("add.py"), "binary_pointwise_kernel");

  ParameterBuffer buffer;
  const int num_args = 4;  // just a estimation
  buffer.reserve(num_args);
  c10::SmallVector<std::string> signature;
  signature.reserve(num_args);
  ArgHandle handler = {f.get_static_sig(), buffer, signature, 0};

  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();

  // add each arg manually
  handler.handle_arg(a);
  handler.handle_arg(b);
  handler.handle_arg(out);
  handler.handle_arg(n);
  handler.handle_arg(tile_size);
  handler.append_global_scratch();

  std::string full_signature = join_sig(signature);

  ensure_cuda_context();
  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  CUdevice device_index;
  checkCudaErrors(cuCtxGetDevice(&device_index));

  const TritonKernel &kernel = f.get_kernel(full_signature, num_warps, num_stages, device_index);
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;
  c10::SmallVector<void *> ptrs = buffer.get_ptrs();
  kernel.launch(num_blocks, 1, 1, num_warps, stream, ptrs.data());
  return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
  m.def("add_tensor_manual_arg_handle(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("add_tensor", TORCH_FN(add_tensor));
  m.impl("add_tensor_manual_arg_handle", TORCH_FN(add_tensor_manual_arg_handle));
}
}  // namespace my_ops
