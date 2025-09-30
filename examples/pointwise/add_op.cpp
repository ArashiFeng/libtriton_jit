

#include "add_op.h"
#include "torch_musa/csrc/core/MUSAStream.h"
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

  // getCurrentMUSAStream ensures that the stream is initialized, a default stream for each device
  c10::musa::MUSAStream stream = c10::musa::getCurrentMUSAStream();
  c10::DeviceGuard guard(out.device());
  MUstream raw_stream = static_cast<MUstream>(stream.stream());
  f(raw_stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
  return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

// TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
  m.impl("add_tensor", TORCH_FN(add_tensor));
}
}  // namespace my_ops
