
#include "add_op.h"
#include "generated_cuda_compatible/c10/musa/MUSA_PORT_Functions.h"
#include "torch/torch.h"

int main() {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);
  // warm up
  at::Tensor result1 = my_ops::add_tensor(a, b);
  at::Tensor result2 = at::add(a, b);

  c10::musa::device_synchronize();
  for (int i = 0; i < 10; ++i) {
    auto tmp = at::add(a, b);
  }
  c10::musa::device_synchronize();
  for (int i = 0; i < 10; ++i) {
    auto tmp = my_ops::add_tensor(a, b);
  }
  c10::musa::device_synchronize();
  return 0;
}
