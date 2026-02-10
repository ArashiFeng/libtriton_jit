#pragma once

#include <ATen/ATen.h>

namespace my_ops {

at::Tensor mm(const at::Tensor& a, const at::Tensor& b);

}  // namespace my_ops
