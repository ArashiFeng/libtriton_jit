#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "cuda.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {

class TritonJITFunction;

template <typename T>
T get_next_multiple_of(T pos, T step) {
  return ((pos + step - 1) / step) * step;
}

struct ParameterBuffer {
  c10::SmallVector<std::byte> buff_;
  size_t cursor_ = 0;
  c10::SmallVector<size_t> offsets_;

  void reserve(size_t new_cap) {
    const int ESTIMATED_BYTES_PER_ARG = 4;
    this->buff_.reserve(new_cap * ESTIMATED_BYTES_PER_ARG);
    this->offsets_.reserve(new_cap);
  }

  template <typename T>
  void push_arg(T &&v) {
    using U = std::decay_t<T>;
    static_assert(std::is_trivially_copyable_v<U>, "Non trivially copyable type");
    size_t align = alignof(U);
    size_t offset = get_next_multiple_of(this->cursor_, align);
    this->offsets_.push_back(offset);

    size_t size = sizeof(U);
    this->buff_.resize(offset + size);
    std::byte *ptr = this->buff_.data() + offset;
    std::memcpy(ptr, &v, size);

    this->cursor_ = offset + size;
  }

  c10::SmallVector<void *> get_ptrs() {
    c10::SmallVector<void *> ptrs;
    ptrs.reserve(this->offsets_.size());
    std::byte *start = this->buff_.data();
    for (const size_t off : this->offsets_) {
      ptrs.push_back(start + off);
    }
    return ptrs;
  }

  size_t size() const {
    return this->offsets_.size();
  }
};

class TritonKernel {
 private:
  // * The directory that contain the IRs(ttir, ttgir, llir, ptx, cubin) & metadata(json file))*/
  std::string dir_;
  /* name of the kernel in cubin */
  std::string kernel_name_;
  unsigned int shared_; /* amount of static shared memory per block (in bytes) required for the cubin*/
  unsigned int arch_;   /* cuda arch */

  mutable CUmodule mod_;
  mutable CUfunction fn_;
  mutable bool loaded_ = false;

 public:
  TritonKernel(const TritonKernel &) = delete;
  TritonKernel &operator=(const TritonKernel &) = delete;
  TritonKernel(TritonKernel &&) = default;
  TritonKernel &operator=(TritonKernel &&) = default;
  TritonKernel() = default;

  void launch(unsigned int grid_x,
              unsigned int grid_y,
              unsigned int grid_z,
              int num_warps,
              CUstream stream,
              void **args) const;
  friend TritonJITFunction;

 private:
  TritonKernel(std::string_view dir, std::string_view kernel_name);
  /* load cubin into a cumodule for a device */
  void lazy_init_handle() const;
};
static_assert(std::is_move_constructible_v<TritonKernel>);
}  // namespace triton_jit
