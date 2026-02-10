// ==============================================================================
// apply_rotary_pos_emb_op.cpp - Multi-backend Rotary Position Embedding
// ==============================================================================

#include "apply_rotary_pos_emb_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#if defined(BACKEND_NPU)
    #if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
        #include "torch_npu/csrc/core/npu/NPUStream.h"
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif
#elif defined(BACKEND_MUSA)
    #include <musa_runtime.h>
#else
    #include "c10/cuda/CUDAStream.h"
#endif

namespace {

#if defined(BACKEND_NPU)
    using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
    using RawStream = musaStream_t;
#else
    using RawStream = CUstream;
#endif

inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& tensor) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif
#elif defined(BACKEND_MUSA)
    return nullptr;
#else
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

namespace my_ops {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rotary_dim) {

    TORCH_CHECK(q.dim() == 3, "Query must be 3D [seq_len, num_heads, head_dim]");
    TORCH_CHECK(k.dim() == 3, "Key must be 3D");

    int64_t num_tokens = q.size(0);
    int64_t num_heads_q = q.size(1);
    int64_t num_heads_k = k.size(1);
    int64_t head_dim = q.size(2);

    if (rotary_dim <= 0) {
        rotary_dim = head_dim;
    }

    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();

    // cos/sin shape: [seq_len, rotary_dim//2] -> expand to [seq_len, 1, rotary_dim//2]
    at::Tensor cos_expanded = cos.unsqueeze(1).contiguous();
    at::Tensor sin_expanded = sin.unsqueeze(1).contiguous();

#if defined(BACKEND_MUSA)
    void* q_out_ptr = nullptr;
    void* k_out_ptr = nullptr;
    size_t q_bytes = num_tokens * num_heads_q * head_dim * at::elementSize(q.scalar_type());
    size_t k_bytes = num_tokens * num_heads_k * head_dim * at::elementSize(k.scalar_type());
    musaMalloc(&q_out_ptr, q_bytes);
    musaMalloc(&k_out_ptr, k_bytes);
    auto opts = at::TensorOptions().dtype(q.scalar_type()).device(q.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor q_out = at::from_blob(q_out_ptr, {num_tokens, num_heads_q, head_dim}, deleter, opts);
    at::Tensor k_out = at::from_blob(k_out_ptr, {num_tokens, num_heads_k, head_dim}, deleter, opts);
#else
    at::Tensor q_out = at::empty_like(q);
    at::Tensor k_out = at::empty_like(k);
#endif

    // rotary_embedding_kernel(state_out, state, cos, sin,
    //     stride_state_n, stride_state_h, stride_state_d,
    //     stride_cos_n, stride_cos_d,
    //     num_tokens, num_heads,
    //     BLOCK_N, BLOCK_H, BLOCK_D)
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("apply_rotary_pos_emb.py"), "rotary_embedding_kernel");

#if defined(BACKEND_NPU)
    constexpr int64_t BLOCK_N = 4;
    constexpr int64_t BLOCK_H = 4;
    constexpr int num_warps = 1;
    constexpr int num_stages = 1;
#else
    constexpr int64_t BLOCK_N = 8;
    constexpr int64_t BLOCK_H = 4;
    constexpr int num_warps = 4;
    constexpr int num_stages = 1;
#endif

    c10::DeviceGuard guard(q.device());
    RawStream stream = get_device_stream(q);

    // Grid: (cdiv(num_tokens, BLOCK_N), cdiv(num_heads, BLOCK_H))
    auto grid_q_x = (num_tokens + BLOCK_N - 1) / BLOCK_N;
    auto grid_q_y = (num_heads_q + BLOCK_H - 1) / BLOCK_H;
    auto grid_k_y = (num_heads_k + BLOCK_H - 1) / BLOCK_H;

    // Launch for Q
    f(stream, grid_q_x, grid_q_y, 1, num_warps, num_stages,
      q_out, q_contig, cos_expanded, sin_expanded,
      q_contig.stride(0), q_contig.stride(1), q_contig.stride(2),
      cos_expanded.stride(0), cos_expanded.stride(2),
      num_tokens, num_heads_q,
      BLOCK_N, BLOCK_H, head_dim);

    // Launch for K
    f(stream, grid_q_x, grid_k_y, 1, num_warps, num_stages,
      k_out, k_contig, cos_expanded, sin_expanded,
      k_contig.stride(0), k_contig.stride(1), k_contig.stride(2),
      cos_expanded.stride(0), cos_expanded.stride(2),
      num_tokens, num_heads_k,
      BLOCK_N, BLOCK_H, head_dim);

    return std::make_tuple(q_out, k_out);
}

TORCH_LIBRARY(apply_rotary_pos_emb_ops, m) {
    m.def("apply_rotary_pos_emb(Tensor q, Tensor k, Tensor cos, Tensor sin, int rotary_dim) -> (Tensor, Tensor)");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(apply_rotary_pos_emb_ops, PrivateUse1, m) {
        m.impl("apply_rotary_pos_emb", TORCH_FN(apply_rotary_pos_emb));
    }
#else
    TORCH_LIBRARY_IMPL(apply_rotary_pos_emb_ops, CUDA, m) {
        m.impl("apply_rotary_pos_emb", TORCH_FN(apply_rotary_pos_emb));
    }
#endif

}  // namespace my_ops
