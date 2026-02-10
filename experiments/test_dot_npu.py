import torch
import triton
from triton import language as tl
import torch_npu

@triton.jit
def simple_dot_kernel(
    A, B, C,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def test_dot():
    device = torch.device('npu:0')
    torch_npu.npu.set_device(device)

    M, K, N = 64, 64, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    grid = (M // BLOCK_M, N // BLOCK_N)

    print(f"Running simple dot kernel on NPU...")
    print(f"M={M}, K={K}, N={N}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"Grid: {grid}")

    try:
        simple_dot_kernel[grid](
            a, b, c,
            M, K, N,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=1
        )
        torch_npu.npu.synchronize()

        expected = torch.mm(a, b)
        torch_npu.npu.synchronize()

        print(f"Result shape: {c.shape}")
        print(f"Expected shape: {expected.shape}")
        print(f"Max diff: {(c - expected).abs().max().item()}")
        print(f"Match: {torch.allclose(c, expected, rtol=1e-2, atol=1e-2)}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dot()
