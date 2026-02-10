import torch
import triton
from triton import language as tl


@triton.jit
def mm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute C = A @ B where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def mm(a: torch.Tensor, b: torch.Tensor, block_m=64, block_n=64, block_k=32, group_m=8) -> torch.Tensor:
    """Matrix multiplication using Triton kernel."""
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = block_m
    BLOCK_N = block_n
    BLOCK_K = block_k
    GROUP_M = group_m

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    return c


def test_mm(dtype, dtype_name, block_m=64, block_n=64, block_k=32, group_m=8,
            matrix_m=128, matrix_k=64, matrix_n=128):
    """Test mm kernel with specified dtype."""
    print(f"\n{'='*50}")
    print(f"Testing with {dtype_name}, BLOCK=({block_m},{block_n},{block_k}), GROUP_M={group_m}")
    print('='*50)

    M, N, K = matrix_m, matrix_n, matrix_k
    a = torch.randn((M, K), device='npu', dtype=dtype)
    b = torch.randn((K, N), device='npu', dtype=dtype)

    print(f"Input shape: A({M}, {K}), B({K}, {N})")
    print(f"Input dtype: {a.dtype}")

    try:
        # Triton result
        c_triton = mm(a, b, block_m=block_m, block_n=block_n, block_k=block_k, group_m=group_m)

        # PyTorch reference
        c_ref = torch.mm(a, b)

        # Compare results
        diff = torch.abs(c_triton - c_ref).max().item()
        print(f"Max difference: {diff}")

        if diff < 1e-2:
            print(f"[PASS] {dtype_name} test passed!")
        else:
            print(f"[FAIL] {dtype_name} results differ significantly")

        print(f"Triton result:\n{c_triton[:4, :4]}")
        print(f"Reference result:\n{c_ref[:4, :4]}")
        return True
    except Exception as e:
        print(f"[ERROR] {dtype_name} test failed with exception:")
        print(f"  {e}")
        return False


if __name__ == "__main__":
    import torch_npu

    print("Testing Triton Ascend with mm kernel...")

    # Test float16 with default block sizes (64x64)
    test_mm(torch.float16, "float16")

    # Test float32 with default block sizes (64x64)
    test_mm(torch.float32, "float32")

    # Test float32 with C++ NPU block sizes (32x32) - same as C++ test
    print("\n" + "="*60)
    print("Testing with C++ NPU configuration (BLOCK=32x32, GROUP_M=4)")
    print("="*60)
    test_mm(torch.float32, "float32 (C++ config)", block_m=32, block_n=32, block_k=32, group_m=4)

    # Test with C++ test matrix sizes (512x256 @ 256x512)
    print("\n" + "="*60)
    print("Testing with C++ matrix sizes (512x256 @ 256x512)")
    print("="*60)
    test_mm(torch.float32, "float32 (C++ sizes)", block_m=32, block_n=32, block_k=32, group_m=4,
            matrix_m=512, matrix_k=256, matrix_n=512)
