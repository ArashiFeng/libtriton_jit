import torch
#import torch_npu  # python直接运行该文件需引用该包
import triton
from triton import language as tl
import time

@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    
    mid = tl.program_id(0)  # M DIMENSION BLOCK INDEX
    nid = tl.program_id(1)  # N DIMENSION BLOCK INDEX
    
    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = a_rows < M
    mask_n = b_cols < N
    
    a_ptrs = a_ptr + a_rows[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    k_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K 

    for k in range(K // BLOCK_SIZE_K):  # USE tl.range TO SUPPORT RUNTIME K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        accumulator = tl.dot(a, b)
        c += accumulator
        
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    
    c = c.to(tl.float16)
    
    c_ptrs = c_ptr + a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])



def matmul(a, b):
    assert a.shape[1] == b.shape[0], f"DIMENSION INCOMPATIBLE: A={a.shape}, B={b.shape}"
    assert a.is_contiguous(), "MATRIX A MUST BE CONTIGUOUS"
    assert b.is_contiguous(), "MATRIX B MUST BE CONTIGUOUS"
    
    M, K = a.shape
    _, N = b.shape
    
    # BLOCK SIZE PARAMETERS
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # CHECK DIMENSION ALIGNMENT
    assert M % BLOCK_SIZE_M == 0, f"M ({M}) MUST BE A MULTIPLE OF BLOCK_SIZE_M ({BLOCK_SIZE_M})"
    assert N % BLOCK_SIZE_N == 0, f"N ({N}) MUST BE A MULTIPLE OF BLOCK_SIZE_N ({BLOCK_SIZE_N})"
    assert K % BLOCK_SIZE_K == 0, f"K ({K}) MUST BE A MULTIPLE OF BLOCK_SIZE_K ({BLOCK_SIZE_K})"
    
    # ALLOCATE OUTPUT TENSOR
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # USE 2D GRID TO START KERNEL
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=1,
    )
    return c


if __name__ == "__main__":
    torch.npu.set_device(7)
    torch.manual_seed(0)
    
    print("=" * 60)
    print("Triton GEMM TEST")
    
    # BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    M, N, K = 1024, 1024, 1024  # ALL ARE MULTIPLES OF 128 AND 32
    
    print(f"\nMATRIX SIZE: A({M}x{K}) × B({K}x{N}) = C({M}x{N})")
    
    device = "npu"
    print(f"USE DEVICE: {device}")
    
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    
    print("\nEXECUTE Triton GEMM...")

    # ========== 首次运行（包含 JIT 编译） ==========
    print("\n" + "=" * 60)
    print("FIRST RUN (includes JIT compilation)")
    
    t_first_total_start = time.perf_counter()
    triton_output_1 = matmul(a, b)
    t_first_total_end = time.perf_counter()
    t_first_total = (t_first_total_end - t_first_total_start) * 1000
    
    print(f"[TIMING] First run total: {t_first_total:.3f} ms")
    
    # ========== 第二次运行（使用缓存） ==========
    print("\n" + "=" * 60)
    print("SECOND RUN (uses cached kernel)")
    
    t_second_total_start = time.perf_counter()
    triton_output_2 = matmul(a, b)
    t_second_total_end = time.perf_counter()
    t_second_total = (t_second_total_end - t_second_total_start) * 1000
    
    print(f"[TIMING] Second run total: {t_second_total:.3f} ms")
    
    # ========== 多次运行取平均（性能基准） ==========
    print("\n" + "=" * 60)
    print("BENCHMARK (100 iterations)")
    
    num_iterations = 100
    torch.npu.synchronize()
    
    t_benchmark_start = time.perf_counter()
    for _ in range(num_iterations):
        _ = matmul(a, b)
    torch.npu.synchronize()
    t_benchmark_end = time.perf_counter()
    
    t_avg = ((t_benchmark_end - t_benchmark_start) * 1000) / num_iterations
    print(f"[TIMING] Average kernel time: {t_avg:.3f} ms")
    
    print("\n" + "=" * 60)
    print("✅ Triton GEMM Performance Test Completed")
    print("=" * 60)
