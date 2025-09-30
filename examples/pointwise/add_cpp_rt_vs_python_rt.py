import torch
import triton
import torch_musa
from triton import language as tl

torch.ops.load_library("libadd_op.so")

@triton.jit
def binary_pointwise_kernel(X, Y, Out, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    o = x + y
    tl.store(Out + offsets, o, mask=mask)


def binary_add_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # lets be simple and assume x and y are the same shape,
    # all-contiguous, the same dtype
    out = torch.empty_like(x, dtype=x.dtype)
    n = out.numel()
    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N), 1, 1)
    with torch.musa.device(x.device):
        binary_pointwise_kernel[grid](
            x, y, out, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1
        )
    return out
    

if __name__ == "__main__":
    x = torch.randn(128 * 1024, device="musa")
    y = torch.randn(128 * 1024, device="musa")
    result1 = torch.ops.my_ops.add_tensor(x, y)
    result2 = binary_add_tensor(x, y)

    # print(result1)
    # print(result2)
    # print(f"torch.allclose: {torch.allclose(result1, result2)}")

    torch.musa.synchronize()
    for _ in range(10):
        binary_add_tensor(x, y)
    torch.musa.synchronize()
    for _ in range(10):
        torch.ops.my_ops.add_tensor(x, y)
    torch.musa.synchronize()
