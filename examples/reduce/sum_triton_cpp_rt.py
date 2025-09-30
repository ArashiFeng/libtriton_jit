import torch
import torch_musa

# in this way, we can skip building a python extension
torch.ops.load_library("libsum_op.so")

if __name__ == "__main__":
    x = torch.randn(16, 4 * 1024, device="musa")
    result1 = torch.ops.my_ops.sum.dim_IntList(x, [1])
    result2 = torch.sum(x, [1])

    # print(result1)
    # print(result2)
    # print(f"torch.allclose: {torch.allclose(result1, result2)}")

    torch.musa.synchronize()
    for _ in range(10):
        torch.sum(x, [1])
    torch.musa.synchronize()
    for _ in range(10):
        torch.ops.my_ops.sum.dim_IntList(x, [1])
    torch.musa.synchronize()
