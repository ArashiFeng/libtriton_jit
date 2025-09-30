import torch
import torch_musa


torch.ops.load_library("libadd_op.so")

if __name__ == "__main__":
    x = torch.randn(128 * 1024, device="musa")
    y = torch.randn(128 * 1024, device="musa")
    result1 = torch.ops.my_ops.add_tensor(x, y)
    result2 = torch.add(x, y)

    # print(result1)
    # print(result2)
    # print(f"torch.allclose: {torch.allclose(result1, result2)}")

    torch.musa.synchronize()
    for _ in range(10):
        torch.add(x, y)
    torch.musa.synchronize()
    for _ in range(10):
        torch.ops.my_ops.add_tensor(x, y)
    torch.musa.synchronize()
