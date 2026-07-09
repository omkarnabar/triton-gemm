import torch

from kernels.gemm import matmul


def main():
    assert torch.cuda.is_available()

    print("GPU:", torch.cuda.get_device_name(0))

    M = N = K = 4096

    A = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )

    B = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )

    # Warmup (Triton autotuning happens here)
    for _ in range(10):
        matmul(A, B)

    torch.cuda.synchronize()

    # Profile this kernel execution
    matmul(A, B)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()