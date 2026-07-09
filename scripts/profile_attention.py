import torch

from kernels.attention import attention


def main():
    assert torch.cuda.is_available()

    print("GPU:", torch.cuda.get_device_name(0))

    # Typical configuration for modern LLM layer workloads (e.g. Llama-3 style execution)
    B = 2         # Batch Size
    H = 8         # Number of Heads
    N_CTX = 4096  # Sequence Length Context
    D_HEAD = 64   # Head Dimension

    q = torch.randn(
        (B, H, N_CTX, D_HEAD),
        device="cuda",
        dtype=torch.float16,
    )

    k = torch.randn(
        (B, H, N_CTX, D_HEAD),
        device="cuda",
        dtype=torch.float16,
    )

    v = torch.randn(
        (B, H, N_CTX, D_HEAD),
        device="cuda",
        dtype=torch.float16,
    )

    # Warmup (Triton autotuning happens here across the Config block parameters)
    for _ in range(10):
        attention(q, k, v)

    torch.cuda.synchronize()

    # Profile this kernel execution using Nsight Compute / Nsight Systems
    attention(q, k, v)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()