"""
Benchmark Triton Causal FlashAttention vs PyTorch Eager Attention (cuBLAS/cuDNN-like math)
across a range of sequence context sizes.

Usage:
    python benchmarks/bench_attention.py
"""

import os
import csv
import torch

# Assuming your attention function is exposed from kernels.attention
from kernels.attention import attention


def check_correctness(B, H, N_CTX, D_HEAD):
    q = torch.randn((B, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16)

    o_triton = attention(q, k, v)

    # Reference implementation using standard PyTorch eager execution with a causal mask
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (D_HEAD ** 0.5))
    mask = torch.triu(torch.full((N_CTX, N_CTX), float('-inf'), device="cuda"), diagonal=1)
    scores = scores + mask[None, None, :, :]
    p = torch.softmax(scores, dim=-1)
    o_ref = torch.matmul(p, v)

    torch.testing.assert_close(
        o_triton,
        o_ref,
        rtol=1e-1,
        atol=1e-1
    )


def benchmark_kernel(kernel, q, k, v, iters=50, warmup=10):
    for _ in range(warmup):
        kernel(q, k, v)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        kernel(q, k, v)

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def benchmark_torch_attention(q, k, v):
    """
    Standard PyTorch Eager causal attention baseline.
    """
    d_head = q.shape[-1]
    n_ctx = q.shape[-2]
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (d_head ** 0.5))
    mask = torch.triu(torch.full((n_ctx, n_ctx), float('-inf'), device="cuda"), diagonal=1)
    scores = scores + mask[None, None, :, :]
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def benchmark_attention_perf(B, H, N_CTX, D_HEAD, iters=50, warmup=10):
    q = torch.randn((B, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N_CTX, D_HEAD), device="cuda", dtype=torch.float16)

    triton_ms = benchmark_kernel(
        attention,
        q, k, v,
        iters,
        warmup
    )

    torch_ms = benchmark_kernel(
        benchmark_torch_attention,
        q, k, v,
        iters,
        warmup
    )

    # FLOP Calculation for Causal FlashAttention Forward:
    # 1. Q @ K^T requires 2 * B * H * N_CTX * N_CTX * D_HEAD operations.
    # 2. P @ V requires 2 * B * H * N_CTX * N_CTX * D_HEAD operations.
    # Total operations for full attention = 4 * B * H * N_CTX^2 * D_HEAD.
    # Since it is CAUSAL, we only process half the matrix (the lower triangle), 
    # cutting total FLOP count exactly in half: 2 * B * H * N_CTX^2 * D_HEAD.
    flops = 2 * B * H * N_CTX * N_CTX * D_HEAD

    triton_tflops = (
        flops /
        (triton_ms * 1e-3) /
        1e12
    )

    torch_tflops = (
        flops /
        (torch_ms * 1e-3) /
        1e12
    )

    return {
        "B": B,
        "H": H,
        "N_CTX": N_CTX,
        "D_HEAD": D_HEAD,
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_tflops": triton_tflops,
        "torch_tflops": torch_tflops,
        "relative_pct": 100 * triton_tflops / torch_tflops,
    }


def save_results(results, path="results/attention_benchmark.csv"):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"

    print("GPU:", torch.cuda.get_device_name(0))
    print()

    print("Running correctness check...")
    check_correctness(B=2, H=4, N_CTX=512, D_HEAD=64)
    print("Correctness passed\n")

    # Fixed Batch Size, Num Heads, Head Dimension typical for standard models
    B = 2
    H = 8
    D_HEAD = 64

    # Scaling sequence context length
    context_sizes = [
        512,
        1024,
        2048,
        4096,
    ]

    results = []

    for size in context_sizes:
        torch.manual_seed(0)

        result = benchmark_attention_perf(
            B, H, size, D_HEAD
        )

        results.append(result)

        print(f"B={B}, H={H}, N_CTX={size}, D_HEAD={D_HEAD}")

        print(
            f"  Triton FlashAttention: "
            f"{result['triton_ms']:.3f} ms | "
            f"{result['triton_tflops']:.2f} TFLOPS"
        )

        print(
            f"  PyTorch Eager:         "
            f"{result['torch_ms']:.3f} ms | "
            f"{result['torch_tflops']:.2f} TFLOPS"
        )

        print(
            f"  Relative performance: "
            f"{result['relative_pct']:.1f}%"
        )

        print()

    save_results(results)

    print("Saved results to results/attention_benchmark.csv")


if __name__ == "__main__":
    main()