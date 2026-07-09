"""
Benchmark Triton GEMM vs PyTorch CUDA GEMM (cuBLAS/cuBLASLt backend)
across a range of matrix sizes.

Usage:
    python benchmarks/bench_gemm.py
"""

import os
import csv
import torch

from kernels.gemm import matmul


def check_correctness(M, N, K):
    A = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16
    )

    B = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16
    )

    C_triton = matmul(A, B)
    C_torch = torch.matmul(A, B)

    torch.testing.assert_close(
        C_triton,
        C_torch,
        rtol=1e-1,
        atol=1e-1
    )


def benchmark_kernel(kernel, A, B, iters=50, warmup=10):
    for _ in range(warmup):
        kernel(A, B)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        kernel(A, B)

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def benchmark_matmul(M, N, K, iters=50, warmup=10):
    A = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16
    )

    B = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16
    )

    triton_ms = benchmark_kernel(
        matmul,
        A,
        B,
        iters,
        warmup
    )

    torch_ms = benchmark_kernel(
        torch.matmul,
        A,
        B,
        iters,
        warmup
    )

    flops = 2 * M * N * K

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
        "M": M,
        "N": N,
        "K": K,
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_tflops": triton_tflops,
        "torch_tflops": torch_tflops,
        "relative_pct": 100 * triton_tflops / torch_tflops,
    }


def save_results(results, path="results/gemm_benchmark.csv"):
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
    check_correctness(512, 512, 512)
    print("Correctness passed\n")

    sizes = [
        512,
        1024,
        2048,
        4096,
    ]

    results = []

    for size in sizes:
        torch.manual_seed(0)

        result = benchmark_matmul(
            size,
            size,
            size
        )

        results.append(result)

        print(f"M=N=K={size}")

        print(
            f"  Triton: "
            f"{result['triton_ms']:.3f} ms | "
            f"{result['triton_tflops']:.2f} TFLOPS"
        )

        print(
            f"  cuBLAS: "
            f"{result['torch_ms']:.3f} ms | "
            f"{result['torch_tflops']:.2f} TFLOPS"
        )

        print(
            f"  Relative performance: "
            f"{result['relative_pct']:.1f}%"
        )

        print()

    save_results(results)

    print("Saved results to results/gemm_benchmark.csv")


if __name__ == "__main__":
    main()