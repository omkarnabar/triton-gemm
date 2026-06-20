"""
Benchmark Triton GEMM vs PyTorch (cuBLAS) across a range of matrix sizes.

Usage:
    python benchmarks/bench_gemm.py
"""

import torch

from kernels.gemm import matmul


def benchmark_matmul(M, N, K, iters=50, warmup=10):
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ---- Triton ----
    for _ in range(warmup):
        matmul(A, B)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / iters

    # ---- cuBLAS ----
    for _ in range(warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / iters

    flops = 2 * M * N * K
    triton_tflops = flops / (triton_ms * 1e-3) / 1e12
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

    return {
        "M": M, "N": N, "K": K,
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_tflops": triton_tflops,
        "torch_tflops": torch_tflops,
        "relative_pct": 100 * triton_tflops / torch_tflops,
    }


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    print()

    sizes = [512, 1024, 2048, 4096]

    results = []
    for size in sizes:
        torch.manual_seed(0)
        r = benchmark_matmul(size, size, size)
        results.append(r)

        print(f"M=N=K={size}")
        print(f"  Triton: {r['triton_ms']:.3f} ms | {r['triton_tflops']:.2f} TFLOPS")
        print(f"  cuBLAS: {r['torch_ms']:.3f} ms | {r['torch_tflops']:.2f} TFLOPS")
        print(f"  Relative performance: {r['relative_pct']:.1f}%")
        print()

    return results


if __name__ == "__main__":
    main()