"""
Profile the Triton GEMM kernel with Nsight Systems.

Run from repo root:
    nsys profile -o profiling/out/gemm_profile --force-overwrite=true \
        python profiling/profile_gemm.py

Then open the .nsys-rep file in the Nsight Systems GUI, or summarize on
the command line:
    nsys stats profiling/out/gemm_profile.nsys-rep

For kernel-level metrics (occupancy, memory throughput, warp stalls),
use Nsight Compute instead:
    ncu -o profiling/out/gemm_ncu --set full \
        python profiling/profile_gemm.py
"""

import torch

from kernels.gemm import matmul


def main():
    torch.manual_seed(0)

    M = N = K = 4096

    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)

    # Warmup — lets the autotuner pick a config and JIT-compile
    # before the profiled region, so the trace reflects steady-state
    # kernel behavior rather than compilation overhead.
    for _ in range(10):
        matmul(A, B)
    torch.cuda.synchronize()

    # Profiled region. Wrap in NVTX ranges so the Nsight Systems
    # timeline clearly marks where the kernel launches are, separate
    # from any Python-side overhead.
    torch.cuda.nvtx.range_push("triton_gemm_profiled_region")
    for _ in range(20):
        matmul(A, B)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()