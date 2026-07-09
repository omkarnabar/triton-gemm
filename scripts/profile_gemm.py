"""
Profile the Triton GEMM kernel using torch.profiler.

Since NSight Compute cannot be used in a Kaggle Environment, I resorted to using torch.profiler.
If ncu does work, use that.

Run from repo root:
    python scripts/profile_gemm_torch.py

Or in a Kaggle/Jupyter cell:
    !python scripts/profile_gemm_torch.py

Outputs:
    - A printed summary table sorted by CUDA time
    - A Chrome trace JSON 
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from kernels.gemm import matmul


def main():
    torch.manual_seed(0)

    M = N = K = 4096

    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)

    # Warmup — lets the autotuner pick a config and JIT-compile before
    # the profiled region, so the trace reflects steady-state kernel
    # behavior rather than compilation/tuning overhead.
    for _ in range(10):
        matmul(A, B)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("triton_gemm_profiled_region"):
            for _ in range(20):
                matmul(A, B)
            torch.cuda.synchronize()

    # ---- Console summary: time breakdown by op ----
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

    print("\nSummary grouped by input shape:")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

    # ---- Memory summary ----
    print("\nMemory summary (CUDA):")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10,
    ))

    # ---- Export chrome trace for visual timeline ----
    trace_path = "scripts/out/gemm_trace.json"
    prof.export_chrome_trace(trace_path)


if __name__ == "__main__":
    main()