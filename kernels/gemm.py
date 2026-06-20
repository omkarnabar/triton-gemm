import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE': 8},
            num_warps=8,
            num_stages=2
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE': 8},
            num_warps=4,
            num_stages=2
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE': 8},
            num_warps=4,
            num_stages=2
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):

    # --------------------------------------------------------
    # program ids
    # --------------------------------------------------------

    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE * num_pid_n

    group_id = pid // num_pid_in_group

    first_pid_m = group_id * GROUP_SIZE

    group_size_m = tl.minimum(
        num_pid_m - first_pid_m,
        GROUP_SIZE
    )

    pid_m = first_pid_m + (
        (pid % num_pid_in_group) % group_size_m
    )

    pid_n = (
        pid % num_pid_in_group
    ) // group_size_m

    # --------------------------------------------------------
    # offsets
    # --------------------------------------------------------

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # --------------------------------------------------------
    # masks
    # --------------------------------------------------------

    rm_mask = rm < M
    rn_mask = rn < N

    # --------------------------------------------------------
    # pointers
    # --------------------------------------------------------

    A_ptr = A + (
        rm[:, None] * stride_am +
        rk[None, :] * stride_ak
    )

    B_ptr = B + (
        rk[:, None] * stride_bk +
        rn[None, :] * stride_bn
    )

    # --------------------------------------------------------
    # accumulator
    # --------------------------------------------------------

    acc = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32
    )

    # --------------------------------------------------------
    # K loop
    # --------------------------------------------------------

    for k in range(0, K, BLOCK_K):

        k_mask = rk < (K - k)

        a = tl.load(
            A_ptr,
            mask=rm_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        b = tl.load(
            B_ptr,
            mask=k_mask[:, None] & rn_mask[None, :],
            other=0.0
        )

        acc += tl.dot(a, b)

        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    # --------------------------------------------------------
    # store
    # --------------------------------------------------------

    C_ptr = C + (
        rm[:, None] * stride_cm +
        rn[None, :] * stride_cn
    )

    c_mask = rm_mask[:, None] & rn_mask[None, :]

    tl.store(
        C_ptr,
        acc.to(tl.float16),
        mask=c_mask
    )


# ============================================================
# Python wrapper
# ============================================================

def matmul(A, B):

    assert A.shape[1] == B.shape[0]

    A = A.contiguous().half()
    B = B.contiguous().half()

    M, K = A.shape
    _, N = B.shape

    C = torch.empty(
        (M, N),
        device='cuda',
        dtype=torch.float16
    )

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) *
        triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,

        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C