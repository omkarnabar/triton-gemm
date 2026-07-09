import torch
import triton
import triton.language as tl


# ============================================================
# Causal FlashAttention — Forward Kernel
# ============================================================
#
# Computes O = softmax(Q @ K^T / sqrt(d) , causal mask) @ V
# without ever materializing the full (seq_len x seq_len) attention
# matrix in HBM. Q, K, V tiles are loaded once; running softmax
# statistics (m_i, l_i) let us normalize correctly across K/V tiles
# processed sequentially, fused into a single kernel launch.
#
# This builds directly on the GEMM kernel's tiling/masking pattern
# (kernels/gemm.py), with two additions:
#   1. Online softmax: track running max (m_i) and running sum (l_i)
#      per query row, rescaling the accumulator as new K/V tiles
#      shift the max.
#   2. Causal masking: skip K/V tiles entirely once they're fully in
#      the future relative to the current query tile.

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, head_dim,
    sm_scale,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # --------------------------------------------------------
    # program ids: one program per (query block, batch*head)
    # --------------------------------------------------------
    pid_m = tl.program_id(0)        # which query tile
    pid_bh = tl.program_id(1)       # which (batch, head) pair

    # Offset base pointers to the right (batch, head) slice.
    # stride_qh already encodes the combined batch*head stride
    # layout set up by the wrapper below.
    Q += pid_bh * stride_qh
    K += pid_bh * stride_kh
    V += pid_bh * stride_vh
    O += pid_bh * stride_oh

    # --------------------------------------------------------
    # offsets
    # --------------------------------------------------------
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query rows
    d_offsets = tl.arange(0, HEAD_DIM)                     # head_dim cols

    m_mask = m_offsets < seq_len

    # --------------------------------------------------------
    # load Q tile once — stays resident for the whole K/V loop
    # --------------------------------------------------------
    q_ptrs = Q + (
        m_offsets[:, None] * stride_qm +
        d_offsets[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # --------------------------------------------------------
    # online softmax running state
    # --------------------------------------------------------
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                       # running sum
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)              # running output

    # Causal: query tile starting at pid_m * BLOCK_M only ever needs
    # K/V tiles up to and including its own position. Tiles fully in
    # the future are skipped entirely — never loaded, never computed.
    hi = (pid_m + 1) * BLOCK_M

    for n_start in range(0, hi, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seq_len

        # ---- load K, V tile ----
        k_ptrs = K + (
            n_offsets[:, None] * stride_kn +
            d_offsets[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        v_ptrs = V + (
            n_offsets[:, None] * stride_vn +
            d_offsets[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # ---- scores: Q @ K^T ----
        scores = tl.dot(q, tl.trans(k)) * sm_scale

        # ---- causal mask: query i can only see key j <= i ----
        causal_mask = m_offsets[:, None] >= n_offsets[None, :]
        valid_mask = causal_mask & m_mask[:, None] & n_mask[None, :]
        scores = tl.where(valid_mask, scores, float('-inf'))

        # ---- online softmax update ----
        m_ij = tl.max(scores, axis=1)                 # tile-local row max
        m_new = tl.maximum(m_i, m_ij)

        # Guard against rows that are still entirely masked (e.g. a
        # query row past seq_len) — avoid exp(-inf - -inf) = NaN.
        m_new_safe = tl.where(m_new == float('-inf'), 0.0, m_new)

        p = tl.exp(scores - m_new_safe[:, None])
        alpha = tl.exp(m_i - m_new_safe)               # rescale factor for old accumulator

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # --------------------------------------------------------
    # final normalize and store
    # --------------------------------------------------------
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)  # avoid div-by-zero for fully-masked rows
    acc = acc / l_i_safe[:, None]

    o_ptrs = O + (
        m_offsets[:, None] * stride_om +
        d_offsets[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ============================================================
# Python wrapper
# ============================================================

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal FlashAttention forward pass.

    q, k, v: (batch, heads, seq_len, head_dim) float16 CUDA tensors
    returns: (batch, heads, seq_len, head_dim) float16 CUDA tensor
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16

    batch, heads, seq_len, head_dim = q.shape
    assert head_dim in (16, 32, 64, 128), "HEAD_DIM must be a power of 2 for tl.arange"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Collapse (batch, heads) into a single combined stride for simple
    # pointer offsetting inside the kernel: pid_bh * stride_qh covers
    # both batch and head indexing since q is contiguous.
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_M']),
        batch * heads,
    )

    _attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len, head_dim,
        sm_scale,
        HEAD_DIM=head_dim,
    )

    return o