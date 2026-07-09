import pytest
import torch

# Assuming your attention function is exposed from kernels.attention
from kernels.attention import attention


@pytest.mark.parametrize(
    "B, H, N_CTX, D_HEAD",
    [
        (1, 1, 64, 64),
        (2, 4, 128, 32),
        (4, 8, 256, 64),
        (2, 16, 512, 64),
        (1, 8, 1024, 64),
        (1, 4, 2048, 128),  # Deep dive test for large context sizes
    ],
)
def test_attention_correctness(device, B, H, N_CTX, D_HEAD):
    torch.manual_seed(0)

    q = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    k = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    v = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)

    # Triton Causal FlashAttention forward execution
    o_triton = attention(q, k, v)

    # Reference implementation using standard PyTorch eager execution with a causal mask
    # 1. Compute scores: Q @ K^T / sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (D_HEAD ** 0.5))
    
    # 2. Apply explicit causal mask
    mask = torch.triu(torch.full((N_CTX, N_CTX), float('-inf'), device=device), diagonal=1)
    scores = scores + mask[None, None, :, :]
    
    # 3. Softmax along key-dimension and multiply with V
    p = torch.softmax(scores, dim=-1).to(torch.float16)
    o_ref = torch.matmul(p, v)

    # Higher tolerances because Triton online softmax (FP32 accumulations) and 
    # eager torch FP16 materializations have slight arithmetic rounding variations.
    torch.testing.assert_close(
        o_triton,
        o_ref,
        atol=1e-1,
        rtol=1e-1,
    )


@pytest.mark.parametrize(
    "B, H, N_CTX, D_HEAD",
    [
        (1, 2, 17, 32),
        (2, 3, 100, 64),
        (1, 4, 511, 64),
    ],
)
def test_attention_non_power_of_two_shapes(device, B, H, N_CTX, D_HEAD):
    """
    Check attention sequence lengths that do not directly align 
    to typical Triton BLOCK_M / BLOCK_N boundaries to test masks.
    """
    q = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    k = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    v = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)

    o_triton = attention(q, k, v)

    # Reference computation
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (D_HEAD ** 0.5))
    mask = torch.triu(torch.full((N_CTX, N_CTX), float('-inf'), device=device), diagonal=1)
    scores = scores + mask[None, None, :, :]
    p = torch.softmax(scores, dim=-1).to(torch.float16)
    o_ref = torch.matmul(p, v)

    torch.testing.assert_close(
        o_triton,
        o_ref,
        atol=1e-1,
        rtol=1e-1,
    )


def test_attention_output_properties(device):
    B, H, N_CTX, D_HEAD = 2, 4, 128, 64

    q = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    k = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    v = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)

    o = attention(q, k, v)

    assert o.shape == (B, H, N_CTX, D_HEAD)
    assert o.dtype == torch.float16
    assert o.device.type == "cuda"


def test_attention_causal_mask_isolation(device):
    """
    Verifies that changing values in the causal 'future' area 
    has zero effect on the generated output past.
    """
    B, H, N_CTX, D_HEAD = 1, 1, 64, 32
    
    q = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    k1 = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    v1 = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    
    # Run a baseline pass
    o1 = attention(q, k1, v1)
    
    # Mutate only the elements strictly in the future (e.g. at the end of the sequence)
    k2 = k1.clone()
    v2 = v1.clone()
    k2[:, :, -5:, :] = torch.randn_like(k2[:, :, -5:, :])
    v2[:, :, -5:, :] = torch.randn_like(v2[:, :, -5:, :])
    
    # Re-run attention with modified future parameters
    o2 = attention(q, k2, v2)
    
    # The output for everything EXCEPT those last 5 sequence rows should remain identical
    torch.testing.assert_close(
        o1[:, :, :-5, :],
        o2[:, :, :-5, :],
        atol=1e-3,
        rtol=1e-3,
    )


def test_attention_extreme_values(device):
    """
    Validates safe-softmax numerical stability under highly divergent value conditions.
    """
    B, H, N_CTX, D_HEAD = 1, 1, 32, 32
    
    # Inject large magnitudes that would typically trigger over/underflow in unsafe softmax
    q = torch.full((B, H, N_CTX, D_HEAD), 50.0, device=device, dtype=torch.float16)
    k = torch.full((B, H, N_CTX, D_HEAD), -50.0, device=device, dtype=torch.float16)
    v = torch.randn((B, H, N_CTX, D_HEAD), device=device, dtype=torch.float16)
    
    o = attention(q, k, v)
    
    # Ensure values didn't NaN out due to exp() overloads
    assert not torch.isnan(o).any()
    assert not torch.isinf(o).any()