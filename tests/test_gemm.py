import pytest
import torch

from kernels.gemm import matmul


@pytest.mark.parametrize(
    "M,N,K",
    [
        (16, 16, 16),
        (32, 64, 16),
        (64, 32, 128),
        (128, 128, 128),
        (256, 512, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 1024, 512),
    ],
)
def test_gemm_correctness(device, M, N, K):
    torch.manual_seed(0)

    A = torch.randn(
        M,
        K,
        device=device,
        dtype=torch.float16,
    )

    B = torch.randn(
        K,
        N,
        device=device,
        dtype=torch.float16,
    )

    C_triton = matmul(A, B)
    C_ref = torch.matmul(A, B)

    torch.testing.assert_close(
        C_triton,
        C_ref,
        atol=1e-1,
        rtol=1e-1,
    )


@pytest.mark.parametrize(
    "M,N,K",
    [
        (17, 31, 23),
        (100, 200, 300),
        (511, 513, 257),
    ],
)
def test_gemm_non_power_of_two_shapes(device, M, N, K):
    """
    Check shapes that are not aligned to typical Triton block sizes.
    """
    A = torch.randn(
        M,
        K,
        device=device,
        dtype=torch.float16,
    )

    B = torch.randn(
        K,
        N,
        device=device,
        dtype=torch.float16,
    )

    C_triton = matmul(A, B)
    C_ref = torch.matmul(A, B)

    torch.testing.assert_close(
        C_triton,
        C_ref,
        atol=1e-1,
        rtol=1e-1,
    )


def test_gemm_output_properties(device):
    M, N, K = 256, 128, 512

    A = torch.randn(
        M,
        K,
        device=device,
        dtype=torch.float16,
    )

    B = torch.randn(
        K,
        N,
        device=device,
        dtype=torch.float16,
    )

    C = matmul(A, B)

    assert C.shape == (M, N)
    assert C.dtype == torch.float16
    assert C.device.type == "cuda"


def test_gemm_zero_input(device):
    """
    Ensure kernel handles zero values correctly.
    """
    A = torch.zeros(
        (128, 128),
        device=device,
        dtype=torch.float16,
    )

    B = torch.randn(
        (128, 128),
        device=device,
        dtype=torch.float16,
    )

    C = matmul(A, B)

    assert torch.all(C == 0)


def test_gemm_identity(device):
    """
    Check multiplication with identity matrix.
    """
    N = 128

    A = torch.randn(
        (N, N),
        device=device,
        dtype=torch.float16,
    )

    I = torch.eye(
        N,
        device=device,
        dtype=torch.float16,
    )

    C = matmul(A, I)

    torch.testing.assert_close(
        C,
        A,
        atol=1e-1,
        rtol=1e-1,
    )