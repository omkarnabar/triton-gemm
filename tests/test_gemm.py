import torch

from kernels.gemm import matmul


def test_gemm_correctness(device, shape):
    M, K = shape
    N = K  # square-ish N tied to K; change if you want N independent

    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    C_triton = matmul(A, B)
    C_ref = torch.matmul(A, B)

    torch.testing.assert_close(C_triton, C_ref, atol=1e-1, rtol=1e-1)


def test_gemm_output_shape(device, shape):
    M, K = shape
    N = K

    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    C = matmul(A, B)

    assert C.shape == (M, N)
    assert C.dtype == torch.float16
    assert C.is_cuda