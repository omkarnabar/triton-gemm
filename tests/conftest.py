import torch
import pytest

def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_no_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            item.add_marker(skip_no_cuda)
            
@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)

@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda")

@pytest.fixture(params=[(128, 64), (512, 128), (1000, 64)])  # N, d — note non-power-of-2
def shape(request):
    return request.param