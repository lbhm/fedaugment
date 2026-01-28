import pytest
import torch

from fedaugment.curation.curators import (
    FarthestFirstTraversal,
    GridSampling,
    KMedoids,
    RandomSampling,
)

DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    ),
]


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_random_sampling_matches_torch_randperm(device: str) -> None:
    candidates = torch.arange(50, dtype=torch.float32, device=device).view(-1, 5)
    curator = RandomSampling(alias="rand")
    seed = 1337
    k = 4

    result = curator.curate(candidates, k=k, seed=seed)

    generator = torch.Generator(device=device).manual_seed(seed)
    expected = torch.randperm(len(candidates), generator=generator, device=device)[:k]

    assert torch.equal(result, expected)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_random_sampling_returns_all_indices_when_k_equals_n(device: str) -> None:
    candidates = torch.arange(12, dtype=torch.float32, device=device).view(-1, 3)
    curator = RandomSampling(alias="rand")

    result = curator.curate(candidates, k=len(candidates), seed=99)
    expected = torch.arange(len(candidates), device=device)

    assert torch.equal(result.sort().values, expected)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_grid_sampling_visits_all_cells_once(device: str) -> None:
    candidates = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=device
    )
    curator = GridSampling(alias="grid", grid_size=2)

    result = curator.curate(candidates, k=4, seed=7)
    expected = torch.tensor([0, 1, 2, 3], device=device)

    assert torch.equal(result.sort().values, expected)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_grid_sampling_with_single_cell_falls_back_to_random(device: str) -> None:
    candidates = torch.arange(50, dtype=torch.float32, device=device).view(-1, 5)
    curator = GridSampling(alias="grid", grid_size=1)
    seed = 123

    result = curator.curate(candidates, k=6, seed=seed)

    generator = torch.Generator(device=device).manual_seed(seed)
    expected = torch.randperm(len(candidates), generator=generator, device=device)[:6]

    assert torch.equal(result, expected)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_farthest_first_traversal_cosine(device: str) -> None:
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
    curator = FarthestFirstTraversal(alias="fft", metric="cosine")

    result = curator.curate(candidates, k=2, seed=7)
    expected = torch.tensor([0, 1], device=device)

    assert torch.equal(result, expected)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_farthest_first_traversal_euclidean(device: str) -> None:
    candidates = torch.tensor([[0.0], [1.0], [3.0], [10.0]], device=device)
    curator = FarthestFirstTraversal(alias="fft", metric="euclidean")

    result = curator.curate(candidates, k=3, seed=28)
    expected = torch.tensor([1, 3, 2], device=device)

    assert torch.equal(result, expected)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_k_medoids_separates_simple_clusters_cosine(device: str) -> None:
    cluster_a = torch.tensor([[1.0, 0.0], [0.9, 0.1]], dtype=torch.float32, device=device)
    cluster_b = torch.tensor([[0.0, 1.0], [0.1, 0.9]], dtype=torch.float32, device=device)
    candidates = torch.cat([cluster_a, cluster_b], dim=0)
    curator = KMedoids(alias="kmed", metric="cosine", max_iters=10)

    result = curator.curate(candidates, k=2, seed=2024)

    assert (((result == 0) | (result == 1)).any()).item()
    assert (((result == 2) | (result == 3)).any()).item()
    assert torch.unique(result).numel() == 2


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_k_medoids_returns_all_indices_when_k_equals_n(device: str) -> None:
    candidates = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
    curator = KMedoids(alias="kmed", metric="euclidean")

    result = curator.curate(candidates, k=len(candidates), seed=0)
    expected = torch.tensor([0, 1], device=device)

    assert torch.equal(result, expected)
