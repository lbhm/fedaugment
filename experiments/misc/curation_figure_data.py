import torch

from fedaugment.curation.curators import FarthestFirstTraversal, GridSampling

MIN_VAL = 0.1
GRID_WIDTH = 3.4
GRID_SIZE = 3
K = 6
SEED = 42

torch.set_printoptions(precision=1)  # type: ignore[no-untyped-call]

data = torch.rand((20, 2), generator=torch.Generator().manual_seed(42))
data = data * GRID_WIDTH + MIN_VAL
data.round_(decimals=1)

fft = FarthestFirstTraversal("fft", metric="euclidean")
grid = GridSampling("grid", GRID_SIZE)

curated_fft = fft.curate(data, k=K, seed=SEED)
curated_grid = grid.curate(data, k=K, seed=SEED)

print(data)
print(curated_fft)
print(curated_grid)
