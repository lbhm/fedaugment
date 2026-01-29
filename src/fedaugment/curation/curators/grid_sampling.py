from collections import defaultdict

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .curator import Curator


class GridSampling(Curator):
    """Divides feature space into a grid and samples uniformly per occupied cell."""

    def __init__(self, alias: str, grid_size: int) -> None:
        super().__init__(alias)
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size

    @torch.no_grad()
    def curate(self, candidates: Tensor, k: int, seed: int) -> Tensor:
        self._validate_inputs(candidates, k)

        n, d = candidates.shape
        if self.grid_size == 1 or d == 0:
            # Degenerate grid reduces to random sampling over all points
            generator = torch.Generator(device=candidates.device).manual_seed(seed)
            return torch.randperm(n, generator=generator, device=candidates.device)[:k]

        bins = self._assign_bins(candidates)
        cell_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, key in enumerate(bins):
            cell_to_indices[key].append(idx)

        occupied_cells = list(cell_to_indices.keys())
        if not occupied_cells:
            raise RuntimeError("No occupied grid cells found; check candidate data")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = torch.empty((k,), dtype=torch.int64, device="cpu")
        order = torch.randperm(len(occupied_cells), generator=generator)
        pointer = 0
        emptied_cells: set[tuple[int, ...]] = set()

        for i in tqdm(
            range(k),
            desc="Selecting points",
            total=k,
            leave=False,
            unit="point",
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            if pointer >= len(order):
                if emptied_cells:
                    occupied_cells = [cell for cell in occupied_cells if cell not in emptied_cells]
                    emptied_cells.clear()

                order = torch.randperm(len(occupied_cells), generator=generator)
                pointer = 0

            cell_idx = occupied_cells[int(order[pointer].item())]
            pointer += 1

            members = cell_to_indices[cell_idx]
            member_idx = int(torch.randint(0, len(members), (1,), generator=generator).item())
            choice = members.pop(member_idx)
            selected[i] = choice

            if not members:
                # Mark cell for deletion if its last member was selected
                cell_to_indices.pop(cell_idx, None)
                emptied_cells.add(cell_idx)

        return selected.to(candidates.device)

    def _assign_bins(self, candidates: Tensor) -> list[tuple[int, ...]]:
        mins = candidates.min(dim=0).values
        maxs = candidates.max(dim=0).values
        ranges = torch.clamp(maxs - mins, min=torch.finfo(candidates.dtype).eps)
        normalized = torch.clamp((candidates - mins) / ranges, min=0.0, max=1.0 - 1e-12)
        scaled = torch.floor(normalized * self.grid_size).to(torch.int64)
        scaled = torch.clamp(scaled, min=0, max=self.grid_size - 1)
        return [tuple(row.tolist()) for row in scaled]
