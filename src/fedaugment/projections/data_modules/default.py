from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split

from fedaugment.config import ExplicitDataSplit, FractionalDataSplit
from fedaugment.projections.dataset import NPYDataset

from .data_module import DataModule

EXPLICIT_DATA_SPLIT_LENGTH = 3


class DefaultDataModule(DataModule):
    def setup(self, stage: str) -> None:
        if self.is_setup:
            return

        self.train_dataset: NPYDataset | Subset[Any] | None = None
        self.val_dataset: NPYDataset | Subset[Any] | None = None
        self.test_dataset: NPYDataset | Subset[Any] | None = None

        if isinstance(self.config.data, ExplicitDataSplit):
            self._setup_from_explicit_splits(self.config.data)
        else:
            self._setup_from_fractional_splits(self.config.data)

        self.is_setup = True

    # NOTE: The DataLoader type does not consider the collate_fn but only looks at the dataset's
    # item type. Therefore, we cannot annotate the DataLoader return type. This is a limitation of
    # the type annotations in PyTorch.

    def train_dataloader(self) -> DataLoader[Any] | None:
        if self.train_dataset is None or len(self.train_dataset) == 0:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=stack_views,
            pin_memory=self.config.pin_memory,
            generator=self.generator,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Any] | None:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=stack_views,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Any] | None:
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=stack_views,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=False,
        )

    def _setup_from_explicit_splits(self, split: ExplicitDataSplit) -> None:
        train_dataset = self._build_dataset(split.train) if split.train is not None else None
        val_dataset = self._build_dataset(split.val) if split.val is not None else None
        test_dataset = self._build_dataset(split.test) if split.test is not None else None

        splits = [ds for ds in (train_dataset, val_dataset, test_dataset) if ds is not None]
        self._validate_split_compatibility(splits)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self._embedding_dims = splits[0].embedding_dims
        self._pipeline_names = splits[0].pipeline_names

    def _setup_from_fractional_splits(self, split: FractionalDataSplit) -> None:
        dataset = self._build_dataset(split.root)

        # Convert fractional splits to integer lengths and distribute the remainder among non-empty
        # splits in a round-robin fashion
        subset_lengths = [int(frac * len(dataset)) for frac in split.fractions]
        remainder = len(dataset) - sum(subset_lengths)

        for i in range(remainder):
            idx = i % len(subset_lengths)
            while split.fractions[idx] == 0:
                idx = (idx + 1) % len(subset_lengths)
            subset_lengths[idx] += 1

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, subset_lengths, generator=self.generator
        )
        self._embedding_dims = dataset.embedding_dims
        self._pipeline_names = dataset.pipeline_names

    def _build_dataset(self, paths: Path | Sequence[Path]) -> NPYDataset:
        return NPYDataset(
            paths, verify_column_ids=self.config.verify_column_ids, mmap_mode=self.config.mmap_mode
        )

    def _validate_split_compatibility(self, datasets: list[NPYDataset]) -> None:
        if not datasets:
            raise ValueError("At least one dataset split must be provided.")

        ref_dims = datasets[0].embedding_dims
        ref_pipelines = datasets[0].pipeline_names

        for dataset in datasets[1:]:
            if dataset.embedding_dims != ref_dims:
                raise ValueError(
                    "All provided dataset splits must share identical embedding dimensions."
                )
            if dataset.pipeline_names != ref_pipelines:
                raise ValueError(
                    "All provided dataset splits must share identical pipeline names."
                )


def stack_views(batch: Sequence[Sequence[Tensor]]) -> list[Tensor]:
    """Stack the different views across the samples in a batch.

    Args:
        batch: A list of B samples, where each sample is a list of V tensors with shape (D_i,).

    Returns:
        A list of V tensors, where each tensor has shape (B, D_i).
    """
    return [torch.stack([sample[i] for sample in batch]) for i in range(len(batch[0]))]
