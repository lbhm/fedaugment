from typing import Any

from torch.utils.data import DataLoader

from .default import DefaultDataModule, stack_views


class SingleStepTrainingDataModule(DefaultDataModule):
    """DataModule for models that require full-dataset single-batch training."""

    def train_dataloader(self) -> DataLoader[Any] | None:
        if self.train_dataset is None or len(self.train_dataset) == 0:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=len(self.train_dataset),
            shuffle=False,
            num_workers=0,
            collate_fn=stack_views,
            pin_memory=self.config.pin_memory,
            worker_init_fn=None,
            generator=None,
            prefetch_factor=None,
            persistent_workers=False,
        )
