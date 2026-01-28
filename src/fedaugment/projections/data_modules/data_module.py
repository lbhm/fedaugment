import lightning as L
from torch import Generator

from fedaugment.config import DataModuleConfig


class DataModule(L.LightningDataModule):
    def __init__(self, config: DataModuleConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.generator = Generator().manual_seed(seed)
        self.is_setup = False

        self._embedding_dims: list[int] = []
        self._pipeline_names: list[str] = []

    @property
    def embedding_dims(self) -> list[int]:
        """Get the embedding dimensions for each pipeline in this data module."""
        if not self.is_setup:
            self.setup(stage="dummy")
        return self._embedding_dims

    @property
    def pipeline_names(self) -> list[str]:
        """Get the names of the embedding pipelines in this data module."""
        if not self.is_setup:
            self.setup(stage="dummy")
        return self._pipeline_names
