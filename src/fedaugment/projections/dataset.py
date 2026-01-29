from abc import ABC, abstractmethod
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, SupportsIndex, cast
from weakref import finalize

import h5py
import numpy as np
import orjson
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fedaugment.types import ColIdArray, EmbeddingArray, FloatArray, IntArray, StrPath
from fedaugment.utils import close_hdf5_files


class EmbeddingDataset[T](ABC, Dataset[T]):
    """Abstract base class for embedding datasets."""

    num_views: int
    """The number of models used to generate the embeddings."""
    pipeline_names: list[str]
    """The names of the embedding pipelines used to generate the embeddings."""
    embedding_dims: list[int]
    """The dimensions of the embeddings for each model."""
    metadata: list[dict[str, Any]]
    """Metadata associated with each embedding view."""

    @property
    @abstractmethod
    def column_ids(self) -> ColIdArray:
        """The column IDs corresponding to the embeddings."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of items in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        """Returns the embeddings from each pipeline/view at the specified index.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A dataset item as a sequence of embeddings for each embedding pipeline in the dataset
            or a transformed version therefore if using transforms.
        """

    @abstractmethod
    def get_view(
        self, view_idx: int, sample_idx: SupportsIndex | slice | IntArray | None = None
    ) -> FloatArray:
        """Returns the embeddings for a specific view/model.

        Args:
            view_idx: The index of the view/model to retrieve.
            sample_idx: The index or indices of the samples to retrieve.
                Returns all data by default.
        """


class HDF5Dataset(EmbeddingDataset[Sequence[Tensor]]):
    """An embedding dataset based on HDF5 files."""

    def __init__(
        self, paths_or_dir: StrPath | Sequence[StrPath], verify_column_ids: bool = True
    ) -> None:
        """Initializes an HDF5 embedding dataset.

        Args:
            paths_or_dir: A path to a directory containing HDF5 files or a list of HDF5 file paths.
            verify_column_ids: If True, verifies that all HDF5 files have the same column IDs.
        """
        file_paths = gather_paths(paths_or_dir, file_format=".h5")

        self.files = [h5py.File(file, "r") for file in file_paths]
        self._finalizer = finalize(self, close_hdf5_files, self.files)
        self.num_views = len(self.files)

        #  Process the first file
        embeddings = cast("h5py.Dataset", self.files[0]["embeddings"])
        col_ids = cast("h5py.Dataset", self.files[0]["column_ids"]).astype("T")
        self._column_ids = cast("ColIdArray", col_ids[:])

        # Retrieve the number of samples
        self.num_samples: int = embeddings.shape[0]

        # Initialize list attributes
        self.embeddings = [embeddings]
        self.embedding_dims = [embeddings.shape[1]]
        self.pipeline_names = [str(self.files[0].attrs["pipeline"])]
        self.metadata = [dict(self.files[0].attrs.items())]

        if len(self.files) == 1:
            return

        # Verify that all files have the same number of samples and fill the lists
        for file in self.files[1:]:
            embeddings = cast("h5py.Dataset", file["embeddings"])
            shape = cast("tuple[int, ...]", embeddings.shape)

            if shape[0] != self.num_samples:
                raise ValueError(
                    f"All HDF5 files must have the same number of samples. File {file.filename} "
                    f"has {shape[0]} samples, while the first file has {self.num_samples} samples."
                )
            if verify_column_ids:
                other_col_ids = cast("h5py.Dataset", file["column_ids"])
                if not np.array_equal(self.column_ids, other_col_ids.astype("T")[:]):
                    raise ValueError(
                        f"Column IDs in file {file.filename} do not match those in the first file."
                    )

            self.embeddings.append(embeddings)
            self.embedding_dims.append(shape[1])
            self.pipeline_names.append(str(file.attrs["pipeline"]))
            self.metadata.append(dict(file.attrs.items()))

    @property
    def column_ids(self) -> ColIdArray:
        return self._column_ids

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Sequence[Tensor]:
        return tuple(torch.from_numpy(embedding[index]) for embedding in self.embeddings)

    def get_view(
        self, view_idx: int, sample_idx: SupportsIndex | slice | IntArray | None = None
    ) -> FloatArray:
        out: FloatArray
        if sample_idx is None:
            out = self.embeddings[view_idx][:]
        else:
            out = self.embeddings[view_idx][sample_idx]
        return out


class NPYDataset(EmbeddingDataset[Sequence[Tensor]]):
    """An embedding dataset based on NPY files with parallel access via memory mapping.

    A FedAugment style NPY dataset is a folder with a `.fa` suffix that consists of the following
    files:
    - embeddings.npy: Embeddings array
    - column_ids.npy: Column IDs array
    - metadata.json: Metadata
    """

    embeddings: list[EmbeddingArray]
    """List of memory-mapped embeddings arrays for each view/model."""

    def __init__(
        self,
        root_or_paths: StrPath | Sequence[StrPath],
        verify_column_ids: bool = True,
        mmap_mode: Literal["r+"] | None = "r+",
    ) -> None:
        """Initializes an NPY embedding dataset.

        Args:
            root_or_paths: A path to a directory containing a FedAugment style NPY folder or a list
                of them.
            verify_column_ids: If True, verifies that all NPY files have the same column IDs.
            mmap_mode: The mode to use for memory mapping the NPY files.
        """
        self.dir_paths = gather_paths(root_or_paths, file_format=".fa")
        self.mmap_mode: Literal["r+"] | None = mmap_mode

        # Process the first embedding folder
        embeddings: EmbeddingArray = np.load(self.dir_paths[0] / "embeddings.npy", mmap_mode="r")
        col_ids: ColIdArray = np.load(self.dir_paths[0] / "column_ids.npy", allow_pickle=True)
        metadata = orjson.loads(Path(self.dir_paths[0] / "metadata.json").read_bytes())

        # Initialize state
        self.embedding_dims = [embeddings.shape[1]]
        self.num_samples = embeddings.shape[0]
        self.num_views = len(self.dir_paths)
        self.pipeline_names = [metadata["pipeline"]]
        self.metadata = [metadata]

        for dir_path in self.dir_paths[1:]:
            embeddings = np.load(dir_path / "embeddings.npy", mmap_mode="r")
            self.embedding_dims.append(embeddings.shape[1])

            metadata = orjson.loads(Path(dir_path / "metadata.json").read_bytes())
            self.pipeline_names.append(metadata["pipeline"])
            self.metadata.append(metadata)

            # Verify consistency across files
            if embeddings.shape[0] != self.num_samples:
                raise ValueError(
                    f"All NPY folders must have the same number of samples. Folder {dir_path} "
                    f"has {embeddings.shape[0]} samples (expected {self.num_samples})."
                )
            if verify_column_ids:
                other_col_ids = np.load(dir_path / "column_ids.npy", allow_pickle=True)
                if not np.array_equal(col_ids, other_col_ids):
                    raise ValueError(
                        f"Column IDs in folder {dir_path} do not match those in the first folder."
                    )
                del other_col_ids

        # Cleanup state to reduce worker load time; data will be loaded lazily
        del embeddings, col_ids, metadata

        # Lazy loaded on first access
        self._embeddings: list[EmbeddingArray] | None = None
        self._column_ids: ColIdArray | None = None

    @property
    def column_ids(self) -> ColIdArray:
        if self._column_ids is None:
            self._embeddings, self._column_ids = self._lazy_load()
        return self._column_ids

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Sequence[Tensor]:
        if self._embeddings is None:
            self._embeddings, self._column_ids = self._lazy_load()

        return tuple(torch.from_numpy(embedding[index]) for embedding in self._embeddings)

    def get_view(
        self, view_idx: int, sample_idx: SupportsIndex | slice | IntArray | None = None
    ) -> FloatArray:
        if self._embeddings is None:
            self._embeddings, self._column_ids = self._lazy_load()

        if sample_idx is None:
            return self._embeddings[view_idx]
        return self._embeddings[view_idx][sample_idx]

    def reset_state(self) -> None:
        """Resets the loaded state to force re-loading on next access."""
        self._embeddings = None
        self._column_ids = None

    def _lazy_load(self) -> tuple[list[EmbeddingArray], ColIdArray]:
        """Lazily loads the memory-mapped embeddings arrays for the current worker."""
        # torch does not support non-writable tensors, so we have to use mmap_mode="r+"
        embeddings = [
            np.load(dir_path / "embeddings.npy", mmap_mode=self.mmap_mode)
            for dir_path in self.dir_paths
        ]
        column_ids = cast(
            "ColIdArray", np.load(self.dir_paths[0] / "column_ids.npy", allow_pickle=True)
        )
        return embeddings, column_ids


def gather_paths(
    paths_or_dir: StrPath | Sequence[StrPath], file_format: Literal[".fa", ".h5"]
) -> list[Path]:
    """Gathers dataset file paths from a directory or a list of paths."""
    if isinstance(paths_or_dir, (str, PathLike)):
        path = Path(paths_or_dir)
        if not path.is_dir():
            raise ValueError("paths_or_dir must be a directory or a list of files.")
        if file_format == ".fa":
            paths = sorted([p for p in path.iterdir() if p.is_dir() and p.suffix == ".fa"])
        else:
            paths = sorted(path.glob("*.h5"))
    else:
        paths = [Path(p) for p in paths_or_dir]

    if len(paths) < 1:
        raise RuntimeError(
            f"No {file_format} dataset files found. You need to provide at least one."
        )

    return paths
