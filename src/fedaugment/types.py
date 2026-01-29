import os
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

type EmbeddingArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
# NOTE: As of now, we cannot use np.dtypes.StringDType because StringDType is not a scalar type.
type ColIdArray = np.ndarray[tuple[int], np.dtype[Any]]

# TODO: Replace StrArray and FloatArray occurrences with the specialized types above when possible
type FloatArray = NDArray[np.floating]
type IntArray = NDArray[np.integer]
type StrArray = NDArray[Any]

type StrPath = str | os.PathLike[str]
type HDF5Dict = dict[
    str, tuple[tuple[int, ...], DTypeLike, dict[str, Any]] | tuple[HDF5Dict, dict[str, Any]]
]
