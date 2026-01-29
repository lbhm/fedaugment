"""This module contains methods for curating training data from tabular dataset collections.

The curation methods can be used to create a minimal representative training dataset.
"""

from .curation_manager import CurationManager
from .curators.curator import Curator

__all__ = ["CurationManager", "Curator"]
