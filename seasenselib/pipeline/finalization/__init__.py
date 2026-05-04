"""
Finalization stage components.

Provides global attribute handling and dataset sorting.
"""

from .handlers.global_attributes import GlobalAttributes
from .handlers.sorting import Sorting
from .stage import FinalizationStage

__all__ = [
    "GlobalAttributes",
    "Sorting",
    "FinalizationStage",
]
