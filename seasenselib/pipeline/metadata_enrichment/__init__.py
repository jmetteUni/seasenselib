"""
Conventions Module

Provides convention implementations (CF, ACDD) and stage orchestration.

Available Conventions:
    - CFConvention: Climate and Forecast Conventions v1.13
    - ACDDConvention: Attribute Convention for Data Discovery v1.3

Stage:
    - MetadataEnrichmentStage: Applies conventions in order
"""

from .handlers.cf_convention import CFConvention
from .handlers.acdd_convention import ACDDConvention
from .stage import MetadataEnrichmentStage

__all__ = [
    'CFConvention',
    'ACDDConvention',
    'MetadataEnrichmentStage',
]
