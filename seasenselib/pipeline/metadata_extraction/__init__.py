"""
Metadata Extractors Module

Provides metadata extraction implementations and stage orchestration.

Available Extractors:
    - AttributeMetadataExtractor: Extract from variable attributes
    - GlobalAttributeMetadataExtractor: Extract from global attributes
    - MetadataExtractionRunner: Composite extractor runner

Stage:
    - MetadataExtractionStage: Composite extraction stage
"""

from .handlers.attribute_extractor import AttributeMetadataExtractor
from .handlers.global_attribute_extractor import GlobalAttributeMetadataExtractor
from .handlers.extraction_runner import MetadataExtractionRunner
from .stage import MetadataExtractionStage

__all__ = [
    'AttributeMetadataExtractor',
    'GlobalAttributeMetadataExtractor',
    'MetadataExtractionRunner',
    'MetadataExtractionStage',
]
