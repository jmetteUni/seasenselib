"""
Metadata extraction handlers.

Includes extractor implementations and the composite extraction logic.
"""

from .attribute_extractor import AttributeMetadataExtractor
from .global_attribute_extractor import GlobalAttributeMetadataExtractor
from .extraction_runner import MetadataExtractionRunner

__all__ = [
    "AttributeMetadataExtractor",
    "GlobalAttributeMetadataExtractor",
    "MetadataExtractionRunner",
]
