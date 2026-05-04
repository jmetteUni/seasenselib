"""
Metadata extraction runner.

Composite extractor that combines multiple metadata extractors.
"""

from typing import List
import logging
from ...base import StageContext
from ...interfaces import IMetadataExtractor, MetadataRegistry
from .attribute_extractor import AttributeMetadataExtractor
from .global_attribute_extractor import GlobalAttributeMetadataExtractor

logger = logging.getLogger(__name__)

class MetadataExtractionRunner:
    """
    Extracts metadata from datasets using multiple extractors (Composite pattern).
    
    Combines metadata from:
    - Variable attributes (units, long_name, etc.)
    - Global attributes (title, institution, etc.)
    - Custom extractors (can be added)
    
    The extracted metadata is stored in a MetadataRegistry and can be used
    by other stages for enrichment.
    
    Usage:
        extractor = MetadataExtractionRunner()
        enriched_ds = extractor.process(context)
        # Metadata available in dataset.attrs['metadata_registry']
    """
    
    def __init__(self, extractors: List[IMetadataExtractor] = None):
        """
        Initialize metadata extraction logic.
        
        Args:
            extractors: Optional list of IMetadataExtractor implementations.
                       If None, uses default extractors (attribute + global)
        """
        # Use default extractors if none provided
        if extractors is None:
            extractors = [
                AttributeMetadataExtractor(),
                GlobalAttributeMetadataExtractor()
            ]
        
        self.extractors = extractors
    
    def process(self, context: StageContext) -> StageContext:
        """
        Process dataset by extracting metadata from all sources.
        
        Args:
            context: StageContext with dataset and metadata
        
        Returns:
            Updated StageContext with metadata_registry
        """
        # Start with existing registry or create new one
        metadata_registry = context.metadata.get('_metadata_registry')
        if metadata_registry is None:
            registry = MetadataRegistry()
        else:
            registry = metadata_registry
        
        # Extract from all sources
        for extractor in self.extractors:
            try:
                extracted = extractor.extract(context.dataset, context.metadata)
                registry = registry.merge(extracted)
                logger.debug(f"Extracted metadata from {extractor.name()}")
            except Exception as e:
                logger.warning(f"Failed to extract from {extractor.name()}: {e}")
        
        # Store registry in context metadata
        context.metadata['_metadata_registry'] = registry
        
        logger.info(f"Extracted metadata from {len(self.extractors)} source(s)")
        
        # Return updated context
        return context
    
    def add_extractor(self, extractor: IMetadataExtractor):
        """
        Add a new metadata extractor.
        
        Args:
            extractor: IMetadataExtractor implementation to add
        """
        self.extractors.append(extractor)
        logger.debug(f"Added extractor: {extractor.name()}")
    
    def list_extractors(self) -> List[str]:
        """Return list of extractor source names"""
        return [e.name() for e in self.extractors]
