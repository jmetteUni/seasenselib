"""
Global Attribute Metadata Extractor

Extracts metadata from global (dataset-level) attributes in xarray Datasets.
"""

from typing import Any, Dict, Optional
import xarray as xr
from ...interfaces import IMetadataExtractor, MetadataRegistry


class GlobalAttributeMetadataExtractor(IMetadataExtractor):
    """
    Extracts metadata from global (dataset-level) attributes.
    
    Looks for metadata in dataset.attrs such as:
    - title
    - institution
    - source
    - history
    - references
    - comment
    - Conventions
    - instrument information
    - etc.
    
    Usage:
        extractor = GlobalAttributeMetadataExtractor()
        registry = extractor.extract(dataset)
    """
    
    def name(self) -> str:
        """Return the name of this extractor."""
        return "global_attribute_extractor"
    
    def extract(self, dataset: xr.Dataset, context: Optional[Dict[str, Any]] = None) -> MetadataRegistry:
        """
        Extract metadata from global attributes.
        
        Args:
            source: xarray Dataset with global attributes
        
        Returns:
            MetadataRegistry containing extracted metadata
        """
        registry = MetadataRegistry()
        
        if not isinstance(dataset, xr.Dataset):
            return registry
        
        # Extract all global attributes
        for attr_name, attr_value in dataset.attrs.items():
            registry.add(
                f"global.{attr_name}", 
                attr_value, 
                source=self.source_name()
            )
        
        # Extract commonly used CF/ACDD attributes with semantic keys
        self._extract_standard_attributes(dataset, registry)
        
        return registry
    
    def _extract_standard_attributes(self, dataset: xr.Dataset, registry: MetadataRegistry):
        """Extract standard global attributes with semantic keys"""
        
        # ACDD (Attribute Convention for Data Discovery) attributes
        acdd_attrs = [
            'title',
            'summary',
            'keywords',
            'Conventions',
            'id',
            'naming_authority',
            'history',
            'source',
            'processing_level',
            'comment',
            'acknowledgment',
            'license',
            'standard_name_vocabulary',
            'date_created',
            'creator_name',
            'creator_email',
            'creator_url',
            'institution',
            'project',
            'publisher_name',
            'publisher_email',
            'publisher_url',
            'geospatial_lat_min',
            'geospatial_lat_max',
            'geospatial_lon_min',
            'geospatial_lon_max',
            'geospatial_vertical_min',
            'geospatial_vertical_max',
            'geospatial_vertical_positive',
            'time_coverage_start',
            'time_coverage_end',
            'time_coverage_duration',
            'time_coverage_resolution',
            'featureType',
            'cdm_data_type',
        ]
        
        for attr in acdd_attrs:
            if attr in dataset.attrs:
                registry.add(
                    f"acdd.{attr}",
                    dataset.attrs[attr],
                    source=self.source_name()
                )
        
        # Instrument-specific attributes
        instrument_attrs = [
            'instrument',
            'instrument_serial_number',
            'instrument_manufacturer',
            'instrument_model',
            'sensor_serial_number',
        ]
        
        for attr in instrument_attrs:
            if attr in dataset.attrs:
                registry.add(
                    f"instrument.{attr}",
                    dataset.attrs[attr],
                    source=self.source_name()
                )
    
    @staticmethod
    def source_name() -> str:
        """Return name of this metadata source"""
        return "global_attributes"
