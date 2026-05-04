"""
Attribute Metadata Extractor

Extracts metadata from variable attributes in xarray Datasets.
"""

from typing import Any, Dict, Optional
import xarray as xr
from ...interfaces import IMetadataExtractor, MetadataRegistry


class AttributeMetadataExtractor(IMetadataExtractor):
    """
    Extracts metadata from variable attributes in xarray Datasets.
    
    Looks for metadata in xarray DataArray attributes such as:
    - units
    - long_name
    - standard_name
    - valid_min, valid_max
    - _FillValue
    - comment
    - etc.
    
    Usage:
        extractor = AttributeMetadataExtractor()
        registry = extractor.extract(dataset)
    """
    
    def name(self) -> str:
        """Return the name of this extractor."""
        return "attribute_metadata_extractor"
    
    def extract(self, dataset: xr.Dataset, context: Optional[Dict[str, Any]] = None) -> MetadataRegistry:
        """
        Extract metadata from variable attributes.
        
        Args:
            source: xarray Dataset containing variables with attributes
        
        Returns:
            MetadataRegistry containing extracted metadata
        """
        registry = MetadataRegistry()
        
        if not isinstance(dataset, xr.Dataset):
            return registry
        
        # Extract from each data variable
        for var_name, data_array in dataset.data_vars.items():
            self._extract_variable_attributes(var_name, data_array, registry)
        
        # Extract from coordinates too
        for coord_name, coord_array in dataset.coords.items():
            self._extract_variable_attributes(coord_name, coord_array, registry)
        
        return registry
    
    def _extract_variable_attributes(
        self, 
        var_name: str, 
        data_array: xr.DataArray, 
        registry: MetadataRegistry
    ):
        """Extract attributes from a single variable"""
        # Standard CF attributes
        if 'units' in data_array.attrs:
            registry.add(f"{var_name}.units", data_array.attrs['units'], source=self.source_name())
        
        if 'long_name' in data_array.attrs:
            registry.add(f"{var_name}.long_name", data_array.attrs['long_name'], source=self.source_name())
        
        if 'standard_name' in data_array.attrs:
            registry.add(f"{var_name}.standard_name", data_array.attrs['standard_name'], source=self.source_name())
        
        # Value range
        if 'valid_min' in data_array.attrs:
            registry.add(f"{var_name}.valid_min", data_array.attrs['valid_min'], source=self.source_name())
        
        if 'valid_max' in data_array.attrs:
            registry.add(f"{var_name}.valid_max", data_array.attrs['valid_max'], source=self.source_name())
        
        # Fill value
        if '_FillValue' in data_array.attrs:
            registry.add(f"{var_name}._FillValue", data_array.attrs['_FillValue'], source=self.source_name())
        
        # Comments and descriptions
        if 'comment' in data_array.attrs:
            registry.add(f"{var_name}.comment", data_array.attrs['comment'], source=self.source_name())
        
        # Store all attributes as a dict for reference
        registry.add(f"{var_name}.all_attributes", dict(data_array.attrs), source=self.source_name())
    
    @staticmethod
    def source_name() -> str:
        """Return name of this metadata source"""
        return "variable_attributes"
