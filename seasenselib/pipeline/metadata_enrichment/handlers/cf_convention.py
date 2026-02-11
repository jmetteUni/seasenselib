"""
CF Conventions implementation.

Provides CF-compliant metadata enrichment and validation.
"""

from __future__ import annotations
from typing import List, Dict
import logging
import re

import xarray as xr
import seasenselib.parameters as params
from ...interfaces import IConvention, MetadataRegistry, ValidationError

logger = logging.getLogger(__name__)


class CFConvention(IConvention):
    """
    CF Conventions implementation.
    
    Enriches datasets with CF-compliant metadata including:
    - standard_name (from CF standard name table)
    - long_name, units
    - coverage_content_type
    - Coordinate attributes (axis, positive)
    """
    
    def __init__(self, version: str = "1.13"):
        """
        Initialize CF convention.
        
        Parameters
        ----------
        version : str, default="1.13"
            CF Conventions version.
        """
        self.version = version
    
    def name(self) -> str:
        return f"CF-{self.version}"
    
    def enrich(self, dataset: xr.Dataset, metadata_registry: MetadataRegistry) -> xr.Dataset:
        """Enrich dataset with CF metadata."""
        variables_enriched = 0
        coords_enriched = 0
        # Enrich data variables
        for var_name in dataset.data_vars:
            added = self._enrich_variable(dataset[var_name], var_name)
            if added:
                variables_enriched += 1
                logger.debug("CF added %s to variable '%s'", ", ".join(added), var_name)
        
        # Enrich coordinates
        for coord_name in dataset.coords:
            added = self._enrich_coordinate(dataset[coord_name], coord_name)
            if added:
                coords_enriched += 1
                logger.debug("CF added %s to coordinate '%s'", ", ".join(added), coord_name)
        
        # Add global Conventions attribute
        conventions = dataset.attrs.get('Conventions', '')
        cf_string = f"CF-{self.version}"
        if cf_string not in conventions:
            if conventions:
                dataset.attrs['Conventions'] = f"{conventions}, {cf_string}"
            else:
                dataset.attrs['Conventions'] = cf_string
            logger.debug("CF added global attribute 'Conventions': %s", dataset.attrs['Conventions'])
        
        logger.info(f"Applied CF-{self.version} conventions")
        if variables_enriched or coords_enriched:
            logger.debug(
                "CF enriched %d variables and %d coordinates",
                variables_enriched,
                coords_enriched
            )
        return dataset
    
    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        """Validate CF compliance."""
        errors = []
        
        # Check for required global attributes
        if 'Conventions' not in dataset.attrs:
            errors.append(ValidationError(
                "Missing required global attribute: Conventions",
                severity="error"
            ))
        
        # Check data variables
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            
            # Check for units (required for most variables)
            if 'units' not in var.attrs and var_name not in [params.TIME]:
                errors.append(ValidationError(
                    f"Missing units attribute",
                    severity="warning",
                    path=var_name
                ))
            
            # Check for standard_name (recommended)
            if 'standard_name' not in var.attrs:
                errors.append(ValidationError(
                    f"Missing standard_name attribute",
                    severity="info",
                    path=var_name
                ))
        
        return errors
    
    def _enrich_variable(self, var: xr.DataArray, name: str) -> List[str]:
        """Add CF metadata to a variable."""
        added: List[str] = []
        # Get metadata from parameters.py if available
        base_name = self._base_name(name)
        if base_name in params.metadata:
            meta = params.metadata[base_name]
            
            # Add standard_name
            if 'standard_name' in meta and 'standard_name' not in var.attrs:
                var.attrs['standard_name'] = meta['standard_name']
                added.append('standard_name')
            
            # Add long_name
            if 'long_name' in meta and 'long_name' not in var.attrs:
                var.attrs['long_name'] = meta['long_name']
                added.append('long_name')
            
            # Add coverage_content_type
            if 'coverage_content_type' in meta and 'coverage_content_type' not in var.attrs:
                var.attrs['coverage_content_type'] = meta['coverage_content_type']
                added.append('coverage_content_type')
            
            # Add units if not present but available in metadata
            if 'units' in meta and 'units' not in var.attrs:
                var.attrs['units'] = meta['units']
                added.append('units')
            
            # Add measurement_type if available
            if 'measurement_type' in meta and 'measurement_type' not in var.attrs:
                var.attrs['measurement_type'] = meta['measurement_type']
                added.append('measurement_type')
        return added

    @staticmethod
    def _base_name(name: str) -> str:
        match = re.match(r"^([a-zA-Z0-9_]+?)(?:_\d{1,2})?$", name)
        return match.group(1) if match else name
    
    def _enrich_coordinate(self, coord: xr.DataArray, name: str) -> List[str]:
        """Add CF metadata to a coordinate."""
        added: List[str] = []
        # Prefer knowledge base defaults
        defaults = self._coordinate_defaults()
        if name in defaults:
            for key, value in defaults[name].items():
                if key not in coord.attrs:
                    coord.attrs[key] = value
                    added.append(key)
            return added

        # Fallback to built-in defaults
        if name == params.TIME:
            if 'standard_name' not in coord.attrs:
                coord.attrs['standard_name'] = 'time'
                added.append('standard_name')
            if 'long_name' not in coord.attrs:
                coord.attrs['long_name'] = 'Time'
                added.append('long_name')
            if 'axis' not in coord.attrs:
                coord.attrs['axis'] = 'T'
                added.append('axis')
        elif name == params.DEPTH:
            if 'standard_name' not in coord.attrs:
                coord.attrs['standard_name'] = 'depth'
                added.append('standard_name')
            if 'long_name' not in coord.attrs:
                coord.attrs['long_name'] = 'Depth'
                added.append('long_name')
            if 'axis' not in coord.attrs:
                coord.attrs['axis'] = 'Z'
                added.append('axis')
            if 'positive' not in coord.attrs:
                coord.attrs['positive'] = 'down'
                added.append('positive')
        elif name == params.LATITUDE:
            if 'standard_name' not in coord.attrs:
                coord.attrs['standard_name'] = 'latitude'
                added.append('standard_name')
            if 'long_name' not in coord.attrs:
                coord.attrs['long_name'] = 'Latitude'
                added.append('long_name')
            if 'axis' not in coord.attrs:
                coord.attrs['axis'] = 'Y'
                added.append('axis')
            if 'units' not in coord.attrs:
                coord.attrs['units'] = 'degrees_north'
                added.append('units')
        elif name == params.LONGITUDE:
            if 'standard_name' not in coord.attrs:
                coord.attrs['standard_name'] = 'longitude'
                added.append('standard_name')
            if 'long_name' not in coord.attrs:
                coord.attrs['long_name'] = 'Longitude'
                added.append('long_name')
            if 'axis' not in coord.attrs:
                coord.attrs['axis'] = 'X'
                added.append('axis')
            if 'units' not in coord.attrs:
                coord.attrs['units'] = 'degrees_east'
                added.append('units')

        return added

    def _coordinate_defaults(self) -> Dict[str, Dict[str, str]]:
        try:
            from seasenselib.knowledge import load_json
            data = load_json("pipeline/metadata_enrichment/cf_coordinate_defaults.json")
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}


__all__ = ["CFConvention"]
