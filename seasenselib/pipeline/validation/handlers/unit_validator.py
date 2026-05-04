"""
Unit validator.

Validates units are present and CF-compliant.
"""

from __future__ import annotations
from typing import List
import logging

import xarray as xr

from ...interfaces import IValidator, ValidationError

logger = logging.getLogger(__name__)


class UnitValidator(IValidator):
    """
    Validates units are present and CF-compliant.
    """
    
    def name(self) -> str:
        return "unit"
    
    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        """Validate units."""
        errors = []
        
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            
            # Check if units attribute exists
            if 'units' not in var.attrs:
                errors.append(ValidationError(
                    f"Missing units attribute",
                    severity="warning",
                    path=var_name
                ))
                continue
            
            # Check for common unit problems
            units = var.attrs['units']
            
            # Check for deprecated units
            if units in ['degrees C', 'deg C']:
                errors.append(ValidationError(
                    f"Non-standard temperature unit '{units}'. Use 'degC'",
                    severity="info",
                    path=var_name
                ))
            
            if units in ['PSU', 'psu']:
                errors.append(ValidationError(
                    f"Deprecated salinity unit '{units}'. Salinity should be dimensionless (use '1')",
                    severity="info",
                    path=var_name
                ))
        
        return errors
