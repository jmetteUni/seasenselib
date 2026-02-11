"""
CF Conventions validator.

Validates dataset compliance with CF Conventions.
"""

from __future__ import annotations
from typing import List
import logging

import xarray as xr
import seasenselib.parameters as params

from ...interfaces import IValidator, ValidationError

logger = logging.getLogger(__name__)


class CFValidator(IValidator):
    """
    Validates CF Conventions compliance.
    """
    
    def name(self) -> str:
        return "cf"
    
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
        
        # Check coordinates
        for coord_name in dataset.coords:
            coord = dataset[coord_name]
            
            # Time coordinate should have units
            if coord_name == params.TIME and 'units' not in coord.attrs:
                errors.append(ValidationError(
                    "Time coordinate missing units attribute",
                    severity="error",
                    path=coord_name
                ))
        
        return errors


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
            if units in ['degrees C', 'deg C', 'degC']:
                errors.append(ValidationError(
                    f"Non-standard temperature unit '{units}'. Use 'ITS-90, deg C'",
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
