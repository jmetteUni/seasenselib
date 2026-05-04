"""
CF Conventions validator.

Validates dataset compliance with CF Conventions.
"""

from __future__ import annotations
from typing import List

import xarray as xr
import seasenselib.parameters as params

from ...interfaces import IValidator, ValidationError

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
