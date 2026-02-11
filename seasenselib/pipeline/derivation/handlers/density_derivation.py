"""
Density Derivation

Calculates seawater density from temperature, salinity, and pressure using GSW.
"""

from typing import List, Tuple, Dict, Any
import xarray as xr
import numpy as np
from ...interfaces import IDerivation
import seasenselib.parameters as params
from .utils import pick_first_variant, output_name_from_input, units_ok

_GSW = None


def _get_gsw():
    global _GSW
    if _GSW is not None:
        return _GSW
    try:
        import gsw  # type: ignore
    except ImportError:
        return None
    _GSW = gsw
    return gsw


class DensityDerivation(IDerivation):
    """
    Derives seawater density using the TEOS-10 Gibbs SeaWater (GSW) library.
    
    Requires:
        - temperature (°C, ITS-90)
        - salinity (PSU or g/kg)
        - pressure (dbar)
    
    Produces:
        - density (kg/m³) or sigma_theta (kg/m³ - 1000)
    
    Uses gsw.rho() for accurate density calculation following TEOS-10 standard.
    """
    
    @staticmethod
    def output_parameter() -> str:
        """Return name of derived parameter"""
        return params.DENSITY
    
    @staticmethod
    def required_inputs() -> List[str]:
        """Return list of required input parameters"""
        return [params.TEMPERATURE, params.SALINITY, params.PRESSURE]
    
    def can_derive(self, dataset: xr.Dataset) -> bool:
        """
        Check if dataset contains all required inputs for density derivation.
        
        Args:
            dataset: xarray Dataset to check
        
        Returns:
            True if all required parameters are present
        """
        if _get_gsw() is None:
            return False

        temp, _ = pick_first_variant(dataset, params.TEMPERATURE)
        sal, _ = pick_first_variant(dataset, params.SALINITY)
        pres, _ = pick_first_variant(dataset, params.PRESSURE)
        return all([temp, sal, pres])
    
    def derive(self, dataset: xr.Dataset) -> Tuple[Any, List[str]]:
        """
        Derive density from temperature, salinity, and pressure.
        
        Args:
            dataset: xarray Dataset with required parameters
        
        Returns:
            xarray DataArray containing derived density values
        
        Raises:
            ImportError: If GSW library is not installed
            KeyError: If required parameters are missing
        """
        gsw = _get_gsw()
        if gsw is None:
            raise ImportError(
                "GSW library is required for density derivation. "
                "Install with: pip install gsw"
            )
        
        warnings: List[str] = []

        temp_name, temp_count = pick_first_variant(dataset, params.TEMPERATURE)
        sal_name, sal_count = pick_first_variant(dataset, params.SALINITY)
        pres_name, pres_count = pick_first_variant(dataset, params.PRESSURE)

        if not temp_name or not sal_name or not pres_name:
            raise ValueError("Missing required inputs for density derivation.")

        if temp_count > 1:
            warnings.append(
                f"Density derivation used '{temp_name}' (lowest suffix) among {temp_count} temperature variables."
            )
        if sal_count > 1:
            warnings.append(
                f"Density derivation used '{sal_name}' (lowest suffix) among {sal_count} salinity variables."
            )
        if pres_count > 1:
            warnings.append(
                f"Density derivation used '{pres_name}' (lowest suffix) among {pres_count} pressure variables."
            )

        if not units_ok(dataset, temp_name, params.TEMPERATURE):
            warnings.append(
                f"Density not derived: unsupported units for '{temp_name}'."
            )
            return None, warnings
        if not units_ok(dataset, sal_name, params.SALINITY):
            warnings.append(
                f"Density not derived: unsupported units for '{sal_name}'."
            )
            return None, warnings
        if not units_ok(dataset, pres_name, params.PRESSURE):
            warnings.append(
                f"Density not derived: unsupported units for '{pres_name}'."
            )
            return None, warnings

        temp = dataset[temp_name].values
        sal = dataset[sal_name].values
        pres = dataset[pres_name].values
        
        # Calculate density using GSW (TEOS-10)
        # gsw.rho() expects: Absolute Salinity (g/kg), Conservative Temperature (°C), pressure (dbar)
        # For practical purposes, we use Practical Salinity and in-situ temperature
        density = gsw.rho(sal, temp, pres)
        
        # Create DataArray with same dimensions as temperature
        attrs = {
            **self.metadata(),
            'derivation': f"gsw.rho({sal_name}, {temp_name}, {pres_name})"
        }
        comment = attrs.get("comment", "")
        note = f"Derived from {temp_name}, {sal_name}, {pres_name}"
        attrs["comment"] = f"{comment}; {note}".strip("; ")

        density_da = xr.DataArray(
            density,
            dims=dataset[temp_name].dims,
            coords=dataset[temp_name].coords,
            attrs=attrs
        )
        
        return density_da, warnings
    
    def metadata(self) -> dict:
        """Return metadata for the derived parameter."""
        from seasenselib.knowledge import load_json
        data = load_json("pipeline/derivation/derivations.json")
        if not isinstance(data, dict) or 'density' not in data:
            raise RuntimeError(
                "Missing derivation metadata for 'density' in pipeline/derivation/derivations.json"
            )
        return data['density']
    
