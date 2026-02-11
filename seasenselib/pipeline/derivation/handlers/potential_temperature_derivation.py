"""
Potential Temperature Derivation

Calculates potential temperature from in-situ temperature, salinity, and pressure using GSW.
"""

from typing import List, Dict, Tuple
import xarray as xr
import numpy as np
from ...interfaces import IDerivation
import seasenselib.parameters as params
from .utils import list_variants, output_name_from_input, units_ok

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


class PotentialTemperatureDerivation(IDerivation):
    """
    Derives potential temperature using the TEOS-10 Gibbs SeaWater (GSW) library.
    
    Potential temperature is the temperature a parcel of water would have if moved
    adiabatically to a reference pressure (typically 0 dbar = sea surface).
    
    Requires:
        - temperature (°C, ITS-90, in-situ)
        - salinity (PSU or g/kg)
        - pressure (dbar)
    
    Produces:
        - potential_temperature (°C, referenced to 0 dbar)
    
    Uses gsw.pt0_from_t() for calculation following TEOS-10 standard.
    """
    
    @staticmethod
    def output_parameter() -> str:
        """Return name of derived parameter"""
        return params.POTENTIAL_TEMPERATURE
    
    @staticmethod
    def required_inputs() -> List[str]:
        """Return list of required input parameters"""
        return [params.TEMPERATURE, params.SALINITY, params.PRESSURE]
    
    def can_derive(self, dataset: xr.Dataset) -> bool:
        """
        Check if dataset contains all required inputs for potential temperature derivation.
        
        Args:
            dataset: xarray Dataset to check
        
        Returns:
            True if all required parameters are present
        """
        if _get_gsw() is None:
            return False

        temps = list_variants(dataset, params.TEMPERATURE)
        sals = list_variants(dataset, params.SALINITY)
        press = list_variants(dataset, params.PRESSURE)
        return bool(temps and sals and press)
    
    def derive(self, dataset: xr.Dataset) -> Tuple[Dict[str, xr.DataArray], List[str]]:
        """
        Derive potential temperature from in-situ temperature, salinity, and pressure.
        
        Args:
            dataset: xarray Dataset with required parameters
        
        Returns:
            xarray DataArray containing derived potential temperature values
        
        Raises:
            ImportError: If GSW library is not installed
            KeyError: If required parameters are missing
        """
        gsw = _get_gsw()
        if gsw is None:
            raise ImportError(
                "GSW library is required for potential temperature derivation. "
                "Install with: pip install gsw"
            )
        
        warnings: List[str] = []
        outputs: Dict[str, xr.DataArray] = {}

        temps = list_variants(dataset, params.TEMPERATURE)
        sals = list_variants(dataset, params.SALINITY)
        press = list_variants(dataset, params.PRESSURE)

        if not temps or not sals or not press:
            return outputs, warnings

        if len(sals) > 1:
            warnings.append(
                "Potential temperature not derived: multiple salinity variables present."
            )
            return outputs, warnings
        if len(press) > 1:
            warnings.append(
                "Potential temperature not derived: multiple pressure variables present."
            )
            return outputs, warnings

        sal_name = sals[0]
        pres_name = press[0]

        if not units_ok(dataset, sal_name, params.SALINITY):
            warnings.append(
                f"Potential temperature not derived: salinity units not supported for '{sal_name}'."
            )
            return outputs, warnings
        if not units_ok(dataset, pres_name, params.PRESSURE):
            warnings.append(
                f"Potential temperature not derived: pressure units not supported for '{pres_name}'."
            )
            return outputs, warnings
        
        # Calculate potential temperature using GSW (TEOS-10)
        # gsw.pt0_from_t() calculates potential temperature referenced to 0 dbar
        for temp_name in temps:
            if not units_ok(dataset, temp_name, params.TEMPERATURE):
                warnings.append(
                    f"Potential temperature skipped for '{temp_name}': unsupported temperature units."
                )
                continue

            temp = dataset[temp_name].values
            sal = dataset[sal_name].values
            pres = dataset[pres_name].values

            pot_temp = gsw.pt0_from_t(sal, temp, pres)

            output_name = output_name_from_input(
                params.POTENTIAL_TEMPERATURE,
                temp_name,
                params.TEMPERATURE,
            )
            pot_temp_da = xr.DataArray(
                pot_temp,
                dims=dataset[temp_name].dims,
                coords=dataset[temp_name].coords,
                attrs={
                    **self.metadata(),
                    "derivation": (
                        f"gsw.pt0_from_t({sal_name}, {temp_name}, {pres_name})"
                    ),
                },
            )
            outputs[output_name] = pot_temp_da

        return outputs, warnings
    
    def metadata(self) -> dict:
        """Return metadata for the derived parameter."""
        from seasenselib.knowledge import load_json
        data = load_json("pipeline/derivation/derivations.json")
        if not isinstance(data, dict) or 'potential_temperature' not in data:
            raise RuntimeError(
                "Missing derivation metadata for 'potential_temperature' in pipeline/derivation/derivations.json"
            )
        return data['potential_temperature']
    
