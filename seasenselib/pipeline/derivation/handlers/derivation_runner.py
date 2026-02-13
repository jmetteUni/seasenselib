"""
Derivation runner.

Manages automatic derivation of oceanographic parameters with dependency resolution.
"""

import xarray as xr
from typing import List, Dict, Any, Optional, Tuple
import logging
from ...base import StageContext
from ...interfaces import IDerivation
from ...utils import record_handler_applied
from .utils import list_variants, canonical_unit, get_input_units

logger = logging.getLogger(__name__)
from .density_derivation import DensityDerivation
from .depth_derivation import DepthDerivation
from .potential_temperature_derivation import PotentialTemperatureDerivation
from .conservative_temperature_derivation import ConservativeTemperatureDerivation
from .absolute_salinity_derivation import AbsoluteSalinityDerivation
from .sound_speed_derivation import SoundSpeedDerivation


class DerivationRunner:
    """
    Automatically derives oceanographic parameters when required inputs are available.
    
    Uses Chain of Responsibility pattern to try each derivation in sequence.
    Only derives parameters that:
    1. Are not already in the dataset
    2. Have all required inputs available
    
    Built-in derivations:
        - density: From T, S, P
        - depth: From pressure and latitude
        - potential_temperature: From T, S, P
        - sound_speed: From T, S, P
    
    Usage:
        derivation = DerivationRunner()
        enriched_ds = derivation.process(context)
    """
    
    def __init__(
        self,
        derivations: Optional[List[IDerivation]] = None,
        unit_guard: bool = True,
        input_units: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize parameter derivation logic.
        
        Args:
            derivations: Optional list of IDerivation implementations.
                        If None, uses default derivations (density, pot_temp, sound_speed)
        """
        # Use default derivations if none provided
        if derivations is None:
            derivations = [
                DensityDerivation(),
                DepthDerivation(),
                PotentialTemperatureDerivation(),
                ConservativeTemperatureDerivation(),
                AbsoluteSalinityDerivation(),
                SoundSpeedDerivation()
            ]
        
        self.derivations = derivations
        self.unit_guard = bool(unit_guard)
        self._input_units = self._load_input_units(input_units)
    
    def process(self, context: StageContext) -> StageContext:
        """
        Process dataset by deriving all possible parameters.
        
        Args:
            context: StageContext with dataset and metadata
        
        Returns:
            Updated StageContext with derived parameters
        """
        result = context.dataset.copy()
        derived_count = 0
        
        derived_params: List[str] = []
        for derivation in self.derivations:
            param_name = derivation.output_parameter()
            
            # Skip if parameter already exists
            if param_name in result.data_vars:
                # Use logger if available, otherwise skip silently
                logger.debug(f"{param_name} already exists, skipping derivation")
                continue
            
            if self.unit_guard:
                ok, reason = self._inputs_units_ok(result, derivation.required_inputs())
                if not ok:
                    warning = f"Skipped derivation '{param_name}': {reason}"
                    context.metadata.setdefault("warnings", []).append(warning)
                    logger.warning(warning)
                    continue

            # Check if we can derive it
            if derivation.can_derive(result):
                record_handler_applied(context.metadata, "derivation", param_name)
                try:
                    output = derivation.derive(result)
                    output, warnings = self._unwrap_output(output)
                    for warning in warnings:
                        context.metadata.setdefault("warnings", []).append(warning)
                        logger.warning(warning)

                    if output is None:
                        continue
                    if isinstance(output, dict):
                        for name, array in output.items():
                            result[name] = array
                            derived_params.append(name)
                            derived_count += 1
                            logger.debug("Derived %s", name)
                    elif isinstance(output, xr.DataArray):
                        result[param_name] = output
                        derived_params.append(param_name)
                        derived_count += 1
                        logger.debug("Derived %s", param_name)
                except Exception as e:
                    warning = f"Failed to derive {param_name}: {e}"
                    logger.warning(warning)
                    context.metadata.setdefault("warnings", []).append(warning)
            else:
                required = derivation.required_inputs()
                missing = [r for r in required if r not in result.data_vars]
                logger.debug(
                    f"Cannot derive {param_name}, missing inputs: {missing}"
                )
        
        if derived_params:
            context.metadata.setdefault("derived_parameters", []).extend(derived_params)
        if derived_count > 0:
            logger.info(f"Derived {derived_count} parameter(s)")
        
        # Return updated context
        return StageContext(dataset=result, metadata=context.metadata)
    
    def register_derivation(self, derivation: IDerivation):
        """
        Register a new derivation.
        
        Args:
            derivation: IDerivation implementation to register
        """
        self.derivations.append(derivation)
        logger.debug(f"Registered derivation: {derivation.output_parameter()}")
    
    def list_derivations(self) -> List[str]:
        """Return list of available derivation names"""
        return [d.output_parameter() for d in self.derivations]

    @staticmethod
    def _load_input_units(
        overrides: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, List[str]]:
        if overrides is not None:
            return {
                str(k): [str(v).lower() for v in values]
                for k, values in overrides.items()
                if isinstance(values, list)
            }
        return get_input_units()

    def _inputs_units_ok(
        self,
        dataset: xr.Dataset,
        required: List[str],
    ) -> tuple[bool, str]:
        for param in required:
            allowed = self._input_units.get(param)
            candidates = list_variants(dataset, param)
            if not candidates:
                return False, f"missing input '{param}'"
            if not allowed:
                continue
            found_units = set()
            ok = False
            for name in candidates:
                unit = dataset[name].attrs.get("units")
                if not unit:
                    continue
                canonical = canonical_unit(unit)
                found_units.add(canonical)
                if canonical in allowed:
                    ok = True
            if not ok:
                if not found_units:
                    return False, f"missing units for '{param}'"
                return False, (
                    f"units for '{param}' are {sorted(found_units)}, expected one of {sorted(allowed)}"
                )
        return True, ""

    @staticmethod
    def _unwrap_output(output: Any) -> Tuple[Any, List[str]]:
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], list):
            return output[0], output[1]
        return output, []
