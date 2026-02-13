"""
Derivation stage.

Derives oceanographic parameters when inputs are available.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

from ..base import Stage, StageContext
from .handlers.derivation_runner import DerivationRunner
from ..interfaces import IDerivation
from ..handler_registry import HandlerRegistry, HANDLER_GROUP_DERIVATIONS


class DerivationStage(Stage):
    """Stage 2: Parameter derivation."""

    def __init__(self, derivations: Optional[List[IDerivation]] = None):
        self._derivation = DerivationRunner(derivations=derivations)

    def name(self) -> str:
        return "derivation"

    def configure(self, config: Dict[str, Any]) -> None:
        unit_guard = bool(config.get('unit_guard', True))
        input_units = config.get('input_units')

        handlers = config.get('handlers')
        if not isinstance(handlers, list) or not handlers:
            self._derivation = DerivationRunner(
                derivations=self._derivation.derivations,
                unit_guard=unit_guard,
                input_units=input_units,
            )
            return

        # Rebuild derivation list based on handler names
        from .handlers.density_derivation import DensityDerivation
        from .handlers.depth_derivation import DepthDerivation
        from .handlers.potential_temperature_derivation import PotentialTemperatureDerivation
        from .handlers.conservative_temperature_derivation import ConservativeTemperatureDerivation
        from .handlers.absolute_salinity_derivation import AbsoluteSalinityDerivation
        from .handlers.sound_speed_derivation import SoundSpeedDerivation

        mapping = {
            'density': DensityDerivation,
            'depth': DepthDerivation,
            'potential_temperature': PotentialTemperatureDerivation,
            'conservative_temperature': ConservativeTemperatureDerivation,
            'absolute_salinity': AbsoluteSalinityDerivation,
            'sound_speed': SoundSpeedDerivation,
        }
        if any(name not in mapping for name in handlers):
            plugin_mapping = HandlerRegistry.get(HANDLER_GROUP_DERIVATIONS, IDerivation)
            for name, cls in plugin_mapping.items():
                if name not in mapping:
                    mapping[name] = cls

        depth_config = config.get('depth', {})
        if not isinstance(depth_config, dict):
            depth_config = {}
        depth_defaults = {
            'use_default_latitude': False,
            'default_latitude': 45.0,
        }
        depth_defaults.update(depth_config)

        derivations: List[IDerivation] = []
        for name in handlers:
            if name == 'depth':
                derivations.append(DepthDerivation(
                    use_default_latitude=bool(depth_defaults.get('use_default_latitude', False)),
                    default_latitude=depth_defaults.get('default_latitude', 45.0),
                ))
                continue
            cls = mapping.get(name)
            if cls is not None:
                try:
                    derivations.append(cls())
                except Exception:
                    continue

        if derivations:
            self._derivation = DerivationRunner(
                derivations=derivations,
                unit_guard=unit_guard,
                input_units=input_units,
            )

    def process(self, context: StageContext) -> StageContext:
        return self._derivation.process(context)
