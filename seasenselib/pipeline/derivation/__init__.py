"""
Derivations Module

Provides parameter derivation implementations and stage orchestration.

Available Derivations:
    - DensityDerivation: Calculate density from T, S, P
    - DepthDerivation: Calculate depth from pressure and latitude
    - PotentialTemperatureDerivation: Calculate potential temperature from T, S, P
    - SoundSpeedDerivation: Calculate sound speed from T, S, P

Stage:
    - DerivationStage: Automatic parameter derivation with dependency resolution
    - DerivationRunner: Composite derivation runner
"""

from .handlers.density_derivation import DensityDerivation
from .handlers.depth_derivation import DepthDerivation
from .handlers.potential_temperature_derivation import PotentialTemperatureDerivation
from .handlers.conservative_temperature_derivation import ConservativeTemperatureDerivation
from .handlers.sound_speed_derivation import SoundSpeedDerivation
from .handlers.derivation_runner import DerivationRunner
from .stage import DerivationStage

__all__ = [
    'DensityDerivation',
    'DepthDerivation',
    'PotentialTemperatureDerivation',
    'ConservativeTemperatureDerivation',
    'SoundSpeedDerivation',
    'DerivationRunner',
    'DerivationStage',
]
