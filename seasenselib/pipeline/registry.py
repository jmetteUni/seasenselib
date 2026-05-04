"""
Stage registry for discovering and managing stages.

This module provides automatic discovery of stages via entry points.
"""

from __future__ import annotations
from typing import Dict, List, Type, Optional
import logging

from .base import Stage

logger = logging.getLogger(__name__)


class StageRegistry:
    """
    Registry for discovering and managing available stages.

    Stages are discovered via Python entry points in the 'seasenselib.pipeline' group.
    This allows third-party packages to register their own stages.
    """

    _instance: Optional['StageRegistry'] = None
    DEFAULT_STAGE_NAMES = [
        "mapping",
        "unit_handling",
        "derivation",
        "metadata_extraction",
        "metadata_enrichment",
        "validation",
        "finalization",
    ]

    def __init__(self):
        self._stages: Dict[str, Type[Stage]] = {}
        self._builtin_stages: Dict[str, Type[Stage]] = {}
        self._discover_stages()

    @classmethod
    def get_instance(cls) -> 'StageRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
    
    @classmethod
    def default_stage_names(cls) -> List[str]:
        """Return the default stage names (without forcing discovery)."""
        return list(cls.DEFAULT_STAGE_NAMES)

    def _discover_stages(self) -> None:
        self._discover_builtin_stages()

        try:
            try:
                from importlib.metadata import entry_points
            except ImportError:
                from importlib_metadata import entry_points

            eps = entry_points()
            if hasattr(eps, "select"):
                discovered = eps.select(group='seasenselib.pipeline')
            elif isinstance(eps, dict):
                discovered = eps.get('seasenselib.pipeline', [])
            else:
                discovered = [ep for ep in eps if getattr(ep, "group", None) == 'seasenselib.pipeline']

            for ep in discovered:
                try:
                    stage_class = ep.load()

                    if not issubclass(stage_class, Stage):
                        logger.warning(
                            f"Entry point '{ep.name}' is not a Stage subclass. Skipping."
                        )
                        continue

                    self._stages[ep.name] = stage_class
                    logger.debug(f"Discovered stage '{ep.name}' from entry point")

                except Exception as e:
                    logger.warning(f"Failed to load stage from entry point '{ep.name}': {e}")

        except Exception as e:
            logger.warning(f"Failed to discover stages via entry points: {e}")

    def _discover_builtin_stages(self) -> None:
        try:
            from .mapping.stage import MappingStage
            from .derivation.stage import DerivationStage
            from .metadata_extraction.stage import MetadataExtractionStage
            from .unit_handling.stage import UnitHandlingStage
            from .metadata_enrichment.stage import MetadataEnrichmentStage
            from .validation.stage import ValidationStage
            from .finalization.stage import FinalizationStage

            builtin = [
                MappingStage,
                DerivationStage,
                MetadataExtractionStage,
                UnitHandlingStage,
                MetadataEnrichmentStage,
                ValidationStage,
                FinalizationStage,
            ]

            for stage_class in builtin:
                instance = stage_class()
                name = instance.name()
                self._builtin_stages[name] = stage_class
                self._stages[name] = stage_class
                logger.debug(f"Registered built-in stage '{name}'")

        except ImportError as e:
            logger.warning(f"Failed to import built-in stages: {e}")

    def register(self, name: str, stage_class: Type[Stage]) -> None:
        if name in self._stages:
            raise ValueError(f"Stage '{name}' is already registered")

        if not issubclass(stage_class, Stage):
            raise ValueError("stage_class must be a Stage subclass")

        self._stages[name] = stage_class
        logger.debug(f"Manually registered stage '{name}'")

    def get_stage(self, name: str, **kwargs) -> Stage:
        if name not in self._stages:
            raise ValueError(
                f"Unknown stage '{name}'. Available stages: {', '.join(self.list_stages())}"
            )

        stage_class = self._stages[name]
        return stage_class(**kwargs)

    def get_stage_class(self, name: str) -> Type[Stage]:
        """Return the stage class without instantiating it."""
        if name not in self._stages:
            raise ValueError(
                f"Unknown stage '{name}'. Available stages: {', '.join(self.list_stages())}"
            )
        return self._stages[name]

    def list_stages(self) -> List[str]:
        return sorted(self._stages.keys())

    def list_builtin_stages(self) -> List[str]:
        return sorted(self._builtin_stages.keys())

    def list_plugin_stages(self) -> List[str]:
        return sorted(set(self._stages.keys()) - set(self._builtin_stages.keys()))

    def has_stage(self, name: str) -> bool:
        return name in self._stages

    def describe(self) -> str:
        lines = [f"Built-in stages ({len(self._builtin_stages)}):"]
        for name in self.list_builtin_stages():
            stage = self.get_stage(name)
            lines.append(f"  - {name} ({stage.__class__.__name__})")

        plugin_stages = self.list_plugin_stages()
        if plugin_stages:
            lines.append(f"Plugin stages ({len(plugin_stages)}):")
            for name in plugin_stages:
                stage = self.get_stage(name)
                lines.append(f"  - {name} ({stage.__class__.__name__})")

        return "\n".join(lines)


__all__ = ["StageRegistry"]
