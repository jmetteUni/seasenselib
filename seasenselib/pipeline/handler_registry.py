"""
Handler registry for pipeline plugins.

Provides discovery and caching for handler entry points.
"""

from __future__ import annotations

from typing import Dict, Type, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

HANDLER_GROUP_MAPPING = "seasenselib.pipeline.mapping_strategies"
HANDLER_GROUP_DERIVATIONS = "seasenselib.pipeline.derivations"
HANDLER_GROUP_METADATA_EXTRACTORS = "seasenselib.pipeline.metadata_extractors"
HANDLER_GROUP_CONVENTIONS = "seasenselib.pipeline.conventions"
HANDLER_GROUP_TRANSFORMATIONS = "seasenselib.pipeline.transformations"
HANDLER_GROUP_VALIDATORS = "seasenselib.pipeline.validators"


class HandlerRegistry:
    """Discover and cache handler plugins by entry-point group."""

    _cache: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def get(cls, group: str, base_class: Type[T]) -> Dict[str, Type[T]]:
        """Return mapping of handler name -> class for a given entry-point group."""
        if group in cls._cache:
            return dict(cls._cache[group])

        discovered: Dict[str, Type[T]] = {}

        try:
            try:
                from importlib.metadata import entry_points
            except ImportError:
                from importlib_metadata import entry_points  # Python < 3.8

            eps = entry_points()
            if hasattr(eps, "select"):
                group_eps = eps.select(group=group)
            elif isinstance(eps, dict):
                group_eps = eps.get(group, [])
            else:
                group_eps = getattr(eps, group, [])

            for ep in group_eps:
                try:
                    cls_obj = ep.load()
                    if not isinstance(cls_obj, type) or not issubclass(cls_obj, base_class):
                        logger.warning(
                            "Entry point '%s' is not a %s subclass. Skipping.",
                            ep.name,
                            base_class.__name__,
                        )
                        continue
                    if ep.name in discovered:
                        logger.warning(
                            "Duplicate handler entry point '%s' for group '%s'. Skipping.",
                            ep.name,
                            group,
                        )
                        continue
                    discovered[ep.name] = cls_obj
                    logger.debug("Discovered handler '%s' in group '%s'", ep.name, group)
                except Exception as exc:
                    logger.warning(
                        "Failed to load handler from entry point '%s': %s",
                        ep.name,
                        exc,
                    )
        except Exception as exc:
            logger.warning("Failed to discover handlers for group '%s': %s", group, exc)

        cls._cache[group] = dict(discovered)
        return dict(discovered)


__all__ = [
    "HandlerRegistry",
    "HANDLER_GROUP_MAPPING",
    "HANDLER_GROUP_DERIVATIONS",
    "HANDLER_GROUP_METADATA_EXTRACTORS",
    "HANDLER_GROUP_CONVENTIONS",
    "HANDLER_GROUP_TRANSFORMATIONS",
    "HANDLER_GROUP_VALIDATORS",
]
