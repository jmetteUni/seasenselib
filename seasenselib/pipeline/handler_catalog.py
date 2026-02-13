"""
Catalog of built-in pipeline handlers and plugin handler discovery.

Note
----
This module exists to keep handler listing fast and side-effect free.
It avoids importing many handler modules just to build CLI lists.
If full dynamic discovery of built-ins is preferred in the future,
this catalog can be replaced with a discovery-based implementation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Type

from .handler_registry import (
    HandlerRegistry,
    HANDLER_GROUP_MAPPING,
    HANDLER_GROUP_DERIVATIONS,
    HANDLER_GROUP_METADATA_EXTRACTORS,
    HANDLER_GROUP_CONVENTIONS,
    HANDLER_GROUP_VALIDATORS,
)
from .interfaces import (
    IMappingStrategy,
    IDerivation,
    IMetadataExtractor,
    IConvention,
    IValidator,
)


# Built-in handler definitions (stage -> handler name -> class path)
BUILTIN_HANDLERS: Dict[str, Dict[str, str]] = {
    "mapping": {
        "user_mapping": "seasenselib.pipeline.mapping.handlers.user_mapping_strategy.UserMappingStrategy",
        "user_config_mapping": "seasenselib.pipeline.mapping.handlers.user_mapping_strategy.UserMappingStrategy",
        "format_specific": "seasenselib.pipeline.mapping.handlers.dict_mapping_strategy.DictMappingStrategy",
        "reader_mapping": "seasenselib.pipeline.mapping.handlers.dict_mapping_strategy.DictMappingStrategy",
        "default_mapping": "seasenselib.pipeline.mapping.handlers.dict_mapping_strategy.DictMappingStrategy",
        "regex_mapping": "seasenselib.pipeline.mapping.handlers.regex_mapping_strategy.RegexMappingStrategy",
    },
    "derivation": {
        "density": "seasenselib.pipeline.derivation.handlers.density_derivation.DensityDerivation",
        "depth": "seasenselib.pipeline.derivation.handlers.depth_derivation.DepthDerivation",
        "potential_temperature": "seasenselib.pipeline.derivation.handlers.potential_temperature_derivation.PotentialTemperatureDerivation",
        "conservative_temperature": "seasenselib.pipeline.derivation.handlers.conservative_temperature_derivation.ConservativeTemperatureDerivation",
        "absolute_salinity": "seasenselib.pipeline.derivation.handlers.absolute_salinity_derivation.AbsoluteSalinityDerivation",
        "sound_speed": "seasenselib.pipeline.derivation.handlers.sound_speed_derivation.SoundSpeedDerivation",
    },
    "metadata_extraction": {
        "attributes": "seasenselib.pipeline.metadata_extraction.handlers.attribute_extractor.AttributeMetadataExtractor",
        "global_attributes": "seasenselib.pipeline.metadata_extraction.handlers.global_attribute_extractor.GlobalAttributeMetadataExtractor",
    },
    "unit_handling": {
        "normalize": "seasenselib.pipeline.unit_handling.handlers.unit_normalizer.UnitNormalizer",
        "convert": "seasenselib.pipeline.unit_handling.handlers.unit_converter.UnitConverter",
    },
    "metadata_enrichment": {
        "cf": "seasenselib.pipeline.metadata_enrichment.handlers.cf_convention.CFConvention",
        "acdd": "seasenselib.pipeline.metadata_enrichment.handlers.acdd_convention.ACDDConvention",
        "acdd_auto": "seasenselib.pipeline.metadata_enrichment.handlers.acdd_auto_metadata.AcddAutoMetadata",
        "whp": "seasenselib.pipeline.metadata_enrichment.handlers.whp_parameters.WHPParameters",
        "user_metadata": "seasenselib.pipeline.metadata_enrichment.handlers.user_metadata_handler.UserMetadataHandler",
    },
    "validation": {
        "cf": "seasenselib.pipeline.validation.handlers.cf_validator.CFValidator",
        "unit": "seasenselib.pipeline.validation.handlers.unit_validator.UnitValidator",
    },
    "finalization": {
        "raw_metadata": "seasenselib.pipeline.finalization.handlers.raw_metadata.RawMetadata",
        "processor_metadata": "seasenselib.pipeline.finalization.handlers.processor_metadata.ProcessorMetadata",
        "global_attributes": "seasenselib.pipeline.finalization.handlers.global_attributes.GlobalAttributes",
        "sorting": "seasenselib.pipeline.finalization.handlers.sorting.Sorting",
    },
}


PLUGIN_GROUPS: Dict[str, Tuple[str, Type]] = {
    "mapping": (HANDLER_GROUP_MAPPING, IMappingStrategy),
    "derivation": (HANDLER_GROUP_DERIVATIONS, IDerivation),
    "metadata_extraction": (HANDLER_GROUP_METADATA_EXTRACTORS, IMetadataExtractor),
    "metadata_enrichment": (HANDLER_GROUP_CONVENTIONS, IConvention),
    "validation": (HANDLER_GROUP_VALIDATORS, IValidator),
}


def _class_name_from_path(path: str) -> str:
    if not path:
        return ""
    return path.rsplit(".", 1)[-1]


def list_builtin_handlers() -> List[dict]:
    """Return built-in handler entries."""
    data: List[dict] = []
    for stage, handlers in BUILTIN_HANDLERS.items():
        for name, class_path in handlers.items():
            data.append({
                "stage": stage,
                "name": name,
                "class": _class_name_from_path(class_path),
                "class_path": class_path,
                "is_plugin": False,
            })
    return data


def list_plugin_handlers() -> List[dict]:
    """Return plugin handler entries discovered via entry points."""
    data: List[dict] = []
    for stage, (group, base_class) in PLUGIN_GROUPS.items():
        try:
            mapping = HandlerRegistry.get(group, base_class)
        except Exception:
            mapping = {}
        for name, cls in mapping.items():
            class_path = f"{cls.__module__}.{cls.__name__}"
            data.append({
                "stage": stage,
                "name": name,
                "class": cls.__name__,
                "class_path": class_path,
                "is_plugin": True,
            })
    return data


def list_handlers(include_plugins: bool = True) -> List[dict]:
    """Return all handler entries (built-in plus plugins)."""
    data = list_builtin_handlers()
    if include_plugins:
        data.extend(list_plugin_handlers())
    return data


__all__ = [
    "BUILTIN_HANDLERS",
    "PLUGIN_GROUPS",
    "list_builtin_handlers",
    "list_plugin_handlers",
    "list_handlers",
]
