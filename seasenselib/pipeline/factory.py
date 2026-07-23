"""
Factory functions for creating pipelines.

This module provides convenience functions for creating common pipeline configurations.
"""

from __future__ import annotations
from typing import List, Optional, Dict

from .pipeline import Pipeline
from .registry import StageRegistry
from .config import PipelineConfig


def default_pipeline() -> Pipeline:
    """
    Create the default pipeline for SeaSenseLib.

    The default pipeline includes (in order):
    1. Mapping
    2. Unit Handling
    3. Transformation
    4. Derivation
    5. Metadata Extraction
    6. Metadata Enrichment
    7. Validation
    8. Finalization
    """
    config = PipelineConfig.from_resource("default")
    return create_pipeline(config=config)


def minimal_pipeline() -> Pipeline:
    """
    Create a minimal pipeline with only essential transformations.

    Includes (in order):
    1. Mapping
    2. Finalization
    """
    config = PipelineConfig.from_resource("minimal")
    return create_pipeline(config=config)




def create_pipeline(
    stage_names: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None
) -> Pipeline:
    """
    Create a pipeline from stage names or configuration.
    """
    registry = StageRegistry.get_instance()

    if config is not None:
        stages = []
        for stage_config in config.get_enabled_stages():
            stage = registry.get_stage(stage_config.name)

            if stage_config.config:
                stage.configure(stage_config.config)

            stages.append(stage)

        return Pipeline(stages)

    if stage_names is not None:
        stages = []
        for name in stage_names:
            stages.append(registry.get_stage(name))
        return Pipeline(stages)

    return default_pipeline()


def list_available_pipelines() -> Dict[str, str]:
    from importlib import resources

    profiles = resources.files('seasenselib.config.pipeline')
    result: Dict[str, str] = {}

    for entry in profiles.iterdir():
        if not entry.name.endswith('.json'):
            continue
        name = entry.name[:-5]
        config = PipelineConfig.from_resource(name)
        stages = [stage.name for stage in config.get_enabled_stages()]
        description = " -> ".join(stages) if stages else "No stages configured"
        result[name] = description

    return dict(sorted(result.items()))


__all__ = [
    'default_pipeline',
    'minimal_pipeline',
    'create_pipeline',
    'list_available_pipelines',
]
