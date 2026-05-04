"""
Stage-based processing pipeline for SeaSenseLib.

Public API for building and executing processing pipelines.
"""

from .base import Stage, StageContext
from .pipeline import Pipeline
from .config import PipelineConfig, StageConfig
from .registry import StageRegistry
from .factory import (
    default_pipeline,
    minimal_pipeline,
    create_pipeline,
    list_available_pipelines,
)

__all__ = [
    "Stage",
    "StageContext",
    "Pipeline",
    "PipelineConfig",
    "StageConfig",
    "StageRegistry",
    "default_pipeline",
    "minimal_pipeline",
    "create_pipeline",
    "list_available_pipelines",
]
