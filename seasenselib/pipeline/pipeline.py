"""
Pipeline executor for running stages in sequence.

This module provides the Pipeline class that executes stages in configuration order.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging
import xarray as xr

from .base import Stage, StageContext

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Executes stages in sequence to transform datasets.

    The pipeline:
    1. Executes stages in the configured order
    2. Checks can_process() for each stage
    3. Executes process() for enabled stages
    4. Returns the final dataset

    Attributes
    ----------
    stages : List[Stage]
        The stages to execute, in the given order.
    """

    def __init__(self, stages: List[Stage]):
        """
        Initialize the pipeline with stages.

        Parameters
        ----------
        stages : List[Stage]
            The stages to execute.
        """
        self.pipeline = stages
    
    def execute(self, dataset: xr.Dataset, metadata: Optional[Dict[str, Any]] = None) -> xr.Dataset:
        """
        Execute all stages and return the final dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset to process.
        metadata : Dict[str, Any], optional
            Initial metadata. If None, starts with empty dict.
        
        Returns
        -------
        xr.Dataset
            The processed dataset after all stages.
        
        Raises
        ------
        Exception
            If any layer raises an exception during processing.
        
        Examples
        --------
        >>> ds = xr.Dataset({'t090C': (['time'], [10, 11, 12])})
        >>> result = pipeline.execute(ds, metadata={'source': 'test.cnv'})
        """
        context = StageContext(dataset=dataset, metadata=metadata or {})

        logger.info(f"Starting pipeline with {len(self.pipeline)} stages")

        for stage in self.pipeline:
            stage_name = stage.name()

            if not stage.can_process(context):
                logger.debug(f"Skipping stage '{stage_name}' - can_process() returned False")
                continue

            logger.debug(f"Executing stage '{stage_name}'")
            try:
                context = stage.process(context)
                logger.debug(f"Stage '{stage_name}' completed successfully")
            except Exception as e:
                logger.error(f"Stage '{stage_name}' failed: {e}")
                raise

        logger.info("Pipeline completed successfully")
        return context.dataset
    
    def get_stage_order(self) -> List[str]:
        """
        Get the execution order of stages.

        Returns
        -------
        List[str]
            Stage names in execution order.
        """
        return [stage.name() for stage in self.pipeline]
    
    def describe(self) -> str:
        """
        Get a human-readable description of the pipeline.

        Returns
        -------
        str
            Multi-line description showing stage order.
        """
        lines = [f"Pipeline with {len(self.pipeline)} stages:"]
        for i, stage in enumerate(self.pipeline, 1):
            lines.append(f"  {i}. {stage.name()}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"Pipeline(stages={len(self.pipeline)})"
