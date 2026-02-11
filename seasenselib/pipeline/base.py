"""
Base classes for the stage system.

This module defines the core interfaces that all stages must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import xarray as xr


@dataclass
class StageContext:
    """
    Context object passed between stages in the pipeline.
    
    Contains the dataset being processed and accumulated metadata.
    
    Attributes
    ----------
    dataset : xr.Dataset
        The xarray Dataset being processed. Each layer may modify this.
    metadata : Dict[str, Any]
        Metadata accumulated during processing. Stages can add information
        here that may be useful for subsequent stages or for users.
    
    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.Dataset({'temp': (['time'], [10, 11, 12])})
    >>> context = StageContext(ds, metadata={'source': 'test.cnv'})
    >>> context.metadata['variables_mapped'] = ['temp']
    """
    
    dataset: xr.Dataset
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'StageContext':
        """
        Create a shallow copy of the context.
        
        The dataset is not copied (xarray datasets are immutable by convention).
        The metadata dictionary is shallow-copied.
        
        Returns
        -------
        StageContext
            A new context with the same dataset and a copy of metadata.
        """
        return StageContext(
            dataset=self.dataset,
            metadata=self.metadata.copy()
        )


class Stage(ABC):
    """
    Abstract base class for all processing stages.
    
    Each stage performs a specific transformation or enrichment on the dataset.
    Stages are executed in the order defined by the pipeline configuration.
    
    Subclasses must implement:
    - name(): Return unique identifier for this stage
    - process(): Transform the dataset and return updated context
    
    Subclasses may override:
    - can_process(): Check if stage should run (default: always True)
    - configure(): Set stage-specific configuration
    - configure(): Set stage-specific configuration
    
    Examples
    --------
    >>> class MyStage(Stage):
    ...     def name(self) -> str:
    ...         return "my_stage"
    ...     
    ...     def process(self, context: StageContext) -> StageContext:
    ...         # Add a new variable
    ...         context.dataset['new_var'] = context.dataset['temp'] * 2
    ...         context.metadata['my_stage_applied'] = True
    ...         return context
    """
    
    @abstractmethod
    def name(self) -> str:
        """
        Get the unique identifier for this stage.
        
        This name is used for:
        - Configuration references
        - Registry lookups
        - Logging and debugging
        
        Returns
        -------
        str
            Unique stage identifier (e.g., 'mapping', 'metadata_enrichment')
        """
        pass
    
    @abstractmethod
    def process(self, context: StageContext) -> StageContext:
        """
        Process the dataset and return updated context.
        
        This is the main method where the layer's transformation logic lives.
        
        The stage should:
        1. Read from context.dataset and context.metadata
        2. Perform transformations on the dataset
        3. Update context.metadata with any relevant information
        4. Return the updated context
        
        Note: xarray Datasets are typically treated as immutable, so most
        operations return a new Dataset. The layer should return a context
        with the updated dataset.
        
        Parameters
        ----------
        context : StageContext
            The current processing context with dataset and metadata.
        
        Returns
        -------
        StageContext
            Updated context with transformed dataset and metadata.
        
        Raises
        ------
        Exception
            If processing fails. Exceptions are propagated to the pipeline.
        """
        pass
    
    def can_process(self, context: StageContext) -> bool:
        """
        Check if this stage can process the current context.
        
        This method is called by the pipeline before process(). If it returns
        False, the stage is skipped.
        
        Use this for:
        - Checking if required variables are present
        - Verifying prerequisites from previous stages
        - Conditional stage execution based on metadata
        
        Parameters
        ----------
        context : StageContext
            The current processing context.
        
        Returns
        -------
        bool
            True if the layer should process this context, False to skip.
        
        Examples
        --------
        >>> def can_process(self, context: StageContext) -> bool:
        ...     # Only process if temperature variable exists
        ...     return 'temperature' in context.dataset.data_vars
        """
        return True
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the stage with settings from configuration.
        
        Called by the pipeline when building from configuration.
        Override this to support layer-specific configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for this stage.
        
        Examples
        --------
        >>> class MyStage(Stage):
        ...     def __init__(self):
        ...         self.option = "default"
        ...     
        ...     def configure(self, config: Dict[str, Any]) -> None:
        ...         if 'option' in config:
        ...             self.option = config['option']
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"{self.__class__.__name__}(name='{self.name()}')"
