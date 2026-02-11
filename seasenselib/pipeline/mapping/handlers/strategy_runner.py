"""
Mapping strategy runner.

The MappingStrategyRunner applies strategies in the order they are registered to find the best mapping.
"""

from __future__ import annotations
from typing import List, Optional
import logging

from ...interfaces import IMappingStrategy

logger = logging.getLogger(__name__)


class MappingStrategyRunner:
    """
    Manages multiple mapping strategies and applies them in registration order.
    """
    
    def __init__(self, strategies: Optional[List[IMappingStrategy]] = None):
        """
        Initialize with mapping strategies.
        
        Parameters
        ----------
        strategies : List[IMappingStrategy], optional
            List of strategies to use. If None, starts empty.
        """
        self.strategies = strategies or []
    
    def add_strategy(self, strategy: IMappingStrategy) -> None:
        """
        Add a mapping strategy.
        
        Parameters
        ----------
        strategy : IMappingStrategy
            The strategy to add.
        """
        self.strategies.append(strategy)
    
    def map(self, variable_name: str) -> Optional[str]:
        """
        Map a variable name using registered strategies.
        
        Tries strategies in registration order and returns the first successful mapping.
        
        Parameters
        ----------
        variable_name : str
            The variable name to map.
        
        Returns
        -------
        Optional[str]
            Canonical name, or None if no mapping found.
        """
        for strategy in self.strategies:
            result = strategy.map(variable_name)
            if result is not None:
                logger.debug(f"Mapped '{variable_name}' -> '{result}' using {strategy.description()}")
                return result
        return None
    
    def describe(self) -> str:
        """Get description of all registered strategies."""
        if not self.strategies:
            return "MappingStrategyRunner with no strategies"
        
        lines = ["MappingStrategyRunner with strategies (in registration order):"]
        for i, strategy in enumerate(self.strategies, 1):
            lines.append(f"  {i}. {strategy.description()}")
        return "\n".join(lines)
