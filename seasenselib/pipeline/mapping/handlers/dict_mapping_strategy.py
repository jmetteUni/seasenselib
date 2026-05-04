"""
Dictionary-based mapping strategy.

This strategy maps variables using a dictionary of canonical names to aliases.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import logging

from ...interfaces import IMappingStrategy

logger = logging.getLogger(__name__)


class DictMappingStrategy(IMappingStrategy):
    """
    Maps variables using a dictionary of canonical -> [aliases].
    
    This is the most common mapping strategy, used for the default mappings
    in parameters.py and reader-provided mappings.
    """
    
    def __init__(self, mappings: Dict[str, List[str]]):
        """
        Initialize with mapping dictionary.
        
        Parameters
        ----------
        mappings : Dict[str, List[str]]
            Dictionary mapping canonical names to lists of aliases.
            Example: {'temperature': ['t090C', 't068', 'TEMP']}
        """
        self.mappings = mappings
        
        # Build reverse lookup for efficiency
        self._reverse_map: Dict[str, str] = {}
        for canonical, aliases in mappings.items():
            for alias in aliases:
                self._reverse_map[alias.lower()] = canonical
    
    def map(self, variable_name: str) -> Optional[str]:
        """Map using dictionary lookup (case-insensitive)."""
        return self._reverse_map.get(variable_name.lower())
    
    def description(self) -> str:
        return f"Dictionary-based mapping ({len(self.mappings)} canonical names)"
