"""
User custom mapping strategy.

This strategy provides user-defined mappings.
"""

from __future__ import annotations
from typing import Dict, Optional
import logging

from ...interfaces import IMappingStrategy

logger = logging.getLogger(__name__)


class UserMappingStrategy(IMappingStrategy):
    """
    User-provided custom mappings.
    """
    
    def __init__(self, mappings: Dict[str, str]):
        """
        Initialize with user mappings.
        
        Parameters
        ----------
        mappings : Dict[str, str]
            Direct mapping from original to canonical name.
            Example: {'my_temp': 'temperature'}
        """
        self.mappings = {k.lower(): v for k, v in mappings.items()}
    
    def map(self, variable_name: str) -> Optional[str]:
        """Map using user-provided dictionary (case-insensitive)."""
        return self.mappings.get(variable_name.lower())
    
    def description(self) -> str:
        return f"User custom mappings ({len(self.mappings)} mappings)"
