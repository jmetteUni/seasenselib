"""
Regex-based mapping strategy.

This strategy maps variables using regular expression patterns.
"""

from __future__ import annotations
from typing import List, Optional
import re
import logging

from ...interfaces import IMappingStrategy

logger = logging.getLogger(__name__)


class RegexMappingStrategy(IMappingStrategy):
    """
    Maps variables using regular expression patterns.
    
    Useful for handling numbered variants (temperature_1, temperature_2)
    or flexible pattern matching.
    """
    
    def __init__(self, patterns: List[tuple[str, str]]):
        """
        Initialize with regex patterns.
        
        Parameters
        ----------
        patterns : List[tuple[str, str]]
            List of (pattern, canonical_name) tuples.
            Example: [(r'^temp(?:erature)?_?\\d*$', 'temperature')]
        """
        self.patterns = [(re.compile(pattern, re.IGNORECASE), canonical) 
                        for pattern, canonical in patterns]

    @staticmethod
    def load_patterns() -> List[tuple[str, str]]:
        """
        Load regex patterns from the knowledge base.

        Raises
        ------
        ValueError
            If the knowledge file is missing or invalid.
        """
        from seasenselib.knowledge import load_json
        data = load_json("pipeline/mapping/mapping_regex.json")
        if not isinstance(data, list) or not data:
            raise ValueError(
                "Regex mapping patterns could not be loaded from "
                "pipeline/mapping/mapping_regex.json"
            )

        patterns: List[tuple[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pattern = item.get('pattern')
            canonical = item.get('canonical')
            if pattern and canonical:
                patterns.append((pattern, canonical))

        if not patterns:
            raise ValueError(
                "Regex mapping patterns could not be loaded from "
                "pipeline/mapping/mapping_regex.json"
            )
        return patterns
    
    def map(self, variable_name: str) -> Optional[str]:
        """Map using regex matching."""
        for pattern, canonical in self.patterns:
            if pattern.match(variable_name):
                return canonical
        return None
    
    def description(self) -> str:
        return f"Regex-based mapping ({len(self.patterns)} patterns)"
