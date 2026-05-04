"""
Variable mapping module.

This module provides strategies for mapping sensor-specific variable names
to canonical names. Each reader class can provide its own format-specific
mappings via the format_mappings() class method.

The mapping follows this order:
1. User custom mappings - User overrides everything
2. Reader-provided mappings - Format-specific from reader class
3. Default mappings - From parameters.py
4. Regex patterns - Fallback patterns
"""

from .handlers.dict_mapping_strategy import DictMappingStrategy
from .handlers.regex_mapping_strategy import RegexMappingStrategy
from .handlers.user_mapping_strategy import UserMappingStrategy
from .handlers.strategy_runner import MappingStrategyRunner
from .handlers.mapping_runner import MappingRunner
from .stage import MappingStage

__all__ = [
    'DictMappingStrategy',
    'RegexMappingStrategy',
    'UserMappingStrategy',
    'MappingStrategyRunner',
    'MappingRunner',
    'MappingStage',
]
