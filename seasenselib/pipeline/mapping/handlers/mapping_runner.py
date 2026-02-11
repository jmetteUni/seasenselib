"""
Mapping runner.

Converts sensor-specific variable names to canonical names using
an ordered strategy pattern.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import logging

import xarray as xr
import seasenselib.parameters as params

from ...base import StageContext
from ...interfaces import IMappingStrategy
from .strategy_runner import MappingStrategyRunner
from .dict_mapping_strategy import DictMappingStrategy
from .regex_mapping_strategy import RegexMappingStrategy
from .user_mapping_strategy import UserMappingStrategy

logger = logging.getLogger(__name__)


class MappingRunner:
    """
    Maps sensor-specific variable names to canonical names.
    
    Uses an ordered strategy pattern with the following order:
    1. User custom mappings
    2. Reader-provided mappings (format-specific)
    3. Default mappings from parameters.py
    4. Regex patterns (fallback)
    
    **IMPORTANT**: Reader-specific mappings should be provided by each reader
    class via the format_mappings() class method, NOT hard-coded in this layer.
    
    Attributes
    ----------
    mapper : MappingStrategyRunner
        The strategy runner with registered strategies.
    preserve_original : bool
        If True, keeps original variable names as attributes.
    """
    
    def __init__(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
        reader_mappings: Optional[Dict[str, List[str]]] = None,
        preserve_original: bool = True,
        use_custom_mappings: bool = True,
        use_reader_mappings: bool = True,
        use_default_mappings: bool = True,
        use_regex: bool = True
    ):
        """
        Initialize the variable mapping logic.
        
        Parameters
        ----------
        custom_mappings : Dict[str, str], optional
            User-provided custom mappings (original -> canonical).
            These are applied first.
        reader_mappings : Dict[str, List[str]], optional
            Reader-provided format-specific mappings (canonical -> [aliases]).
            These should come from the reader class's format_mappings() method.
        preserve_original : bool, default=True
            Whether to preserve original names in variable attributes.
        """
        self.preserve_original = preserve_original
        self.use_custom_mappings = use_custom_mappings
        self.use_reader_mappings = use_reader_mappings
        self.use_default_mappings = use_default_mappings
        self.use_regex = use_regex
        self.mapper = MappingStrategyRunner()
        self._build_mapper(custom_mappings, reader_mappings)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the layer.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration with keys:
            - custom_mappings: Dict[str, str] - User mappings
            - reader_mappings: Dict[str, List[str]] - Reader-provided mappings
            - preserve_original: bool - Whether to preserve original names
        """
        if 'preserve_original' in config:
            self.preserve_original = config['preserve_original']
        if 'use_custom_mappings' in config:
            self.use_custom_mappings = bool(config['use_custom_mappings'])
        if 'use_reader_mappings' in config:
            self.use_reader_mappings = bool(config['use_reader_mappings'])
        if 'use_default_mappings' in config:
            self.use_default_mappings = bool(config['use_default_mappings'])
        if 'use_regex' in config:
            self.use_regex = bool(config['use_regex'])

        custom = config.get('custom_mappings')
        reader = config.get('reader_mappings')
        plugin_strategies = config.get('plugin_strategies') or []
        self._build_mapper(custom, reader, plugin_strategies)

    def _build_mapper(
        self,
        custom_mappings: Optional[Dict[str, str]],
        reader_mappings: Optional[Dict[str, List[str]]],
        extra_strategies: Optional[List[IMappingStrategy]] = None,
    ) -> None:
        self.mapper = MappingStrategyRunner()

        # 1. User custom mappings
        if self.use_custom_mappings and custom_mappings:
            self.mapper.add_strategy(
                UserMappingStrategy(custom_mappings)
            )
            logger.debug(f"Added user custom mappings: {len(custom_mappings)} entries")

        # 2. Reader-provided format-specific mappings
        if self.use_reader_mappings and reader_mappings:
            self.mapper.add_strategy(
                DictMappingStrategy(reader_mappings)
            )
            logger.debug(f"Added reader mappings: {len(reader_mappings)} canonical names")

        # 3. Default mappings from parameters.py
        if self.use_default_mappings:
            self.mapper.add_strategy(
                DictMappingStrategy(params.default_mappings)
            )
            logger.debug(f"Added default mappings: {len(params.default_mappings)} canonical names")

        # 4. Plugin strategies (if any), before regex fallback
        if extra_strategies:
            for strategy in extra_strategies:
                self.mapper.add_strategy(strategy)
            logger.debug(f"Added plugin strategies: {len(extra_strategies)}")

        # 5. Regex patterns for common cases (fallback)
        if self.use_regex:
            regex_patterns = RegexMappingStrategy.load_patterns()
            self.mapper.add_strategy(
                RegexMappingStrategy(regex_patterns)
            )
            logger.debug(f"Added regex patterns: {len(regex_patterns)} patterns")

    def process(self, context: StageContext) -> StageContext:
        """
        Map variable names in the dataset.
        
        This implements smart numbering:
        - If only ONE variable maps to a canonical name: no suffix (e.g., 'temperature')
        - If MULTIPLE variables map to same canonical name: ALL get numbered (_1, _2, _3, ...)
        
        Parameters
        ----------
        context : StageContext
            The processing context.
        
        Returns
        -------
        StageContext
            Updated context with renamed variables.
        """
        ds = context.dataset
        rename_map = {}
        
        logger.debug(f"Mapping strategies: {self.mapper.describe()}")
        
        # PHASE 1: Collect all mappings (original_name -> canonical_name)
        mappings = {}
        for var_name in list(ds.data_vars):
            canonical = self.mapper.map(var_name)
            if canonical and canonical != var_name:
                mappings[var_name] = canonical
        
        # PHASE 2: Group by canonical name to detect duplicates
        from collections import defaultdict
        canonical_groups = defaultdict(list)
        for original, canonical in mappings.items():
            canonical_groups[canonical].append(original)
        
        # Include existing canonical variables to avoid rename collisions
        for canonical in list(canonical_groups.keys()):
            if canonical in ds.data_vars and canonical not in canonical_groups[canonical]:
                # Put existing canonical first to keep numbering stable
                canonical_groups[canonical].insert(0, canonical)
        
        # PHASE 3: Assign final names with smart numbering
        for canonical, originals in canonical_groups.items():
            if len(originals) == 1:
                # Only one variable maps to this canonical name - no suffix needed
                if originals[0] != canonical:
                    rename_map[originals[0]] = canonical
                    logger.debug(f"Mapping '{originals[0]}' -> '{canonical}'")
            else:
                # Multiple variables map to same canonical - ALL get numbered (_1, _2, ...)
                for idx, original in enumerate(originals, start=1):
                    final_name = f"{canonical}_{idx}"
                    rename_map[original] = final_name
                    logger.debug(f"Mapping '{original}' -> '{final_name}'")
        
        # PHASE 4: Apply renaming
        if rename_map:
            ds = ds.rename(rename_map)
            
            # Preserve original names if requested
            if self.preserve_original:
                for original, canonical in rename_map.items():
                    if canonical in ds.data_vars:
                        ds[canonical].attrs['original_name'] = original
            
            # Store mapping in metadata
            context.metadata['variable_mappings'] = rename_map
            context.dataset = ds
            
            logger.info(f"Mapped {len(rename_map)} variables")
        else:
            logger.debug("No variables mapped")

        return context
