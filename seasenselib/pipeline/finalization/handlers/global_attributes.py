"""
Global attributes logic.

Adds processing history and global metadata attributes.
"""

from __future__ import annotations
from typing import Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import logging

from ...base import StageContext
from ...._version import get_version

logger = logging.getLogger(__name__)


class GlobalAttributes:
    """
    Adds global attributes to the dataset.
    
    Adds:
    - history: Processing history with timestamp
    - date_created: ISO timestamp
    - source: Source description (if provided)
    
    Attributes
    ----------
    add_history : bool
        Whether to add/update history attribute.
    add_source : bool
        Whether to add source attribute.
    add_timestamps : bool
        Whether to add date_created/date_modified.
    """
    
    def __init__(
        self,
        add_history: bool = True,
        add_source: bool = True,
        add_timestamps: bool = True
    ):
        """
        Initialize global attributes logic.
        
        Parameters
        ----------
        add_history : bool, default=True
            Whether to add/update history attribute.
        add_source : bool, default=True
            Whether to add source file information.
        add_timestamps : bool, default=True
            Whether to add creation/modification timestamps.
        """
        self.add_history = add_history
        self.add_source = add_source
        self.add_timestamps = add_timestamps
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure global attribute settings."""
        if 'add_history' in config:
            self.add_history = config['add_history']
        if 'add_source' in config:
            self.add_source = config['add_source']
        if 'add_timestamps' in config:
            self.add_timestamps = config['add_timestamps']
    
    def process(self, context: StageContext) -> StageContext:
        """
        Add global attributes.
        
        Parameters
        ----------
        context : StageContext
            The processing context.
        
        Returns
        -------
        StageContext
            Updated context with global attributes.
        """
        ds = context.dataset
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Add history
        if self.add_history:
            history_entry = self._build_history_entry(timestamp, context)
            
            if 'history' in ds.attrs and ds.attrs['history']:
                # Prepend to existing history
                ds.attrs['history'] = f"{history_entry}\n{ds.attrs['history']}"
            else:
                ds.attrs['history'] = history_entry
        
        # Add source information (only if explicitly provided)
        if self.add_source:
            source_value = context.metadata.get('source')
            if source_value and 'source' not in ds.attrs:
                ds.attrs['source'] = source_value
        
        # Add timestamps
        if self.add_timestamps:
            if 'date_created' not in ds.attrs:
                ds.attrs['date_created'] = timestamp
            ds.attrs['date_modified'] = timestamp
        
        context.dataset = ds
        logger.debug("Added global attributes")
        
        return context
    
    def _build_history_entry(self, timestamp: str, context: StageContext) -> str:
        """
        Build history entry from processing context.
        
        Parameters
        ----------
        timestamp : str
            ISO timestamp.
        context : StageContext
            Processing context with metadata.
        
        Returns
        -------
        str
            History entry string.
        """
        parts = [f"{timestamp} - Processed by SeaSenseLib v{get_version()}"]
        
        # Add reader information
        if 'reader_class' in context.metadata:
            parts.append(f"Reader: {context.metadata['reader_class']}")
        if 'format_name' in context.metadata:
            parts.append(f"Format: {context.metadata['format_name']}")
        if 'source_file' in context.metadata:
            parts.append(f"Source file: {Path(str(context.metadata['source_file'])).name}")
        
        # Add layer information
        if 'stages_applied' in context.metadata:
            stages = ', '.join(context.metadata['stages_applied'])
            parts.append(f"Stages: {stages}")
        
        # Add specific transformations
        if 'variable_mappings' in context.metadata:
            count = len(context.metadata['variable_mappings'])
            parts.append(f"Mapped {count} variables")
        
        if 'derived_parameters' in context.metadata:
            derived = ', '.join(context.metadata['derived_parameters'])
            parts.append(f"Derived: {derived}")
        
        return '; '.join(parts)
