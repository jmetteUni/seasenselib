"""
Finalization stage.

Applies RAW metadata, processor metadata, global attributes, and sorting.
"""

from __future__ import annotations

from typing import Dict, Any

from ..base import Stage, StageContext
from .handlers.global_attributes import GlobalAttributes
from .handlers.processor_metadata import ProcessorMetadata
from .handlers.raw_metadata import RawMetadata
from .handlers.sorting import Sorting


class FinalizationStage(Stage):
    """Finalization: global attributes + sorting."""

    def __init__(self):
        self._raw_metadata = RawMetadata()
        self._processor_metadata = ProcessorMetadata()
        self._global_attributes = GlobalAttributes()
        self._sorting = Sorting()
        self._handler_order = [
            "raw_metadata",
            "processor_metadata",
            "global_attributes",
            "sorting",
        ]

    def name(self) -> str:
        return "finalization"

    def configure(self, config: Dict[str, Any]) -> None:
        handlers = config.get('handlers')
        if isinstance(handlers, list) and handlers:
            self._handler_order = list(handlers)
        else:
            self._handler_order = [
                "raw_metadata",
                "processor_metadata",
                "global_attributes",
                "sorting",
            ]

        global_config = config.get('global_attributes')
        if isinstance(global_config, dict):
            self._global_attributes.configure(global_config)

        raw_config = config.get('raw_metadata')
        if isinstance(raw_config, dict):
            self._raw_metadata.configure(raw_config)

        processor_config = config.get('processor_metadata')
        if isinstance(processor_config, dict):
            self._processor_metadata.configure(processor_config)

    def process(self, context: StageContext) -> StageContext:
        from ..utils import record_handler_applied
        handlers = {
            "raw_metadata": self._raw_metadata,
            "processor_metadata": self._processor_metadata,
            "global_attributes": self._global_attributes,
            "sorting": self._sorting,
        }
        for name in self._handler_order:
            handler = handlers.get(name)
            if handler is None:
                continue
            context = handler.process(context)
            record_handler_applied(context.metadata, self.name(), name)
        return context
