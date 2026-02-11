"""
Metadata extraction stage.

Extracts raw metadata into a registry for later enrichment.
"""

from __future__ import annotations

from typing import Dict, Any

from ..base import Stage, StageContext
from .handlers.extraction_runner import MetadataExtractionRunner
from .handlers.attribute_extractor import AttributeMetadataExtractor
from .handlers.global_attribute_extractor import GlobalAttributeMetadataExtractor
from ..interfaces import IMetadataExtractor
from ..handler_registry import HandlerRegistry, HANDLER_GROUP_METADATA_EXTRACTORS
from ..utils import resolve_components


class MetadataExtractionStage(Stage):
    """Stage 3: Metadata extraction."""

    def __init__(self):
        self._extraction = MetadataExtractionRunner()
        self._handler_order: list[str] = ["attributes", "global_attributes"]

    def name(self) -> str:
        return "metadata_extraction"

    def configure(self, config: Dict[str, Any]) -> None:
        handlers = config.get('handlers')
        if not isinstance(handlers, list) or not handlers:
            return
        self._handler_order = list(handlers)

        mapping = {
            'attributes': AttributeMetadataExtractor,
            'global_attributes': GlobalAttributeMetadataExtractor,
        }
        if any(name not in mapping for name in handlers):
            plugin_mapping = HandlerRegistry.get(HANDLER_GROUP_METADATA_EXTRACTORS, IMetadataExtractor)
            for name, cls in plugin_mapping.items():
                if name not in mapping:
                    mapping[name] = cls
        extractors: list[IMetadataExtractor] = resolve_components(handlers, mapping)

        if extractors:
            self._extraction = MetadataExtractionRunner(extractors=extractors)

    def process(self, context: StageContext) -> StageContext:
        from ..utils import record_handler_applied
        for name in self._handler_order:
            record_handler_applied(context.metadata, self.name(), name)
        return self._extraction.process(context)
