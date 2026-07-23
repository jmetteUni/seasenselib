"""Transformation stage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from ..base import Stage, StageContext
from ..handler_registry import HandlerRegistry, HANDLER_GROUP_TRANSFORMATIONS
from ..interfaces import ITransformation
from ..utils import resolve_components
from .handlers.reader_transformations import ReaderTransformations
from .handlers.transformation_runner import TransformationRunner


logger = logging.getLogger(__name__)


class TransformationStage(Stage):
    """Apply optional data/value transformations before validation."""

    def __init__(self, transformations: Optional[List[ITransformation]] = None):
        self._explicit_transformations = (
            list(transformations) if transformations is not None else None
        )
        initial_transformations = (
            [ReaderTransformations()]
            if self._explicit_transformations is None
            else self._explicit_transformations
        )
        self._handler_order: list[str] = ["reader"]
        self._runner = TransformationRunner(
            transformations=initial_transformations,
            stage_name=self.name(),
        )
        self._enabled_formats: set[str] = set()
        self._disabled_formats: set[str] = set()
        self._enabled_reader_classes: set[str] = set()
        self._disabled_reader_classes: set[str] = set()
        self._enabled_reader_groups: set[str] = set()
        self._disabled_reader_groups: set[str] = set()

    def name(self) -> str:
        return "transformation"

    def configure(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            return

        self._enabled_formats = self._normalized_set(config.get("enabled_formats"))
        self._disabled_formats = self._normalized_set(config.get("disabled_formats"))
        self._enabled_reader_classes = self._normalized_set(
            config.get("enabled_reader_classes"),
            lower=False,
        )
        self._disabled_reader_classes = self._normalized_set(
            config.get("disabled_reader_classes"),
            lower=False,
        )
        self._enabled_reader_groups = self._normalized_set(
            config.get("enabled_reader_groups")
        )
        self._disabled_reader_groups = self._normalized_set(
            config.get("disabled_reader_groups")
        )

        handlers = config.get("handlers")
        if "handlers" in config:
            if handlers is None:
                self._handler_order = []
            elif isinstance(handlers, str):
                self._handler_order = [handlers]
            elif isinstance(handlers, list):
                self._handler_order = list(handlers)
            else:
                self._handler_order = []
        if self._explicit_transformations is not None and "handlers" not in config:
            transformations = list(self._explicit_transformations)
        else:
            mapping = self._handler_mapping(self._handler_order)
            transformations = resolve_components(self._handler_order, mapping)

        self._runner = TransformationRunner(
            transformations=transformations,
            stage_name=self.name(),
        )

    def can_process(self, context: StageContext) -> bool:
        return self._context_enabled(context.metadata)

    def process(self, context: StageContext) -> StageContext:
        if not self.can_process(context):
            return context
        return self._runner.process(context)

    @staticmethod
    def _handler_mapping(handler_order: List[str]) -> Dict[str, type[ITransformation]]:
        mapping: Dict[str, type[ITransformation]] = {
            "reader": ReaderTransformations,
        }
        if any(name not in mapping for name in handler_order):
            plugin_mapping = HandlerRegistry.get(
                HANDLER_GROUP_TRANSFORMATIONS,
                ITransformation,
            )
            for name, cls in plugin_mapping.items():
                mapping.setdefault(name, cls)
        return mapping

    def _context_enabled(self, metadata: Dict[str, Any]) -> bool:
        format_key = str(metadata.get("format_key") or "").lower()
        reader_class = str(metadata.get("reader_class") or "")
        reader_groups = self._reader_groups(metadata)

        if format_key and format_key in self._disabled_formats:
            return False
        if reader_class and reader_class in self._disabled_reader_classes:
            return False
        if reader_groups & self._disabled_reader_groups:
            return False

        if self._enabled_formats and format_key not in self._enabled_formats:
            return False
        if (
            self._enabled_reader_classes
            and reader_class not in self._enabled_reader_classes
        ):
            return False
        if (
            self._enabled_reader_groups
            and not (reader_groups & self._enabled_reader_groups)
        ):
            return False
        return bool(self._runner.transformations)

    @staticmethod
    def _reader_groups(metadata: Dict[str, Any]) -> set[str]:
        groups: set[str] = set()
        raw_groups = metadata.get("reader_groups") or metadata.get("reader_group")
        if isinstance(raw_groups, str):
            groups.add(raw_groups.strip().lower())
        elif isinstance(raw_groups, list):
            groups.update(str(item).strip().lower() for item in raw_groups if item)

        format_key = str(metadata.get("format_key") or "").strip().lower()
        if format_key:
            groups.add(format_key)
            groups.add(format_key.split("-", 1)[0])
        return {group for group in groups if group}

    @staticmethod
    def _normalized_set(value: Any, lower: bool = True) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            return set()
        result = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            result.add(text.lower() if lower else text)
        return result


__all__ = ["TransformationStage"]
