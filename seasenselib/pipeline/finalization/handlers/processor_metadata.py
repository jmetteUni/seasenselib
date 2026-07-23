"""
Processor metadata handler.

Adds processor_* provenance attributes describing the L1 conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any
import json
import logging
import platform

from ...base import StageContext
from ...._version import get_version

logger = logging.getLogger(__name__)


@dataclass
class ProcessorMetadata:
    """Add processor_* metadata attributes."""

    level: str = "L1"
    include_machine: bool = True
    include_os: bool = True

    def configure(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            return
        if "level" in config:
            self.level = str(config["level"])
        if "include_machine" in config:
            self.include_machine = bool(config["include_machine"])
        if "include_os" in config:
            self.include_os = bool(config["include_os"])

    def process(self, context: StageContext) -> StageContext:
        ds = context.dataset
        meta = context.metadata

        def set_attr(key: str, value: Any) -> None:
            if value is None or value == "":
                return
            if key in ds.attrs:
                return
            ds.attrs[key] = value

        set_attr("processor_name", "SeaSenseLib")
        set_attr("processor_version", get_version())
        set_attr("processor_level", self.level)
        set_attr("processing_level", self.level)

        reader_module = meta.get("reader_module")
        reader_name = meta.get("format_name") or meta.get("reader_class")
        reader_key = meta.get("format_key")

        set_attr("processor_module", reader_module)
        set_attr("processor_module_name", reader_name)
        set_attr("processor_module_key", reader_key)

        set_attr("processor_runtime", platform.python_implementation())
        set_attr("processor_runtime_version", platform.python_version())
        set_attr(
            "processor_execution_time_utc",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        if self.include_machine:
            set_attr("processor_machine", platform.node())
        if self.include_os:
            set_attr("processor_os", f"{platform.system()} {platform.release()}")

        transformations = meta.get("transformations")
        if transformations:
            set_attr(
                "processor_transformations",
                json.dumps(transformations, ensure_ascii=False, default=str),
            )
            set_attr("processor_transformations_count", len(transformations))

        context.dataset = ds
        logger.debug("Added processor metadata attributes")
        return context


__all__ = ["ProcessorMetadata"]
