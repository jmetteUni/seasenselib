"""
Metadata enrichment stage.

Applies CF, optional ACDD, and user metadata overrides.
"""

from __future__ import annotations

from typing import Dict, Any

from ..base import Stage, StageContext
from .handlers.cf_convention import CFConvention
from .handlers.acdd_convention import ACDDConvention
from .handlers.acdd_auto_metadata import AcddAutoMetadata
from .handlers.user_metadata_handler import UserMetadataHandler
from .handlers.whp_parameters import WHPParameters
from ..handler_registry import HandlerRegistry, HANDLER_GROUP_CONVENTIONS
from ..interfaces import IConvention


class MetadataEnrichmentStage(Stage):
    """Stage 5: Metadata enrichment and standardization."""

    def __init__(self, include_acdd: bool = False):
        self.include_cf = True
        self.include_acdd = include_acdd
        self.include_user_metadata = False
        self._cf = CFConvention()
        self._acdd = ACDDConvention()
        self._acdd_auto = AcddAutoMetadata()
        self._user_metadata = UserMetadataHandler()
        self._whp = WHPParameters()
        self._conventions: list[tuple[str, IConvention]] | None = None
        self._handler_order: list[str] = []

    def name(self) -> str:
        return "metadata_enrichment"

    def configure(self, config: Dict[str, Any]) -> None:
        handlers = config.get('handlers')
        if isinstance(handlers, list) and handlers:
            self.include_cf = 'cf' in handlers
            self.include_acdd = 'acdd' in handlers
            self.include_user_metadata = 'user_metadata' in handlers

            conventions: list[tuple[str, IConvention]] = []
            builtin = {
                'cf': self._cf,
                'acdd': self._acdd,
                'acdd_auto': self._acdd_auto,
                'whp': self._whp,
                'user_metadata': self._user_metadata,
            }
            builtin_names = set(builtin.keys())
            plugin_map = {}
            if any(name not in builtin_names for name in handlers):
                plugin_map = HandlerRegistry.get(HANDLER_GROUP_CONVENTIONS, IConvention)
            for name in handlers:
                if name in builtin:
                    conventions.append((name, builtin[name]))
                    continue
                cls = plugin_map.get(name)
                if cls is not None:
                    try:
                        conventions.append((name, cls()))
                    except Exception:
                        continue
            self._conventions = conventions
            self._handler_order = [name for name, _ in conventions]
        if 'include_cf' in config:
            self.include_cf = bool(config['include_cf'])
        if 'include_acdd' in config:
            self.include_acdd = bool(config['include_acdd'])
        if 'include_user_metadata' in config:
            self.include_user_metadata = bool(config['include_user_metadata'])

    def process(self, context: StageContext) -> StageContext:
        metadata_registry = context.metadata.get('_metadata_registry')
        ds = context.dataset

        # Provide raw/source format hints early for conservative ACDD text generation
        if not ds.attrs.get("raw_format") and context.metadata.get("format_key"):
            ds.attrs["raw_format"] = context.metadata["format_key"]
        if not ds.attrs.get("raw_filename") and context.metadata.get("source_file"):
            try:
                from pathlib import Path
                ds.attrs["raw_filename"] = Path(str(context.metadata["source_file"])).name
            except Exception:
                pass
        if not ds.attrs.get("source_format_name") and context.metadata.get("format_name"):
            ds.attrs["source_format_name"] = context.metadata["format_name"]

        from ..utils import record_handler_applied

        if self._conventions is not None:
            for name, conv in self._conventions:
                record_handler_applied(context.metadata, self.name(), name)
                if isinstance(conv, UserMetadataHandler):
                    conv.set_user_metadata(context.metadata.get('user_metadata'))
                ds = conv.enrich(ds, metadata_registry)
                if isinstance(conv, UserMetadataHandler):
                    if conv.applied:
                        context.metadata['user_metadata_applied'] = True
                        context.metadata['user_metadata_global_keys'] = conv.applied_global_keys
                        context.metadata['user_metadata_variable_keys'] = conv.applied_variable_keys
                    if conv.warnings:
                        context.metadata.setdefault('warnings', []).extend(conv.warnings)
                errors = conv.validate(ds)
                if errors:
                    context.metadata.setdefault('warnings', []).extend([str(e) for e in errors[:5]])
        else:
            if self.include_cf:
                record_handler_applied(context.metadata, self.name(), "cf")
                ds = self._cf.enrich(ds, metadata_registry)
                # Validate CF (warnings only)
                errors = self._cf.validate(ds)
                if errors:
                    context.metadata.setdefault('warnings', []).extend([str(e) for e in errors[:5]])

            if self.include_acdd:
                record_handler_applied(context.metadata, self.name(), "acdd")
                ds = self._acdd.enrich(ds, metadata_registry)
                errors = self._acdd.validate(ds)
                if errors:
                    context.metadata.setdefault('warnings', []).extend([str(e) for e in errors[:5]])

            if self.include_user_metadata:
                record_handler_applied(context.metadata, self.name(), "user_metadata")
                self._user_metadata.set_user_metadata(context.metadata.get('user_metadata'))
                ds = self._user_metadata.enrich(ds, metadata_registry)
                if self._user_metadata.applied:
                    context.metadata['user_metadata_applied'] = True
                    context.metadata['user_metadata_global_keys'] = self._user_metadata.applied_global_keys
                    context.metadata['user_metadata_variable_keys'] = self._user_metadata.applied_variable_keys
                if self._user_metadata.warnings:
                    context.metadata.setdefault('warnings', []).extend(self._user_metadata.warnings)

        return StageContext(dataset=ds, metadata=context.metadata)
