"""Reader-provided transformation handler."""

from __future__ import annotations

from typing import Any, Dict, List
import logging

import xarray as xr

from ...interfaces import ITransformation, TransformationRecord


logger = logging.getLogger(__name__)


class ReaderTransformations(ITransformation):
    """Apply transformation handlers supplied by the active reader."""

    def name(self) -> str:
        return "reader"

    def can_transform(
        self,
        dataset: xr.Dataset,
        context: Dict[str, Any] | None = None,
    ) -> bool:
        return bool((context or {}).get("reader_transformations"))

    def transform(
        self,
        dataset: xr.Dataset,
        context: Dict[str, Any] | None = None,
    ) -> tuple[xr.Dataset, List[TransformationRecord | Dict[str, Any]]]:
        metadata = context or {}
        records: List[TransformationRecord | Dict[str, Any]] = []
        result = dataset

        for transformation in metadata.get("reader_transformations", []) or []:
            handler = self._resolve_transformation(transformation)
            if handler is None:
                logger.debug(
                    "Ignoring unsupported reader transformation specification: %r",
                    transformation,
                )
                continue
            if not handler.can_transform(result, metadata):
                continue

            result, handler_records = handler.transform(result, metadata)
            records.extend(handler_records or [])

        return result, records

    @staticmethod
    def _resolve_transformation(value: Any) -> ITransformation | None:
        if isinstance(value, ITransformation):
            return value
        if isinstance(value, type) and issubclass(value, ITransformation):
            return value()
        return None
