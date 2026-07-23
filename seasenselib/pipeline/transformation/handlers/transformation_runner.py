"""Transformation runner."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from ...base import StageContext
from ...interfaces import ITransformation, TransformationRecord
from ...utils import record_handler_applied


logger = logging.getLogger(__name__)


class TransformationRunner:
    """Run transformation handlers and record reproducible provenance."""

    def __init__(
        self,
        transformations: Optional[List[ITransformation]] = None,
        stage_name: str = "transformation",
    ):
        self.transformations = transformations or []
        self.stage_name = stage_name

    def process(self, context: StageContext) -> StageContext:
        ds = context.dataset
        records: List[Dict[str, Any]] = []

        for transformation in self.transformations:
            name = transformation.name()
            if not transformation.can_transform(ds, context.metadata):
                logger.debug("Skipping transformation '%s'", name)
                continue

            before_id = id(ds)
            ds, raw_records = transformation.transform(ds, context.metadata)
            normalized = [
                self._normalize_record(record, name)
                for record in raw_records or []
            ]
            if not normalized:
                continue

            records.extend(normalized)
            record_handler_applied(context.metadata, self.stage_name, name)
            logger.info(
                "Applied transformation '%s' with %d provenance record(s)",
                name,
                len(normalized),
            )
            if before_id == id(ds):
                logger.debug(
                    "Transformation '%s' returned the same Dataset object",
                    name,
                )

        if records:
            existing = context.metadata.setdefault("transformations", [])
            existing.extend(records)
            ds = self._annotate_dataset(ds, records)

        context.dataset = ds
        return context

    @staticmethod
    def _normalize_record(
        record: TransformationRecord | Dict[str, Any],
        handler_name: str,
    ) -> Dict[str, Any]:
        if isinstance(record, TransformationRecord):
            data = record.to_dict()
        elif isinstance(record, dict):
            data = dict(record)
        else:
            data = {
                "transformation": str(record),
                "description": "Transformation applied.",
            }

        data.setdefault("handler", handler_name)
        data.setdefault("transformation", handler_name)
        data.setdefault("description", "Transformation applied.")
        data.setdefault(
            "applied_at_utc",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        if "variables" in data:
            data["variables"] = [str(name) for name in data.get("variables") or []]
        return data

    def _annotate_dataset(
        self,
        ds: xr.Dataset,
        records: List[Dict[str, Any]],
    ) -> xr.Dataset:
        summary = [
            {
                key: value
                for key, value in record.items()
                if key in {
                    "handler",
                    "transformation",
                    "description",
                    "variables",
                    "parameters",
                }
            }
            for record in records
        ]
        ds.attrs["processor_transformations"] = self._json_dumps(summary)
        ds.attrs["processor_transformations_count"] = len(summary)

        by_variable: Dict[str, List[Dict[str, Any]]] = {}
        for record in summary:
            for variable_name in record.get("variables", []) or []:
                if variable_name in ds.data_vars or variable_name in ds.coords:
                    by_variable.setdefault(variable_name, []).append(record)

        for variable_name, variable_records in by_variable.items():
            ds[variable_name].attrs["processing_transformations"] = self._json_dumps(
                variable_records
            )

        return ds

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)
