"""
RAW metadata handler.

Moves raw-format-specific attributes into raw_* attributes and
adds a structured RAW metadata container.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import hashlib
import json
import logging

from ...base import StageContext

logger = logging.getLogger(__name__)


@dataclass
class RawMetadata:
    """Add RAW metadata attributes and container."""

    schema: str = "seasenselib/raw-opaque-1.0"

    def configure(self, config: Dict[str, Any]) -> None:
        if isinstance(config, dict) and config.get("schema"):
            self.schema = str(config["schema"])

    def process(self, context: StageContext) -> StageContext:
        ds = context.dataset
        metadata = context.metadata

        source_file = metadata.get("source_file")
        raw_format = metadata.get("format_key") or metadata.get("format_name")
        raw_filename = None
        if source_file:
            raw_filename = Path(source_file).name

        # Move reader-specific attributes into RAW container
        extracted_globals: Dict[str, Any] = {}
        extracted_vars: Dict[str, Dict[str, Any]] = {}

        user_globals = set(metadata.get("user_metadata_global_keys", []) or [])
        protected_globals = set(self._protected_global_keys())
        protected_globals.update(user_globals)

        raw_prefixes = (
            "cnv_",
            "rsk_",
            "adcp_",
            "rdadcp_",
            "uhhds_",
            "nortek_",
            "rcm_",
            "rbr_",
            "sbe_",
            "seasun_",
            "tob_",
        )

        # Move global attrs that are reader-specific or non-standard
        for key in list(ds.attrs.keys()):
            if key.startswith(raw_prefixes):
                extracted_globals[key] = ds.attrs.pop(key)
                continue
            if key in protected_globals or key.startswith("raw_") or key.startswith("processor_"):
                continue
            extracted_globals[key] = ds.attrs.pop(key)

        # Keep variable-level reader-specific attributes on the variables.
        # Raw metadata focuses on global/raw container content only.

        # File-based raw attributes
        if raw_format and "raw_format" not in ds.attrs:
            ds.attrs["raw_format"] = raw_format
        if raw_filename and "raw_filename" not in ds.attrs:
            ds.attrs["raw_filename"] = raw_filename

        if source_file:
            try:
                path = Path(source_file)
                if path.exists():
                    if "raw_filesize_bytes" not in ds.attrs:
                        ds.attrs["raw_filesize_bytes"] = int(path.stat().st_size)
                    if "raw_mtime_utc" not in ds.attrs:
                        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                        ds.attrs["raw_mtime_utc"] = mtime.isoformat().replace("+00:00", "Z")
                    if "raw_sha256" not in ds.attrs:
                        ds.attrs["raw_sha256"] = self._sha256(path)
            except Exception as exc:
                logger.debug("Failed to compute raw file attributes: %s", exc)

        raw_header = metadata.get("raw_header")

        if "raw_metadata_schema" not in ds.attrs:
            ds.attrs["raw_metadata_schema"] = self.schema

        # Build RAW metadata container (opaque JSON)
        raw_container: Dict[str, Any] = {
            "schema": self.schema,
            "raw_format": raw_format or "",
            "raw_filename": raw_filename or "",
            "blocks": {
                "header": raw_header or None,
                "calibration": None,
                "configuration": None,
                "other": None,
            },
        }

        if extracted_globals or extracted_vars:
            raw_container["blocks"]["other"] = {
                "global_attributes": extracted_globals,
                "variables": extracted_vars,
            }

        if "raw_metadata" not in ds.attrs:
            try:
                ds.attrs["raw_metadata"] = json.dumps(
                    raw_container,
                    ensure_ascii=False,
                    default=self._json_default,
                )
            except Exception:
                ds.attrs["raw_metadata"] = json.dumps(
                    raw_container,
                    default=self._json_default,
                )

        context.dataset = ds
        logger.debug("Added RAW metadata attributes")
        return context

    @staticmethod
    def _protected_global_keys() -> list[str]:
        keys: list[str] = [
            "Conventions",
            "history",
            "date_created",
            "date_modified",
            "featureType",
            "cdm_data_type",
            "processing_level",
            "standard_name_vocabulary",
            "raw_metadata",
            "raw_metadata_schema",
            "raw_format",
            "raw_filename",
            "raw_sha256",
            "raw_filesize_bytes",
            "raw_mtime_utc",
        ]
        try:
            from seasenselib.knowledge import load_json
            data = load_json("pipeline/metadata_enrichment/acdd.json")
            if isinstance(data, dict):
                keys.extend(data.get("required_global_attributes", []) or [])
                keys.extend(data.get("recommended_global_attributes", []) or [])
                defaults = data.get("defaults", {})
                if isinstance(defaults, dict):
                    keys.extend(defaults.keys())
            raw_policy = load_json("pipeline/finalization/raw_metadata.json")
            if isinstance(raw_policy, dict):
                keys.extend(raw_policy.get("preserve_global_attributes", []) or [])
        except Exception:
            logger.debug("Failed to load protected global keys from knowledge files", exc_info=True)
        return sorted(set(keys))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json_default(value: Any) -> Any:
        try:
            import numpy as np
        except Exception:
            np = None

        if np is not None:
            if isinstance(value, np.datetime64):
                if np.isnat(value):
                    return None
                return np.datetime_as_string(value, unit="s")
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.datetime64):
                    if value.size == 0:
                        return []
                    flat = value.reshape(-1)
                    return [
                        None if np.isnat(item) else np.datetime_as_string(item, unit="s")
                        for item in flat
                    ]
                return value.tolist()
            if isinstance(value, np.generic):
                try:
                    return value.item()
                except Exception:
                    return str(value)

        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return list(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.hex()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                logger.debug("Failed to serialize isoformat value", exc_info=True)
                return str(value)
        return str(value)


__all__ = ["RawMetadata"]
