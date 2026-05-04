"""
User metadata handler.

Applies user-provided metadata overrides to the dataset.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging

import xarray as xr

from ...interfaces import IConvention, MetadataRegistry, ValidationError
from ...utils import apply_user_metadata, normalize_user_metadata

logger = logging.getLogger(__name__)


class UserMetadataHandler(IConvention):
    """Apply user-specified metadata overrides."""

    def __init__(self):
        self._user_metadata: Optional[Dict[str, Any]] = None
        self._warnings: List[str] = []
        self._applied = False
        self._applied_global_keys: List[str] = []
        self._applied_variable_keys: Dict[str, List[str]] = {}

    def name(self) -> str:
        return "user_metadata"

    def set_user_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        self._user_metadata = metadata

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    @property
    def applied(self) -> bool:
        return self._applied

    @property
    def applied_global_keys(self) -> List[str]:
        return list(self._applied_global_keys)

    @property
    def applied_variable_keys(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._applied_variable_keys.items()}

    def enrich(self, dataset: xr.Dataset, metadata_registry: MetadataRegistry) -> xr.Dataset:
        if not self._user_metadata:
            self._warnings = []
            self._applied = False
            self._applied_global_keys = []
            self._applied_variable_keys = {}
            return dataset

        normalized = normalize_user_metadata(self._user_metadata)
        normalized, filtered_warnings = self._filter_reserved_metadata(normalized)
        self._applied_global_keys = list(normalized.get("global", {}).keys())
        self._applied_variable_keys = {
            name: list(attrs.keys()) if isinstance(attrs, dict) else []
            for name, attrs in normalized.get("variables", {}).items()
        }
        ds, warnings = apply_user_metadata(dataset, normalized, warn_missing=True)
        self._warnings = filtered_warnings + warnings
        self._applied = True
        if normalized.get("global"):
            logger.debug(
                "User metadata applied to globals: %s",
                ", ".join(sorted(normalized["global"].keys()))
            )
        if normalized.get("variables"):
            var_summaries = [
                f"{name}({len(attrs)})"
                for name, attrs in sorted(normalized["variables"].items())
                if isinstance(attrs, dict)
            ]
            if var_summaries:
                logger.debug(
                    "User metadata applied to variables: %s",
                    ", ".join(var_summaries)
                )
        for warning in warnings:
            logger.warning(warning)
        return ds

    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        return []

    @staticmethod
    def _filter_reserved_metadata(
        metadata: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        reserved_prefixes = ("raw_", "processor_")

        filtered = {
            "global": {},
            "variables": {},
        }

        for key, value in metadata.get("global", {}).items():
            if key.startswith(reserved_prefixes):
                warnings.append(
                    f"Ignored user metadata for reserved global attribute '{key}'"
                )
                continue
            filtered["global"][key] = value

        for var_name, attrs in metadata.get("variables", {}).items():
            if not isinstance(attrs, dict):
                filtered["variables"][var_name] = attrs
                continue
            new_attrs = {}
            for key, value in attrs.items():
                if key.startswith(reserved_prefixes):
                    warnings.append(
                        f"Ignored user metadata for reserved attribute '{var_name}.{key}'"
                    )
                    continue
                new_attrs[key] = value
            filtered["variables"][var_name] = new_attrs

        return filtered, warnings


__all__ = ["UserMetadataHandler"]
