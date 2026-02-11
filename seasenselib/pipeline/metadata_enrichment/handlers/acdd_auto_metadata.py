"""
Auto-generate conservative ACDD text fields (title, summary, keywords).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Iterable
import logging
import re
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from ...interfaces import IConvention, MetadataRegistry, ValidationError
from ....knowledge.loader import load_json

logger = logging.getLogger(__name__)


class AcddAutoMetadata(IConvention):
    """Generate title/summary/keywords when missing, conservatively."""

    def __init__(self) -> None:
        self._rules = self._load_rules()
        self._normalize_rules()

    def name(self) -> str:
        return "acdd_auto"

    def enrich(self, dataset: xr.Dataset, metadata_registry: MetadataRegistry) -> xr.Dataset:
        generated: List[str] = []

        if self._is_missing(dataset.attrs.get("title")):
            title = self._build_title(dataset)
            if title:
                dataset.attrs["title"] = title
                generated.append("title")

        if self._is_missing(dataset.attrs.get("summary")):
            summary = self._build_summary(dataset)
            if summary:
                dataset.attrs["summary"] = summary
                generated.append("summary")

        if self._is_missing(dataset.attrs.get("keywords")):
            keywords = self._build_keywords(dataset)
            if keywords:
                dataset.attrs["keywords"] = keywords
                generated.append("keywords")

        if generated:
            dataset.attrs["acdd_autogen_fields"] = ",".join(generated)
            logger.debug("Auto-generated ACDD fields: %s", ", ".join(generated))

        return dataset

    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        return []

    # ------------------------------------------------------------------
    # Title / Summary / Keywords
    # ------------------------------------------------------------------

    def _build_title(self, ds: xr.Dataset) -> Optional[str]:
        base = self._source_format_label(ds)
        if base:
            title = f"Level-1 dataset from {base}"
        else:
            title = "Level-1 oceanographic dataset"

        time_range = self._time_range(ds)
        if time_range:
            start, end = time_range
            if start[:10] == end[:10]:
                title += f" on {start[:10]}"
            else:
                title += f" between {start[:10]} and {end[:10]}"

        spatial = self._spatial_phrase(ds)
        if spatial:
            title += f" {spatial}"

        depth_var = self._find_depth(ds)
        depth_units = self._depth_unit_safe(depth_var) if depth_var is not None else ""
        depth = self._depth_range_text(ds, include_units=depth_units)
        if depth:
            title += f" (depth {depth})"

        return title

    def _build_summary(self, ds: xr.Dataset) -> Optional[str]:
        source = self._source_phrase(ds)
        parts: List[str] = []
        if source:
            parts.append(
                f"Level-1 dataset decoded from {source} with canonical variable names and units; "
                "RAW metadata preserved verbatim; no quality control applied."
            )
        else:
            parts.append(
                "Level-1 dataset decoded from a RAW sensor file with canonical variable names and units; "
                "RAW metadata preserved verbatim; no quality control applied."
            )

        coverage_parts: List[str] = []
        time_range = self._time_range(ds)
        if time_range:
            start, end = time_range
            if start == end:
                coverage_parts.append(f"Time coverage: {start}.")
            else:
                coverage_parts.append(f"Time coverage: {start} to {end}.")

        spatial = self._spatial_sentence(ds)
        if spatial:
            coverage_parts.append(spatial)

        depth_var = self._find_depth(ds)
        depth_units = self._depth_unit_safe(depth_var) if depth_var is not None else ""
        depth = self._depth_range_text(ds, include_units=depth_units)
        if depth:
            coverage_parts.append(f"Depth range: {depth}.")

        if coverage_parts:
            parts.append(" ".join(coverage_parts))

        variables = self._variable_list(ds, limit=8)
        if variables:
            parts.append(f"Variables include: {', '.join(variables)}.")

        return " ".join(parts)

    def _build_keywords(self, ds: xr.Dataset) -> Optional[str]:
        tokens: List[str] = list(self._rule_list("base_keywords"))

        format_tokens = self._format_tokens(ds)
        for token in sorted(format_tokens):
            if token not in tokens:
                tokens.append(token)

        var_tokens = self._variable_keyword_tokens(ds, limit=6)
        for token in var_tokens:
            if token not in tokens:
                tokens.append(token)

        return ", ".join(tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rules() -> Dict[str, Any]:
        try:
            return load_json("pipeline/metadata_enrichment/acdd_auto.json")
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Missing knowledge file: seasenselib/knowledge/pipeline/metadata_enrichment/acdd_auto.json"
            ) from exc

    def _normalize_rules(self) -> None:
        def _lower_list(values: Iterable[str]) -> List[str]:
            return [str(value).lower() for value in values]

        for key in (
            "aux_variable_suffixes",
            "aux_variable_substrings",
            "exclude_variable_names",
            "format_token_blacklist",
            "depth_unit_safe",
        ):
            if key in self._rules:
                self._rules[key] = _lower_list(self._rules[key])

        if "format_token_map" in self._rules:
            self._rules["format_token_map"] = {
                str(k).lower(): str(v).lower() for k, v in self._rules["format_token_map"].items()
            }

        if "keyword_standard_name_overrides" in self._rules:
            self._rules["keyword_standard_name_overrides"] = {
                str(k).lower(): str(v).lower() for k, v in self._rules["keyword_standard_name_overrides"].items()
            }

    def _rule_list(self, key: str) -> List[str]:
        value = self._rules.get(key)
        if not isinstance(value, list):
            raise RuntimeError(f"Invalid ACDD auto rules: '{key}' must be a list")
        return value

    def _rule_map(self, key: str) -> Dict[str, Any]:
        value = self._rules.get(key)
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid ACDD auto rules: '{key}' must be a mapping")
        return value

    def _rule_value(self, key: str) -> Any:
        if key not in self._rules:
            raise RuntimeError(f"Invalid ACDD auto rules: missing key '{key}'")
        return self._rules[key]

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple)):
            return len(value) == 0
        return False

    def _source_format_label(self, ds: xr.Dataset) -> Optional[str]:
        for key in self._rule_list("source_format_attr_keys"):
            value = ds.attrs.get(key)
            if value:
                return self._ensure_file_suffix(str(value).strip())
        raw_filename_attr = self._rule_value("raw_filename_attr")
        raw_filename = ds.attrs.get(raw_filename_attr)
        if raw_filename:
            return f"RAW file {raw_filename}"
        return None

    def _source_phrase(self, ds: xr.Dataset) -> Optional[str]:
        for key in self._rule_list("source_format_attr_keys"):
            value = ds.attrs.get(key)
            if value:
                return self._ensure_file_suffix(str(value).strip())
        raw_filename_attr = self._rule_value("raw_filename_attr")
        raw_filename = ds.attrs.get(raw_filename_attr)
        if raw_filename:
            return f"RAW file {raw_filename}"
        return None

    @staticmethod
    def _ensure_file_suffix(label: str) -> str:
        lowered = label.strip().lower()
        if lowered.startswith("raw file"):
            return label
        if "file" in lowered:
            return label
        return f"{label} file"

    def _time_range(self, ds: xr.Dataset) -> Optional[Tuple[str, str]]:
        coord_name = None
        for candidate in self._rule_list("time_coord_names"):
            if candidate in ds.coords:
                coord_name = candidate
                break
        if coord_name is None:
            return None
        values = ds.coords[coord_name].values
        bounds = self._datetime_min_max(values)
        if bounds is None:
            return None
        start, end = bounds
        return self._format_datetime(start), self._format_datetime(end)

    @staticmethod
    def _datetime_min_max(values: Any) -> Optional[Tuple[Any, Any]]:
        try:
            arr = np.asarray(values)
        except Exception:
            return None
        if arr.size == 0:
            return None

        if np.issubdtype(arr.dtype, np.datetime64):
            try:
                valid = arr.reshape(-1)
                valid = valid[~np.isnat(valid)]
            except Exception:
                valid = arr
            if getattr(valid, "size", 0) == 0:
                return None
            return valid.min(), valid.max()

        if arr.dtype == object:
            try:
                import pandas as pd
            except Exception:
                return None
            flat = arr.reshape(-1)
            series = pd.to_datetime(flat, errors="coerce", utc=False)
            if getattr(series, "size", 0) == 0:
                return None
            series = series.dropna()
            if series.empty:
                return None
            return series.min(), series.max()

        return None

    @staticmethod
    def _format_datetime(value: Any) -> str:
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit="s")
        if hasattr(value, "tzinfo"):
            if value.tzinfo is not None:
                utc_val = value.astimezone(timezone.utc)
                return utc_val.isoformat().replace("+00:00", "Z")
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def _find_coord(self, ds: xr.Dataset, names: List[str], standard_name: str) -> Optional[xr.DataArray]:
        for name in names:
            if name in ds.coords:
                return ds.coords[name]
        for name, coord in ds.coords.items():
            if coord.attrs.get("standard_name") == standard_name:
                return coord
        for name in names:
            if name in ds.data_vars:
                return ds.data_vars[name]
        for name, var in ds.data_vars.items():
            if var.attrs.get("standard_name") == standard_name:
                return var
        return None

    def _spatial_phrase(self, ds: xr.Dataset) -> Optional[str]:
        lat = self._find_coord(ds, self._rule_list("lat_names"), "latitude")
        lon = self._find_coord(ds, self._rule_list("lon_names"), "longitude")
        if lat is None or lon is None:
            return None
        lat_min, lat_max = self._numeric_min_max(lat.values)
        lon_min, lon_max = self._numeric_min_max(lon.values)
        if lat_min is None or lon_min is None:
            return None
        if lat_max == lat_min and lon_max == lon_min:
            return f"at {self._format_lat(lat_min)}, {self._format_lon(lon_min)}"
        return f"within {self._format_lat_range(lat_min, lat_max)}, {self._format_lon_range(lon_min, lon_max)}"

    def _spatial_sentence(self, ds: xr.Dataset) -> Optional[str]:
        lat = self._find_coord(ds, self._rule_list("lat_names"), "latitude")
        lon = self._find_coord(ds, self._rule_list("lon_names"), "longitude")
        if lat is None or lon is None:
            return None
        lat_min, lat_max = self._numeric_min_max(lat.values)
        lon_min, lon_max = self._numeric_min_max(lon.values)
        if lat_min is None or lon_min is None:
            return None
        if lat_max == lat_min and lon_max == lon_min:
            return f"Spatial coverage: {self._format_lat(lat_min)}, {self._format_lon(lon_min)}."
        return (
            f"Spatial coverage: {self._format_lat_range(lat_min, lat_max)}, "
            f"{self._format_lon_range(lon_min, lon_max)}."
        )

    def _depth_range_text(
        self,
        ds: xr.Dataset,
        include_units: Optional[str] = None,
    ) -> Optional[str]:
        depth = self._find_depth(ds)
        if depth is None:
            return None
        depth_min, depth_max = self._numeric_min_max(depth.values)
        if depth_min is None:
            return None
        unit = include_units if include_units is not None else self._depth_unit_safe(depth)
        if depth_max == depth_min:
            value = self._format_number(depth_min)
            return f"{value}{unit}".strip()
        return f"{self._format_number(depth_min)}–{self._format_number(depth_max)}{unit}".strip()

    def _find_depth(self, ds: xr.Dataset) -> Optional[xr.DataArray]:
        for name in self._rule_list("depth_names"):
            if name in ds.coords:
                return ds.coords[name]
            if name in ds.data_vars:
                return ds.data_vars[name]
        for name in self._rule_list("depth_alt_names"):
            if name in ds.coords and ds.coords[name].attrs.get("standard_name") == "depth":
                return ds.coords[name]
            if name in ds.data_vars and ds.data_vars[name].attrs.get("standard_name") == "depth":
                return ds.data_vars[name]
        return None

    def _depth_unit_safe(self, depth: Optional[xr.DataArray]) -> str:
        if depth is None:
            return ""
        unit = depth.attrs.get("units")
        if not unit:
            return ""
        normalized = str(unit).strip().lower()
        if normalized in set(self._rule_list("depth_unit_safe")):
            return " m"
        return ""

    @staticmethod
    def _numeric_min_max(values: Any) -> Tuple[Optional[float], Optional[float]]:
        try:
            arr = np.asarray(values, dtype=float)
        except Exception:
            return None, None
        if arr.size == 0:
            return None, None
        if np.isnan(arr).all():
            return None, None
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_lat(value: float) -> str:
        hemi = "N" if value >= 0 else "S"
        return f"{abs(value):.4f}{hemi}"

    @staticmethod
    def _format_lon(value: float) -> str:
        hemi = "E" if value >= 0 else "W"
        return f"{abs(value):.4f}{hemi}"

    @staticmethod
    def _format_lat_range(min_val: float, max_val: float) -> str:
        return f"{abs(min_val):.2f}–{abs(max_val):.2f}{'N' if max_val >= 0 else 'S'}"

    @staticmethod
    def _format_lon_range(min_val: float, max_val: float) -> str:
        return f"{abs(min_val):.2f}–{abs(max_val):.2f}{'E' if max_val >= 0 else 'W'}"

    def _variable_list(self, ds: xr.Dataset, limit: int) -> List[str]:
        names: List[str] = []
        for name in ds.data_vars:
            if self._is_aux_variable(name):
                continue
            base = self._base_var_name(name)
            if base not in names:
                names.append(base)

        names = sorted(names)
        if len(names) <= limit:
            return names
        return names[:limit] + [f"and {len(names) - limit} more"]

    def _variable_keyword_tokens(self, ds: xr.Dataset, limit: int) -> List[str]:
        tokens: List[str] = []
        for name in sorted(ds.data_vars):
            if self._is_aux_variable(name):
                continue
            var = ds.data_vars[name]
            base = self._base_var_name(name)
            token_source = var.attrs.get("standard_name") or base
            token = self._keyword_from_standard_name(token_source, fallback=base)
            if token and token not in tokens:
                tokens.append(token)
            if len(tokens) >= limit:
                break
        return tokens

    @staticmethod
    def _clean_keyword_token(value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _is_aux_variable(self, name: str) -> bool:
        lowered = name.lower()
        if lowered in set(self._rule_list("exclude_variable_names")):
            return True
        if any(lowered.endswith(suffix) for suffix in self._rule_list("aux_variable_suffixes")):
            return True
        if any(token in lowered for token in self._rule_list("aux_variable_substrings")):
            return True
        return False

    @staticmethod
    def _base_var_name(name: str) -> str:
        match = re.match(r"^(.+?)_(\d{1,3})$", name)
        return match.group(1) if match else name

    def _format_tokens(self, ds: xr.Dataset) -> List[str]:
        tokens: List[str] = []
        for key in self._rule_list("source_format_attr_keys"):
            fmt = ds.attrs.get(key)
            if fmt:
                for token in re.split(r"[^a-zA-Z0-9]+", str(fmt).lower()):
                    if token:
                        tokens.append(token)

        raw_filename_attr = self._rule_value("raw_filename_attr")
        filename = ds.attrs.get(raw_filename_attr)
        if filename and "." in filename:
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext:
                tokens.append(ext)
        mapping = self._rule_map("format_token_map")
        blacklist = set(self._rule_list("format_token_blacklist"))
        normalized = []
        for token in tokens:
            mapped = mapping.get(token, token)
            if mapped and mapped not in blacklist:
                normalized.append(mapped)
        return sorted(set(normalized))

    def _keyword_from_standard_name(self, value: str, fallback: str) -> str:
        text = str(value).strip().lower()
        for prefix in self._rule_list("keyword_prefix_strip"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        cleaned = self._clean_keyword_token(text)
        fallback_clean = self._clean_keyword_token(fallback)

        overrides = self._rule_map("keyword_standard_name_overrides")
        override = overrides.get(cleaned)
        if override:
            return self._clean_keyword_token(override)

        if fallback_clean and cleaned and fallback_clean in cleaned:
            return fallback_clean
        if cleaned.count("_") >= 2:
            return fallback_clean or cleaned
        if cleaned.startswith("volume_fraction_of_") or cleaned.startswith("mole_fraction_of_"):
            return fallback_clean or cleaned
        return cleaned or fallback_clean


__all__ = ["AcddAutoMetadata"]
