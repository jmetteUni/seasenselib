"""
ACDD Conventions implementation.

Provides ACDD-compliant global metadata enrichment and validation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import xarray as xr
import seasenselib.parameters as params
from ...interfaces import IConvention, MetadataRegistry, ValidationError

logger = logging.getLogger(__name__)


class ACDDConvention(IConvention):
    """
    Attribute Convention for Data Discovery (ACDD) implementation.

    Focuses on global attributes that improve data discovery and usability.
    The implementation is conservative: it only fills attributes when
    values are known or can be safely derived.
    """

    DEFAULT_RECOMMENDED = [
        'title',
        'summary',
        'keywords',
        'Conventions',
        'id',
        'naming_authority',
        'history',
        'source',
        'processing_level',
        'comment',
        'acknowledgment',
        'license',
        'standard_name_vocabulary',
        'date_created',
        'creator_name',
        'creator_email',
        'creator_url',
        'institution',
        'project',
        'publisher_name',
        'publisher_email',
        'publisher_url',
        'geospatial_lat_min',
        'geospatial_lat_max',
        'geospatial_lon_min',
        'geospatial_lon_max',
        'geospatial_vertical_min',
        'geospatial_vertical_max',
        'geospatial_vertical_positive',
        'time_coverage_start',
        'time_coverage_end',
        'time_coverage_duration',
        'time_coverage_resolution',
    ]

    def __init__(self, version: str = "1.3"):
        self.version = version
        self.required_global_attributes: List[str] = []
        self.recommended_global_attributes: List[str] = list(self.DEFAULT_RECOMMENDED)
        self.defaults: Dict[str, Any] = {}
        self.auto_geospatial = True
        self.auto_time_coverage = True

        self._load_knowledge()

    def name(self) -> str:
        return f"ACDD-{self.version}"

    def enrich(self, dataset: xr.Dataset, metadata_registry: Optional[MetadataRegistry]) -> xr.Dataset:
        """Enrich dataset with ACDD global attributes."""
        added: List[str] = []

        def set_global_attr(key: str, value: Any) -> None:
            if value is None or value == "":
                return
            if key in dataset.attrs:
                return
            dataset.attrs[key] = value
            added.append(key)

        # Ensure ACDD is listed in Conventions
        conventions_before = dataset.attrs.get('Conventions', '')
        dataset.attrs['Conventions'] = self._update_conventions(conventions_before)
        if dataset.attrs['Conventions'] != conventions_before:
            added.append('Conventions')

        # Use extracted metadata when available
        if metadata_registry is not None:
            for attr in self._acdd_attribute_names():
                if attr not in dataset.attrs:
                    value = self._registry_value(metadata_registry, attr)
                    if value is not None:
                        set_global_attr(attr, value)

        # Apply defaults (only if non-empty and not already present)
        for key, value in self.defaults.items():
            set_global_attr(key, value)

        # Derive geospatial coverage from coordinates when safe
        if self.auto_geospatial:
            before = set(dataset.attrs.keys())
            self._fill_geospatial(dataset)
            added.extend(sorted(set(dataset.attrs.keys()) - before))

        # Derive time coverage from time coordinate when safe
        if self.auto_time_coverage:
            before = set(dataset.attrs.keys())
            self._fill_time_coverage(dataset)
            added.extend(sorted(set(dataset.attrs.keys()) - before))

        # Infer feature type / cdm data type (very conservative)
        before = set(dataset.attrs.keys())
        self._infer_feature_type(dataset)
        added.extend(sorted(set(dataset.attrs.keys()) - before))

        logger.info("Applied ACDD conventions")
        if added:
            logger.debug("ACDD added global attributes: %s", ", ".join(sorted(set(added))))
        return dataset

    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        """Validate ACDD compliance (lightweight)."""
        errors: List[ValidationError] = []

        # Required attributes (if configured)
        for attr in self.required_global_attributes:
            if attr not in dataset.attrs:
                errors.append(ValidationError(
                    f"Missing required global attribute: {attr}",
                    severity="warning"
                ))

        # Recommended attributes
        for attr in self.recommended_global_attributes:
            if attr not in dataset.attrs:
                errors.append(ValidationError(
                    f"Missing recommended global attribute: {attr}",
                    severity="info"
                ))

        return errors

    def _load_knowledge(self) -> None:
        try:
            from seasenselib.knowledge import load_json
            data = load_json("pipeline/metadata_enrichment/acdd.json")
            if isinstance(data, dict):
                self.version = data.get('version', self.version)
                self.required_global_attributes = data.get(
                    'required_global_attributes',
                    self.required_global_attributes
                )
                self.recommended_global_attributes = data.get(
                    'recommended_global_attributes',
                    self.recommended_global_attributes
                )
                defaults = data.get('defaults')
                if isinstance(defaults, dict):
                    self.defaults = defaults
                auto = data.get('auto_fill', {})
                if isinstance(auto, dict):
                    if 'geospatial' in auto:
                        self.auto_geospatial = bool(auto['geospatial'])
                    if 'time_coverage' in auto:
                        self.auto_time_coverage = bool(auto['time_coverage'])
        except Exception:
            pass

    def _acdd_attribute_names(self) -> List[str]:
        names = set(self.required_global_attributes) | set(self.recommended_global_attributes)
        names.update(self.defaults.keys())
        return sorted(names)

    @staticmethod
    def _registry_value(registry: MetadataRegistry, attr: str) -> Any:
        value = registry.get(f"acdd.{attr}")
        if value is None:
            value = registry.get(f"global.{attr}")
        return value

    def _fill_geospatial(self, dataset: xr.Dataset) -> None:
        lat = self._get_dataarray(dataset, [params.LATITUDE])
        lon = self._get_dataarray(dataset, [params.LONGITUDE])
        vert = self._get_dataarray(dataset, [params.DEPTH, params.PRESSURE])

        if lat is not None:
            lat_min, lat_max = self._numeric_min_max(lat.values)
            if lat_min is not None and 'geospatial_lat_min' not in dataset.attrs:
                dataset.attrs['geospatial_lat_min'] = lat_min
            if lat_max is not None and 'geospatial_lat_max' not in dataset.attrs:
                dataset.attrs['geospatial_lat_max'] = lat_max

        if lon is not None:
            lon_min, lon_max = self._numeric_min_max(lon.values)
            if lon_min is not None and 'geospatial_lon_min' not in dataset.attrs:
                dataset.attrs['geospatial_lon_min'] = lon_min
            if lon_max is not None and 'geospatial_lon_max' not in dataset.attrs:
                dataset.attrs['geospatial_lon_max'] = lon_max

        if vert is not None:
            vert_min, vert_max = self._numeric_min_max(vert.values)
            if vert_min is not None and 'geospatial_vertical_min' not in dataset.attrs:
                dataset.attrs['geospatial_vertical_min'] = vert_min
            if vert_max is not None and 'geospatial_vertical_max' not in dataset.attrs:
                dataset.attrs['geospatial_vertical_max'] = vert_max
            if 'geospatial_vertical_positive' not in dataset.attrs:
                positive = self._infer_vertical_positive(dataset)
                if positive:
                    dataset.attrs['geospatial_vertical_positive'] = positive

    def _fill_time_coverage(self, dataset: xr.Dataset) -> None:
        time_var = self._get_dataarray(dataset, [params.TIME])
        if time_var is None:
            return

        bounds = self._datetime_min_max(time_var.values)
        if bounds is None:
            return

        start, end = bounds
        if start is not None and 'time_coverage_start' not in dataset.attrs:
            dataset.attrs['time_coverage_start'] = self._format_datetime(start)
        if end is not None and 'time_coverage_end' not in dataset.attrs:
            dataset.attrs['time_coverage_end'] = self._format_datetime(end)

        if start is not None and end is not None:
            if 'time_coverage_duration' not in dataset.attrs:
                duration = end - start
                seconds = self._timedelta_seconds(duration)
                if seconds is not None:
                    dataset.attrs['time_coverage_duration'] = self._format_duration(seconds)

            if 'time_coverage_resolution' not in dataset.attrs:
                resolution = self._infer_time_resolution(time_var.values)
                if resolution is not None:
                    dataset.attrs['time_coverage_resolution'] = self._format_duration(resolution)

    @staticmethod
    def _get_dataarray(dataset: xr.Dataset, names: List[str]) -> Optional[xr.DataArray]:
        for name in names:
            if name in dataset.coords:
                return dataset.coords[name]
            if name in dataset.data_vars:
                return dataset.data_vars[name]
        return None

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
    def _datetime_min_max(values: Any) -> Optional[Tuple[np.datetime64, np.datetime64]]:
        try:
            arr = np.asarray(values)
        except Exception:
            return None

        if arr.size == 0:
            return None

        if np.issubdtype(arr.dtype, np.datetime64):
            try:
                valid = arr[~np.isnat(arr)]
            except Exception:
                valid = arr
            if getattr(valid, "size", 0) == 0:
                return None
            return (valid.min(), valid.max())

        if arr.dtype == object:
            try:
                flat = [v for v in arr.ravel() if v is not None]
                if not flat:
                    return None
                dt64 = np.array(flat, dtype='datetime64[ns]')
                return (dt64.min(), dt64.max())
            except Exception:
                return None

        return None

    @staticmethod
    def _format_datetime(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit='s')
        if hasattr(value, 'isoformat'):
            try:
                return value.isoformat()
            except Exception:
                return None
        return None

    @staticmethod
    def _timedelta_seconds(value: Any) -> Optional[float]:
        try:
            delta = np.timedelta64(value)
            seconds = delta / np.timedelta64(1, 's')
            return float(seconds)
        except Exception:
            return None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            seconds = abs(seconds)
        whole = int(round(seconds))
        return f"PT{whole}S"

    @staticmethod
    def _infer_time_resolution(values: Any) -> Optional[float]:
        try:
            arr = np.asarray(values)
        except Exception:
            return None
        if arr.size < 2:
            return None

        if not np.issubdtype(arr.dtype, np.datetime64):
            try:
                arr = np.array(arr, dtype='datetime64[ns]')
            except Exception:
                return None

        try:
            flat = arr.reshape(-1)
            flat = flat[~np.isnat(flat)]
        except Exception:
            flat = arr

        if flat.size < 2:
            return None

        flat = np.sort(flat)
        diffs = np.diff(flat).astype('timedelta64[ns]')
        if diffs.size == 0:
            return None
        seconds = diffs / np.timedelta64(1, 's')
        seconds = seconds[seconds > 0]
        if seconds.size == 0:
            return None
        return float(np.median(seconds))

    @staticmethod
    def _infer_vertical_positive(dataset: xr.Dataset) -> Optional[str]:
        candidates = []
        if params.DEPTH in dataset.coords:
            candidates.append(dataset.coords[params.DEPTH])
        if params.DEPTH in dataset.data_vars:
            candidates.append(dataset.data_vars[params.DEPTH])
        if params.PRESSURE in dataset.coords:
            candidates.append(dataset.coords[params.PRESSURE])
        if params.PRESSURE in dataset.data_vars:
            candidates.append(dataset.data_vars[params.PRESSURE])

        for candidate in candidates:
            if isinstance(candidate, xr.DataArray):
                positive = candidate.attrs.get('positive')
                if positive:
                    return str(positive)

        if params.DEPTH in dataset.coords or params.DEPTH in dataset.data_vars:
            return "down"
        if params.PRESSURE in dataset.coords or params.PRESSURE in dataset.data_vars:
            return "down"
        return None

    def _update_conventions(self, existing: str) -> str:
        conventions = set(v.strip() for v in existing.split(',') if v.strip())
        conventions.add(f"ACDD-{self.version}")
        return ', '.join(sorted(conventions))

    @staticmethod
    def _infer_feature_type(dataset: xr.Dataset) -> None:
        if 'featureType' in dataset.attrs or 'cdm_data_type' in dataset.attrs:
            return
        dims = set(dataset.dims)
        if dims == {params.TIME}:
            dataset.attrs['featureType'] = 'timeSeries'
            dataset.attrs['cdm_data_type'] = 'TimeSeries'


__all__ = ["ACDDConvention"]
