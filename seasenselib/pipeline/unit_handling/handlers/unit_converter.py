"""
Unit converter.

Uses pint to convert data values to expected units when possible.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Tuple
import logging

import xarray as xr

try:
    import pint
    _HAS_PINT = True
except Exception:
    _HAS_PINT = False

from .utils import _base_name, get_unit_aliases, get_expected_units

logger = logging.getLogger(__name__)

class UnitConverter:
    """Convert numeric values to expected units using pint."""

    def __init__(
        self,
        use_pint: bool = True,
        expected_units: Optional[Dict[str, str]] = None,
        conversion_mode: str = "duplicate_keep_original",
        original_suffix: str = "_original",
    ):
        self.use_pint = use_pint and _HAS_PINT
        self.expected_units = expected_units or get_expected_units()
        self.conversion_mode = conversion_mode
        self.original_suffix = original_suffix
        self._ureg = pint.UnitRegistry() if self.use_pint else None
        self._aliases = get_unit_aliases()

    def convert(self, ds: xr.Dataset) -> Tuple[xr.Dataset, List[str]]:
        conversions: List[str] = []
        if not self.use_pint:
            return ds, conversions

        for var_name in ds.data_vars:
            var = ds[var_name]
            if 'units' not in var.attrs:
                continue

            expected = self.expected_units.get(_base_name(var_name))
            if not expected:
                continue

            original_units = var.attrs['units']
            from_unit = self._aliases.get(original_units, original_units)
            to_unit = self._aliases.get(expected, expected)
            if from_unit == to_unit:
                continue

            try:
                quantity = self._ureg.Quantity(var.values, from_unit)
                converted = quantity.to(to_unit).magnitude

                if self.conversion_mode == "duplicate_keep_original":
                    original_name = self._unique_name(ds, f"{var_name}{self.original_suffix}")
                    original_var = var.copy(deep=True)
                    self._append_comment(
                        original_var.attrs,
                        f"Original values before unit conversion to {expected} by SeaSenseLib"
                    )
                    ds[original_name] = original_var

                var.values[:] = converted
                var.attrs['units'] = expected
                var.attrs.setdefault('units_original', original_units)
                self._append_comment(var.attrs, f"Units converted from {original_units} to {expected} by SeaSenseLib")
                ds[var_name] = var
                conversions.append(f"{var_name}: {from_unit} -> {to_unit}")
                logger.debug("Converted units for '%s': %s -> %s", var_name, from_unit, to_unit)
            except Exception:
                # Conversion failed; leave as is
                logger.debug("Unit conversion failed for '%s': %s -> %s", var_name, from_unit, to_unit)
                continue

        return ds, conversions

    @staticmethod
    def _append_comment(attrs: Dict[str, str], note: str) -> None:
        """Append a note to the CF 'comment' attribute without duplication."""
        existing = attrs.get('comment')
        if existing:
            if note in existing:
                return
            attrs['comment'] = f"{existing}; {note}"
        else:
            attrs['comment'] = note

    @staticmethod
    def _unique_name(ds: xr.Dataset, base: str) -> str:
        """Return a unique variable name within the dataset."""
        if base not in ds:
            return base
        idx = 1
        while f"{base}_{idx}" in ds:
            idx += 1
        return f"{base}_{idx}"
