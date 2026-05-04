"""
Derivation utilities.

Helpers for selecting numbered input variables and validating units.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re

import xarray as xr

from ...unit_handling.handlers.utils import get_unit_aliases, get_unit_normalizations
from ....knowledge.loader import load_json

_INPUT_UNITS: Optional[Dict[str, List[str]]] = None
_ALIASES: Optional[Dict[str, str]] = None
_NORMALIZATIONS: Optional[Dict[str, str]] = None


def get_input_units() -> Dict[str, List[str]]:
    global _INPUT_UNITS
    if _INPUT_UNITS is None:
        try:
            data = load_json("pipeline/derivation/input_units.json")
            if isinstance(data, dict):
                _INPUT_UNITS = {
                    str(k): [str(v).lower() for v in values]
                    for k, values in data.items()
                    if isinstance(values, list)
                }
            else:
                _INPUT_UNITS = {}
        except Exception:
            _INPUT_UNITS = {}
    return _INPUT_UNITS


def list_variants(dataset: xr.Dataset, base: str) -> List[str]:
    """Return available variable variants like base, base_1, base_2 (sorted)."""
    names: List[str] = []
    if base in dataset.data_vars:
        names.append(base)
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")
    numbered: List[Tuple[int, str]] = []
    for name in dataset.data_vars:
        match = pattern.match(name)
        if match:
            numbered.append((int(match.group(1)), name))
    for _, name in sorted(numbered, key=lambda item: item[0]):
        names.append(name)
    return names


def pick_first_variant(dataset: xr.Dataset, base: str) -> Tuple[Optional[str], int]:
    variants = list_variants(dataset, base)
    if not variants:
        return None, 0
    return variants[0], len(variants)


def output_name_from_input(base_output: str, input_name: str, base_input: str) -> str:
    if input_name == base_input:
        return base_output
    suffix = input_name[len(base_input) + 1:]
    return f"{base_output}_{suffix}"


def canonical_unit(unit: str) -> str:
    global _ALIASES, _NORMALIZATIONS
    if _ALIASES is None:
        _ALIASES = get_unit_aliases()
    if _NORMALIZATIONS is None:
        _NORMALIZATIONS = get_unit_normalizations()
    raw = str(unit).strip()
    aliased = _ALIASES.get(raw, raw)
    normalized = _NORMALIZATIONS.get(aliased, aliased)
    return str(normalized).strip().lower()


def units_ok(dataset: xr.Dataset, var_name: str, base: str) -> bool:
    allowed = get_input_units().get(base)
    if not allowed:
        return True
    if var_name not in dataset:
        return False
    unit = dataset[var_name].attrs.get("units")
    if not unit:
        return False
    return canonical_unit(unit) in allowed


__all__ = [
    "get_input_units",
    "list_variants",
    "pick_first_variant",
    "output_name_from_input",
    "canonical_unit",
    "units_ok",
]
