"""
Unit handling utilities.

Shared helpers for unit normalization/conversion/validation.
"""

from __future__ import annotations

from typing import Dict
import logging
import re

import seasenselib.parameters as params

logger = logging.getLogger(__name__)

def _base_name(name: str) -> str:
    match = re.match(r"^([a-zA-Z0-9_]+?)(?:_\d{1,2})?$", name)
    return match.group(1) if match else name


def get_expected_units() -> Dict[str, str]:
    """Get expected units for each parameter (conversion targets)."""
    try:
        from seasenselib.knowledge import load_json
        data = load_json("pipeline/unit_handling/expected_units.json")
        if isinstance(data, dict) and data:
            return data
    except Exception:
        logger.debug("Failed to load expected units from knowledge file", exc_info=True)

    expected: Dict[str, str] = {}
    for param_name, meta in params.metadata.items():
        if 'units' in meta:
            expected[param_name] = meta['units']
    return expected


def get_unit_normalizations() -> Dict[str, str]:
    """Get unit normalization mappings (variant -> preferred string)."""
    from seasenselib.knowledge import load_json
    data = load_json("pipeline/unit_handling/unit_normalizations.json")
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "Unit normalizations could not be loaded from "
            "pipeline/unit_handling/unit_normalizations.json"
        )
    return data


def get_unit_aliases() -> Dict[str, str]:
    """Get unit aliases for conversion engines like pint."""
    try:
        from seasenselib.knowledge import load_json
        data = load_json("pipeline/unit_handling/unit_aliases.json")
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("Failed to load unit aliases from knowledge file", exc_info=True)
    return {}


__all__ = [
    '_base_name',
    'get_expected_units',
    'get_unit_normalizations',
    'get_unit_aliases',
]
