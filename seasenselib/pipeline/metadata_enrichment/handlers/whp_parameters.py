"""
WHP-Exchange parameter mapping handler.

Adds a whp_parameter attribute for variables with a known WHP code.
"""

from __future__ import annotations

from typing import Dict, Any, List
import re

import xarray as xr

from ...interfaces import IConvention, MetadataRegistry, ValidationError
from ....knowledge.loader import load_json


class WHPParameters(IConvention):
    """Apply minimal WHP parameter codes based on canonical variable names."""

    def __init__(self) -> None:
        self._rules = self._load_rules()

    def name(self) -> str:
        return "whp"

    def enrich(self, dataset: xr.Dataset, metadata_registry: MetadataRegistry) -> xr.Dataset:
        attribute_name = str(self._rules.get("attribute_name", "whp_parameter"))
        mapping = self._rule_map("parameter_map")
        apply_to = set(self._rule_list("apply_to"))
        allow_numbered = bool(self._rules.get("allow_numbered", True))
        exclude_names = set(self._rule_list("exclude_names"))
        exclude_suffixes = self._rule_list("exclude_suffixes")

        def iter_targets() -> List[str]:
            names: List[str] = []
            if "data_vars" in apply_to:
                names.extend(list(dataset.data_vars))
            if "coords" in apply_to:
                names.extend([n for n in dataset.coords if n not in names])
            return names

        for var_name in iter_targets():
            if not isinstance(var_name, str):
                continue
            if var_name in exclude_names:
                continue
            if any(var_name.endswith(suffix) for suffix in exclude_suffixes):
                continue

            target = dataset[var_name]
            if str(target.attrs.get(attribute_name, "")).strip():
                continue

            base_name = var_name
            if allow_numbered:
                match = re.match(r"^(.*)_\d+$", var_name)
                if match:
                    base_name = match.group(1)

            code = mapping.get(base_name)
            if code:
                target.attrs[attribute_name] = code

        return dataset

    def validate(self, dataset: xr.Dataset) -> List[ValidationError]:
        return []

    @staticmethod
    def _load_rules() -> Dict[str, Any]:
        try:
            return load_json("pipeline/metadata_enrichment/whp_parameters.json")
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Missing knowledge file: seasenselib/knowledge/pipeline/metadata_enrichment/whp_parameters.json"
            ) from exc

    def _rule_list(self, key: str) -> List[str]:
        value = self._rules.get(key)
        if not isinstance(value, list):
            raise RuntimeError(f"Invalid WHP rules: '{key}' must be a list")
        return [str(item) for item in value]

    def _rule_map(self, key: str) -> Dict[str, Any]:
        value = self._rules.get(key)
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid WHP rules: '{key}' must be a mapping")
        return {str(k): v for k, v in value.items()}
