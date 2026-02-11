"""
Shared stage utilities.
"""

from __future__ import annotations

from typing import Dict, List, Type, TypeVar, Tuple, Any

import xarray as xr

T = TypeVar("T")


def resolve_components(names: object, mapping: Dict[str, Type[T]]) -> List[T]:
    """
    Resolve handler names to instantiated components.

    Parameters
    ----------
    names : object
        Expected to be a list of handler names.
    mapping : Dict[str, Type[T]]
        Map from handler name to class.
    """
    if not isinstance(names, list) or not names:
        return []

    components: List[T] = []
    for name in names:
        cls = mapping.get(name)
        if cls is not None:
            components.append(cls())
    return components


def normalize_user_metadata(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Validate and normalize user-provided metadata."""
    if not isinstance(metadata, dict):
        raise ValueError("User metadata must be a dictionary")

    allowed_keys = {"global", "variables"}
    extra_keys = set(metadata.keys()) - allowed_keys
    if extra_keys:
        raise ValueError(
            "User metadata must contain only 'global' and 'variables' sections"
        )

    global_meta = metadata.get("global", {}) or {}
    variables_meta = metadata.get("variables", {}) or {}

    if not isinstance(global_meta, dict):
        raise ValueError("User metadata 'global' section must be a dictionary")
    if not isinstance(variables_meta, dict):
        raise ValueError("User metadata 'variables' section must be a dictionary")

    return {
        "global": dict(global_meta),
        "variables": dict(variables_meta),
    }


def merge_user_metadata(
    base: Dict[str, Any] | None,
    override: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]] | None:
    """Merge two user metadata dictionaries (override wins)."""
    if base is None and override is None:
        return None
    if base is None:
        return normalize_user_metadata(override or {})
    if override is None:
        return normalize_user_metadata(base)

    base_norm = normalize_user_metadata(base)
    override_norm = normalize_user_metadata(override)

    merged = {
        "global": dict(base_norm.get("global", {})),
        "variables": dict(base_norm.get("variables", {})),
    }

    merged["global"].update(override_norm.get("global", {}))

    for var_name, attrs in override_norm.get("variables", {}).items():
        existing = merged["variables"].get(var_name, {})
        if isinstance(existing, dict) and isinstance(attrs, dict):
            new_attrs = dict(existing)
            new_attrs.update(attrs)
            merged["variables"][var_name] = new_attrs
        else:
            merged["variables"][var_name] = attrs

    return merged


def apply_user_metadata(
    ds: xr.Dataset,
    metadata: Dict[str, Any],
    apply_globals: bool = True,
    apply_variables: bool = True,
    warn_missing: bool = True,
) -> Tuple[xr.Dataset, List[str]]:
    """Apply user metadata to a dataset."""
    warnings: List[str] = []
    meta = normalize_user_metadata(metadata)

    if apply_globals:
        for key, value in meta.get("global", {}).items():
            if value is None:
                continue
            ds.attrs[key] = value

    if apply_variables:
        for var_name, attrs in meta.get("variables", {}).items():
            if not isinstance(attrs, dict):
                raise ValueError(
                    f"User metadata for variable '{var_name}' must be a dictionary"
                )
            target = None
            if var_name in ds.data_vars:
                target = ds[var_name]
            elif var_name in ds.coords:
                target = ds.coords[var_name]
            else:
                if warn_missing:
                    warnings.append(
                        f"User metadata refers to unknown variable '{var_name}'"
                    )
                continue
            for key, value in attrs.items():
                if value is None:
                    continue
                target.attrs[key] = value

    return ds, warnings


def parse_handler_selectors(value: Any) -> Dict[str, List[str]]:
    """
    Parse handler selectors in the form "stage:handler".

    Accepts:
    - comma-separated string
    - list of "stage:handler" strings
    - dict {"stage": [handlers]}
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        parsed: Dict[str, List[str]] = {}
        for stage, handlers in value.items():
            if not isinstance(stage, str):
                raise ValueError("Handler selector stage must be a string")
            if isinstance(handlers, str):
                handler_list = [handlers]
            elif isinstance(handlers, list):
                handler_list = [str(h) for h in handlers]
            else:
                raise ValueError("Handler selector handlers must be a list or string")
            parsed[stage] = handler_list
        return parsed

    if isinstance(value, str):
        items = [v.strip() for v in value.split(',') if v.strip()]
    elif isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        raise ValueError("Handler selectors must be a string, list, or dict")

    result: Dict[str, List[str]] = {}
    for item in items:
        if ':' not in item:
            raise ValueError(f"Invalid handler selector: '{item}'. Use stage:handler")
        stage, handler = item.split(':', 1)
        stage = stage.strip()
        handler = handler.strip()
        if not stage or not handler:
            raise ValueError(f"Invalid handler selector: '{item}'. Use stage:handler")
        result.setdefault(stage, []).append(handler)
    return result


def apply_handler_filters(
    config: Any,
    apply_map: Dict[str, List[str]] | None,
    skip_map: Dict[str, List[str]] | None,
) -> Any:
    """Apply handler include/exclude filters to a PipelineConfig."""
    if not apply_map and not skip_map:
        return config

    from seasenselib.pipeline.handler_catalog import BUILTIN_HANDLERS

    stage_key = {
        'validation': 'validators',
    }

    stages = {stage.name: stage for stage in config.pipeline}
    target_stages = set()
    if apply_map:
        target_stages.update(apply_map.keys())
    if skip_map:
        target_stages.update(skip_map.keys())

    for stage_name in target_stages:
        if stage_name not in stages:
            raise ValueError(f"Unknown stage in handler filter: {stage_name}")

        stage_cfg = stages[stage_name]
        key = stage_key.get(stage_name, 'handlers')
        handlers = stage_cfg.config.get(key)
        if handlers is None:
            handlers = list(BUILTIN_HANDLERS.get(stage_name, {}).keys())

        if apply_map and stage_name in apply_map:
            handlers = list(apply_map[stage_name])

        if skip_map and stage_name in skip_map:
            skip = set(skip_map[stage_name])
            handlers = [h for h in handlers if h not in skip]

        stage_cfg.config[key] = handlers

    return config


def record_handler_applied(metadata: Dict[str, Any], stage: str, handler: str) -> None:
    """Record an applied handler in the processing metadata."""
    if not stage or not handler:
        return
    metadata.setdefault("handlers_applied", []).append(f"{stage}:{handler}")


__all__ = [
    "resolve_components",
    "normalize_user_metadata",
    "merge_user_metadata",
    "apply_user_metadata",
    "parse_handler_selectors",
    "apply_handler_filters",
    "record_handler_applied",
]
