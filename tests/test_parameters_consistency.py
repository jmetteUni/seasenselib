"""
Consistency tests for canonical parameter model.
"""

from pathlib import Path
import json

import seasenselib.parameters as params


def _load_json(path: str):
    return json.loads(Path(path).read_text())


def _parameter_constants():
    return {
        value
        for name, value in params.__dict__.items()
        if name.isupper() and isinstance(value, str)
    }


def test_parameters_metadata_keys_are_constants():
    """All metadata keys must map to declared parameter constants."""
    meta = _load_json("seasenselib/knowledge/pipeline/metadata_enrichment/parameters_metadata.json")
    constants = _parameter_constants()
    missing = sorted([key for key in meta.keys() if key not in constants])
    assert not missing, f"Missing constants for metadata keys: {missing}"


def test_allowed_parameters_matches_metadata():
    """Allowed parameters must cover canonical model keys."""
    meta = _load_json("seasenselib/knowledge/pipeline/metadata_enrichment/parameters_metadata.json")
    defaults = _load_json("seasenselib/knowledge/pipeline/mapping/mappings_default.json")
    allowed = _load_json("seasenselib/knowledge/pipeline/mapping/allowed_parameters.json")
    assert set(allowed.keys()) == (set(meta.keys()) | set(defaults.keys()))


def test_default_mappings_keys_are_constants():
    """All default mapping keys must map to declared parameter constants."""
    defaults = _load_json("seasenselib/knowledge/pipeline/mapping/mappings_default.json")
    constants = _parameter_constants()
    missing = sorted([key for key in defaults.keys() if key not in constants])
    assert not missing, f"Missing constants for default mapping keys: {missing}"
