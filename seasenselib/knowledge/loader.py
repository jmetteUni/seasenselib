"""
Knowledge loader utilities.

Provides a minimal, decoupled loader so each stage can load the
specific knowledge files it needs.
"""

from __future__ import annotations
from typing import Any, Dict
from importlib import resources
import json


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON knowledge file from package resources.

    Parameters
    ----------
    path : str
        Relative path under seasenselib/knowledge, e.g. "pipeline/mapping/mappings_default.json".
    """
    with resources.files("seasenselib.knowledge").joinpath(path).open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["load_json"]
