"""Version helpers for SeaSenseLib."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    tomllib = None


PACKAGE_NAME = "seasenselib"


def _load_project_table(pyproject_path: Path) -> Dict[str, str]:
    """Load the PEP 621 project table from pyproject.toml."""
    if tomllib is not None:
        with pyproject_path.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        return {
            key: value
            for key, value in project.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    project = {}
    in_project_section = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project_section = line == "[project]"
            continue
        if not in_project_section:
            continue

        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if key not in {"name", "version"}:
            continue
        if len(value) < 2 or value[0] not in {"'", '"'} or value[-1] != value[0]:
            continue
        project[key] = value[1:-1]

    return project


def _read_pyproject_version(pyproject_path: Path) -> Optional[str]:
    """Read the package version from a pyproject.toml file."""
    project = _load_project_table(pyproject_path)

    if project.get("name") != PACKAGE_NAME:
        return None

    version = project.get("version")
    if version is None:
        return None

    return str(version)


def _find_project_version() -> Optional[str]:
    """Find the nearest pyproject.toml that belongs to SeaSenseLib."""
    package_path = Path(__file__).resolve()
    for directory in package_path.parents:
        pyproject_path = directory / "pyproject.toml"
        if not pyproject_path.is_file():
            continue

        try:
            version = _read_pyproject_version(pyproject_path)
        except Exception:
            continue
        if version is not None:
            return version

    return None


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the current SeaSenseLib version."""
    project_version = _find_project_version()
    if project_version is not None:
        return project_version

    try:
        return package_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()


__all__ = ["PACKAGE_NAME", "__version__", "get_version"]
