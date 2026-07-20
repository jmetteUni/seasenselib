from pathlib import Path

import seasenselib as ssl
from seasenselib.cli.router import CLIRouter
from seasenselib._version import _read_pyproject_version


ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    version = _read_pyproject_version(ROOT / "pyproject.toml")
    assert version is not None
    return version


def test_get_version_reads_pyproject_version():
    expected = _pyproject_version()

    assert ssl.get_version() == expected
    assert ssl.__version__ == expected


def test_cli_version_flags_show_current_version(capsys):
    expected = f"seasenselib {_pyproject_version()}\n"
    router = CLIRouter()

    assert router.route_and_execute(["--version"]) == 0
    assert capsys.readouterr().out == expected

    assert router.route_and_execute(["-v"]) == 0
    assert capsys.readouterr().out == expected
