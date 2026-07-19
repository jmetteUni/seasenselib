from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import seasenselib as ssl
from seasenselib.readers import get_reader_by_format_key, get_readers_by_extension
from seasenselib.readers.rdi_raw_reader import RdiRawReader


def _write_raw_file(tmp_path, suffix=".000"):
    raw_file = tmp_path / f"sample{suffix}"
    raw_file.write_bytes(b"\x7f\x7f\x00\x00")
    return raw_file


def _dolfyn_like_dataset(coord_sys="earth"):
    return xr.Dataset(
        {
            "vel": (
                ("dir", "range", "time"),
                np.zeros((4, 2, 3), dtype=float),
            ),
            "temp": ("time", np.array([7.1, 7.2, 7.3])),
            "c_sound": ("time", np.array([1450.0, 1450.1, 1450.2])),
            "pressure": ("time", np.array([100.0, 100.1, 100.2])),
        },
        coords={
            "dir": ("dir", ["east", "north", "up", "err_vel"]),
            "range": ("range", [1.0, 2.0]),
            "time": (
                "time",
                np.array(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-01T00:01:00",
                        "2026-01-01T00:02:00",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
        },
        attrs={
            "coord_sys": coord_sys,
            "inst_model": "Workhorse",
            "serialnum": 12345,
        },
    )


def _install_fake_mhkit(monkeypatch, read_rdi):
    mhkit = ModuleType("mhkit")
    mhkit.dolfyn = SimpleNamespace(
        io=SimpleNamespace(rdi=SimpleNamespace(read_rdi=read_rdi))
    )
    monkeypatch.setitem(sys.modules, "mhkit", mhkit)


def test_rdi_raw_reader_exposes_format_metadata():
    assert RdiRawReader.format_key() == "rdi-raw"
    assert RdiRawReader.format_name() == "RDI ADCP raw"
    assert RdiRawReader.file_extension() == ".000"
    assert RdiRawReader.file_extensions() == (
        ".000",
        ".pd0",
        ".enr",
        ".ens",
        ".enx",
    )
    assert RdiRawReader._get_valid_extensions() == (
        ".000",
        ".pd0",
        ".enr",
        ".ens",
        ".enx",
    )


def test_rdi_raw_reader_is_discoverable_by_format_key():
    assert get_reader_by_format_key("rdi-raw") is RdiRawReader


def test_rdi_raw_reader_is_discoverable_by_each_extension():
    assert get_readers_by_extension(".000") == [RdiRawReader]
    assert get_readers_by_extension(".PD0") == [RdiRawReader]
    assert get_readers_by_extension("enr") == [RdiRawReader]


def test_rdi_raw_reader_wraps_mhkit_dolfyn(monkeypatch, tmp_path):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_rdi(
        filename,
        userdata=None,
        nens=None,
        debug=None,
        vmdas_search=False,
        winriver=False,
        search_num=None,
    ):
        calls.append(
            {
                "filename": filename,
                "userdata": userdata,
                "nens": nens,
                "debug": debug,
                "vmdas_search": vmdas_search,
                "winriver": winriver,
                "search_num": search_num,
            }
        )
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    reader = RdiRawReader(
        str(raw_file),
        userdata=False,
        nens=2,
        debug=1,
        vmdas_search=True,
        winriver=True,
        search_num=500,
        perform_default_postprocessing=False,
    )
    ds = reader.data

    assert ds is expected
    assert calls == [
        {
            "filename": str(raw_file),
            "userdata": False,
            "nens": 2,
            "debug": 1,
            "vmdas_search": True,
            "winriver": True,
            "search_num": 500,
        }
    ]
    assert ds.attrs["raw_data_reader"] == "mhkit.dolfyn.io.rdi.read_rdi"
    assert "vel" in ds
    assert "east_velocity" not in ds
    assert ds["temp"].attrs["original_name"] == "temp"
    assert ds["temp"].attrs["original_units"] == "degree_C"
    assert ds["temp"].attrs["standard_name"] == "sea_water_temperature"
    assert ds["c_sound"].attrs["original_name"] == "c_sound"
    assert ds["c_sound"].attrs["original_units"] == "m s-1"
    assert ds["c_sound"].attrs["standard_name"] == "speed_of_sound_in_sea_water"
    assert reader._raw_metadata_variables["vel"]["dims"] == [
        "dir",
        "range",
        "time",
    ]
    assert reader._raw_metadata_variables["temp"]["original_name"] == "temp"
    assert reader._raw_metadata_variables["temp"]["units"] == "degree_C"
    assert reader._raw_metadata_variables["temp"]["original_units"] == "degree_C"

    hints = json.loads(ds.attrs["rdi_mapping_hints"])
    assert hints["safe_reader_mappings"]["temp"] == "temperature"
    assert hints["velocity"]["coordinate_system"] == "earth"
    assert hints["velocity"]["confidence"] == "conditional"


def test_rdi_raw_reader_supports_older_debug_level_api(monkeypatch, tmp_path):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("beam")
    calls = []

    def fake_read_rdi(
        filename,
        userdata=None,
        nens=None,
        debug_level=-1,
        vmdas_search=False,
        winriver=False,
        **kwargs,
    ):
        calls.append(
            {
                "filename": filename,
                "debug_level": debug_level,
                "kwargs": kwargs,
            }
        )
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    reader = RdiRawReader(
        str(raw_file),
        debug=2,
        search_num=200,
        perform_default_postprocessing=False,
    )

    assert reader.data is expected
    assert calls == [
        {
            "filename": str(raw_file),
            "debug_level": 2,
            "kwargs": {"search_num": 200},
        }
    ]
    hints = json.loads(reader.data.attrs["rdi_mapping_hints"])
    assert hints["velocity"]["coordinate_system"] == "beam"
    assert hints["velocity"]["confidence"] == "uncertain"


def test_rdi_raw_reader_loads_through_public_api_autodetect(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_rdi(filename, **kwargs):
        calls.append((filename, kwargs))
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    ds = ssl.read(str(raw_file), use_steps=False, nens=1)

    assert ds is expected
    assert calls == [
        (
            str(raw_file),
            {
                "userdata": None,
                "nens": 1,
                "vmdas_search": False,
                "winriver": False,
            },
        )
    ]


def test_rdi_raw_reader_loads_pd0_through_public_api_autodetect(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path, ".PD0")
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_rdi(filename, **kwargs):
        calls.append((filename, kwargs))
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    ds = ssl.read(str(raw_file), use_steps=False, nens=1)

    assert ds is expected
    assert calls == [
        (
            str(raw_file),
            {
                "userdata": None,
                "nens": 1,
                "vmdas_search": False,
                "winriver": False,
            },
        )
    ]


def test_rdi_raw_reader_keeps_mapped_variable_provenance(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")

    def fake_read_rdi(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    ds = ssl.read(str(raw_file), nens=1)

    assert "temperature" in ds
    assert "speed_of_sound" in ds
    assert ds["temperature"].attrs["original_name"] == "temp"
    assert ds["temperature"].attrs["original_units"] == "degree_C"
    assert ds["speed_of_sound"].attrs["original_name"] == "c_sound"
    assert ds["speed_of_sound"].attrs["original_units"] == "m s-1"
    assert ds["pressure"].attrs["original_name"] == "pressure"
    assert ds["pressure"].attrs["original_units"] == "dbar"

    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    assert raw_metadata["variables"]["temperature"]["original_name"] == "temp"
    assert raw_metadata["variables"]["temperature"]["original_units"] == "degree_C"
    assert raw_metadata["variables"]["speed_of_sound"]["original_name"] == "c_sound"
    assert raw_metadata["variables"]["speed_of_sound"]["original_units"] == "m s-1"


def test_rdi_raw_reader_accepts_pd0_extension_without_autodetect(tmp_path):
    raw_file = _write_raw_file(tmp_path, ".PD0")

    reader = RdiRawReader(
        str(raw_file),
        perform_default_postprocessing=False,
    )

    assert reader.input_file == str(raw_file)


def test_rdi_raw_reader_lists_primary_and_all_extensions():
    info = next(item for item in ssl.list_readers() if item["key"] == "rdi-raw")

    assert info["extension"] == ".000"
    assert info["extensions"] == [".000", ".pd0", ".enr", ".ens", ".enx"]


def test_rdi_example_smoke_when_mhkit_is_available():
    pytest.importorskip("mhkit")
    example = Path("examples/trdi_adcp/DS2_2025_recovery.000")
    if not example.exists():
        pytest.skip("RDI ADCP example file is not available")

    ds = ssl.read(
        str(example),
        file_format="rdi-raw",
        use_steps=False,
        nens=2,
    )

    assert "time" in ds.coords
    assert ds.sizes["time"] > 0
    assert "vel" in ds.data_vars
