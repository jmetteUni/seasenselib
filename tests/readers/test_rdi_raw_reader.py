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
            "salinity": ("time", np.array([35.0, 35.0, 35.0])),
            "depth": ("time", np.array([1220.0, 1220.0, 1220.0])),
            "heading": (
                "time",
                np.array([265.0, 265.1, 265.2]),
                {"units": "degree"},
            ),
            "pitch": (
                "time",
                np.array([1.0, 1.1, 1.2]),
                {"units": "degree"},
            ),
            "roll": (
                "time",
                np.array([-0.5, -0.4, -0.3]),
                {"units": "degree"},
            ),
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
            "sensors_src": "01111101",
            "sensors_avail": "00011101",
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


def test_rdi_raw_reader_wraps_mhkit_dolfyn(monkeypatch, tmp_path, capsys):
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
        print(f"Reading file {filename} ...")
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
    captured = capsys.readouterr()

    assert dict(ds.sizes) == dict(expected.sizes)
    assert "Reading file" not in captured.out
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
    assert "raw_data_reader" not in ds.attrs
    assert "rdi_reader_options" not in ds.attrs
    assert reader._raw_metadata_blocks["configuration"] == {
        "decoder": "mhkit.dolfyn.io.rdi.read_rdi",
        "reader_options": {
            "userdata": False,
            "nens": 2,
            "debug": 1,
            "vmdas_search": True,
            "winriver": True,
            "search_num": 500,
            "show_decoder_output": False,
        },
    }
    assert "attributes" not in reader._raw_metadata_blocks
    assert "vel" in ds
    assert "east_velocity" not in ds
    assert ds["temp"].attrs["original_name"] == "temp"
    assert ds["temp"].attrs["original_units"] == "degree_C"
    assert ds["temp"].attrs["standard_name"] == "sea_water_temperature"
    assert ds["c_sound"].attrs["original_name"] == "c_sound"
    assert ds["c_sound"].attrs["original_units"] == "m s-1"
    assert ds["c_sound"].attrs["standard_name"] == "speed_of_sound_in_sea_water"
    for orientation_variable in ("heading", "pitch", "roll"):
        assert ds[orientation_variable].attrs["measurement_type"] == "Measured"
        assert ds[orientation_variable].attrs["sensor_source"] == "sensor"
        assert ds[orientation_variable].attrs["sensor_source_basis"] == (
            "rdi_fixed_leader_source_and_available_flags"
        )
    assert "salinity" not in ds
    assert "depth" not in ds
    assert ds["rdi_salinity_setting"].attrs["original_name"] == "salinity"
    assert ds["rdi_salinity_setting"].attrs["units"] == "1e-3"
    assert ds["rdi_salinity_setting"].attrs["measurement_type"] == "Configured"
    assert ds["rdi_salinity_setting"].attrs["sensor_source"] == "configured"
    assert ds["rdi_salinity_setting"].attrs["sensor_source_basis"] == (
        "rdi_fixed_leader_source_flag"
    )
    assert "standard_name" not in ds["rdi_salinity_setting"].attrs
    assert ds["rdi_transducer_depth"].attrs["original_name"] == "depth"
    assert ds["rdi_transducer_depth"].attrs["measurement_type"] == "Configured"
    assert ds["rdi_transducer_depth"].attrs["sensor_source"] == (
        "configured_fallback"
    )
    assert ds["rdi_transducer_depth"].attrs["sensor_source_basis"] == (
        "rdi_fixed_leader_source_requested_but_unavailable"
    )
    assert "standard_name" not in ds["rdi_transducer_depth"].attrs
    assert reader._raw_metadata_variables["vel"]["dims"] == [
        "dir",
        "range",
        "time",
    ]
    assert reader._raw_metadata_variables["temp"]["original_name"] == "temp"
    assert reader._raw_metadata_variables["temp"]["units"] == "degree_C"
    assert reader._raw_metadata_variables["temp"]["original_units"] == "degree_C"

    assert "rdi_mapping_hints" not in ds.attrs
    hints = reader._raw_metadata_blocks["mapping_notes"]
    assert hints["safe_reader_mappings"]["temp"] == "temperature"
    assert hints["safe_reader_mappings"]["pressure"] == "pressure"
    assert "salinity" not in hints["safe_reader_mappings"]
    assert "depth" not in hints["safe_reader_mappings"]
    assert "rdi_salinity_setting" in hints["not_mapped"]["salinity"]
    assert "rdi_transducer_depth" in hints["not_mapped"]["depth"]
    assert reader._raw_metadata_blocks["sensor_sources"]["fields"]["salinity"][
        "source"
    ] == "configured"
    assert hints["velocity"]["coordinate_system"] == "earth"
    assert hints["velocity"]["cf_component_mapping"] == "possible_after_review"


def test_rdi_raw_reader_can_show_decoder_output(monkeypatch, tmp_path, capsys):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")

    def fake_read_rdi(filename, **kwargs):
        print(f"Reading file {filename} ...")
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    reader = RdiRawReader(
        str(raw_file),
        show_decoder_output=True,
        perform_default_postprocessing=False,
    )

    assert "rdi_salinity_setting" in reader.data
    assert "rdi_transducer_depth" in reader.data
    assert "Reading file" in capsys.readouterr().out


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

    assert "rdi_salinity_setting" in reader.data
    assert "rdi_transducer_depth" in reader.data
    assert calls == [
        {
            "filename": str(raw_file),
            "debug_level": 2,
            "kwargs": {"search_num": 200},
        }
    ]
    hints = reader._raw_metadata_blocks["mapping_notes"]
    assert hints["velocity"]["coordinate_system"] == "beam"
    assert hints["velocity"]["cf_component_mapping"] == "not_applied"


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

    assert "rdi_salinity_setting" in ds
    assert "rdi_transducer_depth" in ds
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

    assert "rdi_salinity_setting" in ds
    assert "rdi_transducer_depth" in ds
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
    assert "salinity" not in ds
    assert "depth" not in ds
    assert ds["rdi_salinity_setting"].attrs["original_name"] == "salinity"
    assert ds["rdi_transducer_depth"].attrs["original_name"] == "depth"

    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    raw_metadata_text = json.dumps(raw_metadata)
    assert raw_metadata["blocks"]["configuration"]["decoder"] == (
        "mhkit.dolfyn.io.rdi.read_rdi"
    )
    assert "raw_data_reader" not in raw_metadata_text
    assert "rdi_reader_options" not in raw_metadata_text
    assert raw_metadata_text.lower().count("dolfyn") == 1
    assert raw_metadata["variables"]["temperature"]["original_name"] == "temp"
    assert raw_metadata["variables"]["temperature"]["original_units"] == "degree_C"
    assert raw_metadata["variables"]["speed_of_sound"]["original_name"] == "c_sound"
    assert raw_metadata["variables"]["speed_of_sound"]["original_units"] == "m s-1"
    assert raw_metadata["variables"]["heading"]["sensor_source"] == "sensor"
    assert raw_metadata["variables"]["rdi_salinity_setting"]["original_name"] == "salinity"
    assert raw_metadata["variables"]["rdi_transducer_depth"]["original_name"] == "depth"
    assert raw_metadata["blocks"]["sensor_sources"]["fields"]["depth"]["source"] == (
        "configured_fallback"
    )
    assert "not external standard terms" in raw_metadata["blocks"]["sensor_sources"][
        "note"
    ]
    assert "sensor_source_basis" in raw_metadata["blocks"]["sensor_sources"][
        "definitions"
    ]


def test_rdi_raw_reader_does_not_promote_zero_pressure_placeholder(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    expected["pressure"] = ("time", np.array([0.0, 0.0, 0.0]))

    def fake_read_rdi(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    ds = ssl.read(str(raw_file), nens=1)

    assert "pressure" not in ds
    assert "rdi_pressure_placeholder" in ds
    assert ds["rdi_pressure_placeholder"].attrs["original_name"] == "pressure"
    assert ds["rdi_pressure_placeholder"].attrs["measurement_type"] == "Placeholder"
    assert "standard_name" not in ds["rdi_pressure_placeholder"].attrs
    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    hints = raw_metadata["blocks"]["mapping_notes"]
    assert "pressure" not in hints["safe_reader_mappings"]
    assert "rdi_pressure_placeholder" in hints["not_mapped"]["pressure"]


def test_rdi_raw_reader_preserves_sensor_backed_salinity(monkeypatch, tmp_path):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    expected.attrs["sensors_src"] = "01111111"
    expected.attrs["sensors_avail"] = "00111111"
    expected["salinity"].attrs["units"] = "psu"

    def fake_read_rdi(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    reader = RdiRawReader(
        str(raw_file),
        perform_default_postprocessing=False,
    )
    ds = reader.data

    assert "salinity" in ds
    assert "rdi_salinity_setting" not in ds
    assert ds["salinity"].attrs["measurement_type"] == "Measured"
    assert ds["salinity"].attrs["units"] == "1e-3"
    assert ds["salinity"].attrs["original_units"] == "psu"
    assert ds["salinity"].attrs["sensor_source"] == "sensor"
    hints = reader._raw_metadata_blocks["mapping_notes"]
    assert hints["safe_reader_mappings"]["salinity"] == "salinity"


def test_rdi_raw_reader_does_not_promote_manual_temperature(monkeypatch, tmp_path):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    expected.attrs["sensors_src"] = "01111100"

    def fake_read_rdi(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_rdi)

    ds = ssl.read(str(raw_file), nens=1)

    assert "temperature" not in ds
    assert "rdi_temperature_setting" in ds
    assert ds["rdi_temperature_setting"].attrs["original_name"] == "temp"
    assert ds["rdi_temperature_setting"].attrs["measurement_type"] == "Configured"
    assert ds["rdi_temperature_setting"].attrs["sensor_source"] == "configured"
    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    hints = raw_metadata["blocks"]["mapping_notes"]
    assert "temp" not in hints["safe_reader_mappings"]
    assert "rdi_temperature_setting" in hints["not_mapped"]["temp"]


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


def test_rdi_example_can_be_written_to_netcdf_when_mhkit_is_available(tmp_path):
    pytest.importorskip("mhkit")
    example = Path("examples/trdi_adcp/_RDI_000.000")
    if not example.exists():
        pytest.skip("RDI ADCP netCDF example file is not available")

    ds = ssl.read(
        str(example),
        file_format="rdi-raw",
        nens=2,
    )
    output = tmp_path / "rdi.nc"

    ssl.write(ds, str(output))

    assert output.exists()
    with xr.open_dataset(output) as written:
        assert "time" in written.coords
        assert "vel" in written.data_vars
