from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import seasenselib as ssl
from seasenselib.readers import get_reader_by_format_key, get_readers_by_extension
from seasenselib.readers.nortek_raw_reader import NortekRawReader


def _write_raw_file(tmp_path, suffix=".aqd"):
    raw_file = tmp_path / f"sample{suffix}"
    raw_file.write_bytes(b"\xa5\x05\x18\x00")
    return raw_file


def _dolfyn_like_dataset(
    coord_sys="earth",
    *,
    pressure_sensor="yes",
    user_specified_sound_speed="False",
):
    return xr.Dataset(
        {
            "vel": (
                ("dir", "time"),
                np.array(
                    [
                        [-0.10, -0.20, -0.30],
                        [0.01, 0.02, 0.03],
                        [0.001, 0.002, 0.003],
                    ],
                    dtype=np.float32,
                ),
                {"units": "m s-1"},
            ),
            "amp": (
                ("beam", "time"),
                np.array([[70, 71, 72], [73, 74, 75], [76, 77, 78]], dtype=np.uint8),
                {"units": "1"},
            ),
            "corr": (
                ("beam", "time"),
                np.array([[90, 91, 92], [93, 94, 95], [96, 97, 98]], dtype=np.uint8),
                {"units": "%"},
            ),
            "temp": ("time", np.array([7.1, 7.2, 7.3]), {"units": "degree_C"}),
            "c_sound": (
                "time",
                np.array([1450.0, 1450.1, 1450.2]),
                {"units": "m s-1"},
            ),
            "pressure": ("time", np.array([100.0, 100.1, 100.2]), {"units": "dbar"}),
            "batt": ("time", np.array([13.1, 13.1, 13.0]), {"units": "V"}),
            "heading": (
                "time",
                np.array([265.0, 265.1, 265.2]),
                {"units": "degree", "standard_name": "platform_orientation"},
            ),
            "pitch": (
                "time",
                np.array([1.0, 1.1, 1.2]),
                {"units": "degree", "standard_name": "platform_pitch"},
            ),
            "roll": (
                "time",
                np.array([-0.5, -0.4, -0.3]),
                {"units": "degree", "standard_name": "platform_roll"},
            ),
            "status": ("time", np.array([52, 52, 52], dtype=np.uint8)),
            "error": ("time", np.array([0, 0, 0], dtype=np.uint8)),
        },
        coords={
            "dir": ("dir", ["E", "N", "U"]),
            "beam": ("beam", [1, 2, 3]),
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
                {"units": "seconds since 1970-01-01 00:00:00 UTC"},
            ),
        },
        attrs={
            "coord_sys": coord_sys,
            "inst_make": "Nortek",
            "inst_model": "Aquadopp",
            "serial_number": "AQD1234",
            "pressure_sensor": pressure_sensor,
            "compass": "yes",
            "tilt_sensor": "yes",
            "user_specified_sound_speed": user_specified_sound_speed,
        },
    )


def _install_fake_mhkit(monkeypatch, read_nortek):
    mhkit = ModuleType("mhkit")
    mhkit.dolfyn = SimpleNamespace(
        io=SimpleNamespace(nortek=SimpleNamespace(read_nortek=read_nortek))
    )
    monkeypatch.setitem(sys.modules, "mhkit", mhkit)


def _install_fake_mhkit_with_nortek_modules(monkeypatch, read_nortek):
    mhkit = ModuleType("mhkit")
    dolfyn = ModuleType("mhkit.dolfyn")
    io = ModuleType("mhkit.dolfyn.io")
    nortek = ModuleType("mhkit.dolfyn.io.nortek")
    nortek_defs = ModuleType("mhkit.dolfyn.io.nortek_defs")

    nortek_defs.vec_data = {}
    nortek_defs.vec_sys = {
        name: object()
        for name in (
            "time",
            "error",
            "batt",
            "c_sound",
            "heading",
            "pitch",
            "roll",
            "status",
            "temp",
        )
    }
    read_nortek.__module__ = "mhkit.dolfyn.io.nortek"
    nortek.read_nortek = read_nortek
    nortek.defs = nortek_defs
    io.nortek = nortek
    dolfyn.io = io
    mhkit.dolfyn = dolfyn

    monkeypatch.setitem(sys.modules, "mhkit", mhkit)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn", dolfyn)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn.io", io)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn.io.nortek", nortek)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn.io.nortek_defs", nortek_defs)
    return nortek_defs


def _nortek_smoke_test_file() -> Path:
    path = os.environ.get("SEASENSELIB_NORTEK_RAW_EXAMPLE")
    if not path:
        pytest.skip("Set SEASENSELIB_NORTEK_RAW_EXAMPLE to run Nortek raw smoke tests")

    example = Path(path)
    if not example.exists():
        pytest.skip(f"Nortek raw smoke-test file does not exist: {example}")
    return example


def test_nortek_raw_reader_exposes_format_metadata():
    assert NortekRawReader.format_key() == "nortek-raw"
    assert NortekRawReader.format_name() == "Nortek Raw (experimental)"
    assert NortekRawReader.file_extension() == ".aqd"
    assert NortekRawReader.file_extensions() == (".aqd", ".vec", ".wpr")
    assert NortekRawReader._get_valid_extensions() == (".aqd", ".vec", ".wpr")


def test_nortek_raw_reader_is_discoverable_by_format_key():
    assert get_reader_by_format_key("nortek-raw") is NortekRawReader


def test_nortek_raw_reader_is_discoverable_by_extension():
    assert get_readers_by_extension(".aqd") == [NortekRawReader]
    assert get_readers_by_extension(".AQD") == [NortekRawReader]
    assert get_readers_by_extension("aqd") == [NortekRawReader]
    assert get_readers_by_extension(".VEC") == [NortekRawReader]
    assert get_readers_by_extension(".wpr") == [NortekRawReader]


def test_nortek_raw_reader_wraps_mhkit_dolfyn(monkeypatch, tmp_path, capsys):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_nortek(
        filename,
        userdata=True,
        debug=False,
        do_checksum=False,
        nens=None,
    ):
        print(f"Reading file {filename} ...")
        calls.append(
            {
                "filename": filename,
                "userdata": userdata,
                "debug": debug,
                "do_checksum": do_checksum,
                "nens": nens,
            }
        )
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    reader = NortekRawReader(
        str(raw_file),
        userdata=False,
        nens=2,
        debug=True,
        do_checksum=True,
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
            "debug": True,
            "do_checksum": True,
            "nens": 2,
        }
    ]
    assert reader._raw_metadata_blocks["configuration"] == {
        "decoder": "mhkit.dolfyn.io.nortek.read_nortek",
        "status": "experimental",
        "note": (
            "Experimental reader: Nortek raw support is available for early "
            "validation and may be refined as additional Nortek binary "
            "variants are tested."
        ),
        "reader_options": {
            "userdata": False,
            "nens": 2,
            "debug": True,
            "do_checksum": True,
            "show_decoder_output": False,
            "apply_aquadopp_compatibility": True,
        },
    }
    assert "attributes" not in reader._raw_metadata_blocks
    assert "vel" in ds
    assert "east_velocity" not in ds
    assert ds["time"].attrs["original_units"] == (
        "seconds since 1970-01-01 00:00:00 UTC"
    )
    assert "units" not in ds["time"].attrs
    assert ds["temp"].attrs["original_name"] == "temp"
    assert ds["temp"].attrs["original_units"] == "degree_C"
    assert ds["c_sound"].attrs["original_name"] == "c_sound"
    assert ds["c_sound"].attrs["standard_name"] == "speed_of_sound_in_sea_water"
    assert ds["pressure"].attrs["measurement_type"] == "Measured"
    assert ds["pressure"].attrs["sensor_source"] == "sensor"
    assert ds["pressure"].attrs["sensor_source_basis"] == (
        "nortek_pressure_sensor_header"
    )
    assert ds["batt"].attrs["original_units"] == "V"
    assert ds["heading"].attrs["original_standard_name"] == "platform_orientation"
    assert ds["heading"].attrs["standard_name"] == "platform_heading_angle"
    assert ds["heading"].attrs["sensor_source"] == "sensor"
    assert ds["heading"].attrs["sensor_source_basis"] == "nortek_compass_header"
    assert ds["pitch"].attrs["standard_name"] == "platform_pitch_angle"
    assert ds["pitch"].attrs["sensor_source"] == "sensor"
    assert ds["pitch"].attrs["sensor_source_basis"] == "nortek_tilt_sensor_header"
    assert ds["roll"].attrs["standard_name"] == "platform_roll_angle"
    assert ds["roll"].attrs["sensor_source"] == "sensor"
    assert ds["roll"].attrs["sensor_source_basis"] == "nortek_tilt_sensor_header"
    assert "sensor_source" not in ds["temp"].attrs
    assert "sensor_source" not in ds["vel"].attrs
    assert reader._raw_metadata_variables["vel"]["dims"] == ["dir", "time"]
    assert reader._raw_metadata_variables["temp"]["original_name"] == "temp"
    assert reader._raw_metadata_variables["temp"]["units"] == "degree_C"
    assert reader._raw_metadata_variables["pressure"]["sensor_source"] == "sensor"

    hints = reader._raw_metadata_blocks["mapping_notes"]
    assert hints["safe_reader_mappings"]["temp"] == "temperature"
    assert hints["safe_reader_mappings"]["c_sound"] == "speed_of_sound"
    assert hints["safe_reader_mappings"]["pressure"] == "pressure"
    assert hints["safe_reader_mappings"]["batt"] == "battery_voltage"
    assert hints["velocity"]["coordinate_system"] == "earth"
    assert hints["velocity"]["cf_component_mapping"] == "possible_after_review"


def test_nortek_raw_reader_can_show_decoder_output(monkeypatch, tmp_path, capsys):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")

    def fake_read_nortek(filename, **kwargs):
        print(f"Reading file {filename} ...")
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    reader = NortekRawReader(
        str(raw_file),
        show_decoder_output=True,
        perform_default_postprocessing=False,
    )

    assert "vel" in reader.data
    assert "Reading file" in capsys.readouterr().out


def test_nortek_raw_reader_applies_aquadopp_template_compatibility(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_nortek(filename, **kwargs):
        from mhkit.dolfyn.io import nortek_defs

        calls.append(sorted(nortek_defs.vec_data))
        assert "time" in nortek_defs.vec_data
        return expected

    nortek_defs = _install_fake_mhkit_with_nortek_modules(
        monkeypatch,
        fake_read_nortek,
    )

    reader = NortekRawReader(
        str(raw_file),
        perform_default_postprocessing=False,
    )

    assert "vel" in reader.data
    assert "time" in calls[0]
    assert "time" not in nortek_defs.vec_data
    assert reader._raw_metadata_blocks["configuration"]["compatibility"] == [
        {
            "name": "aquadopp_single_point_template",
            "scope": "in_memory_for_this_read",
            "reason": (
                "Adds missing timestamp and environmental variable definitions "
                "for classic Aquadopp 0x01 blocks when the backend template is "
                "incomplete."
            ),
        }
    ]


def test_nortek_raw_reader_loads_through_public_api_autodetect(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path, ".AQD")
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_nortek(filename, **kwargs):
        calls.append((filename, kwargs))
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    ds = ssl.read(str(raw_file), use_steps=False, nens=1)

    assert "vel" in ds
    assert calls == [(str(raw_file), {"nens": 1})]


@pytest.mark.parametrize("suffix", [".VEC", ".wpr"])
def test_nortek_raw_reader_loads_additional_classic_nortek_suffixes(
    monkeypatch,
    tmp_path,
    suffix,
):
    raw_file = _write_raw_file(tmp_path, suffix)
    expected = _dolfyn_like_dataset("earth")
    calls = []

    def fake_read_nortek(filename, **kwargs):
        calls.append((filename, kwargs))
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    ds = ssl.read(str(raw_file), use_steps=False)

    assert "vel" in ds
    assert calls == [(str(raw_file), {})]


def test_nortek_raw_reader_keeps_mapped_variable_provenance(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")

    def fake_read_nortek(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    ds = ssl.read(str(raw_file), userdata=False)

    assert "temperature" in ds
    assert "speed_of_sound" in ds
    assert "battery_voltage" in ds
    assert "pressure" in ds
    assert "vel" in ds
    assert ds["temperature"].attrs["original_name"] == "temp"
    assert ds["temperature"].attrs["original_units"] == "degree_C"
    assert ds["speed_of_sound"].attrs["original_name"] == "c_sound"
    assert ds["speed_of_sound"].attrs["original_units"] == "m s-1"
    assert ds["battery_voltage"].attrs["original_name"] == "batt"
    assert ds["battery_voltage"].attrs["original_units"] == "V"
    assert ds["pressure"].attrs["original_name"] == "pressure"

    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    raw_metadata_text = json.dumps(raw_metadata)
    assert raw_metadata["blocks"]["configuration"]["decoder"] == (
        "mhkit.dolfyn.io.nortek.read_nortek"
    )
    assert raw_metadata["blocks"]["configuration"]["status"] == "experimental"
    assert "early validation" in raw_metadata["blocks"]["configuration"]["note"]
    assert raw_metadata_text.lower().count("dolfyn") == 1
    assert raw_metadata["variables"]["temperature"]["original_name"] == "temp"
    assert raw_metadata["variables"]["speed_of_sound"]["original_name"] == "c_sound"
    assert raw_metadata["variables"]["battery_voltage"]["original_name"] == "batt"
    assert raw_metadata["variables"]["pressure"]["sensor_source"] == "sensor"
    assert raw_metadata["variables"]["heading"]["sensor_source"] == "sensor"
    assert raw_metadata["blocks"]["sensor_sources"]["fields"]["pressure"] == {
        "source": "sensor",
        "basis": "nortek_pressure_sensor_header",
    }
    assert raw_metadata["blocks"]["sensor_sources"]["raw_fields"] == {
        "pressure_sensor": "yes",
        "compass": "yes",
        "tilt_sensor": "yes",
        "user_specified_sound_speed": "False",
    }
    assert raw_metadata["blocks"]["mapping_notes"]["safe_reader_mappings"] == {
        "temp": "temperature",
        "c_sound": "speed_of_sound",
        "pressure": "pressure",
        "batt": "battery_voltage",
    }


def test_nortek_raw_reader_keeps_configured_sound_speed_separate(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset(
        "earth",
        user_specified_sound_speed="True",
    )

    def fake_read_nortek(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    ds = ssl.read(str(raw_file), userdata=False)

    assert "speed_of_sound" not in ds
    assert "nortek_sound_speed_setting" in ds
    assert ds["nortek_sound_speed_setting"].attrs["original_name"] == "c_sound"
    assert ds["nortek_sound_speed_setting"].attrs["measurement_type"] == "Configured"
    assert ds["nortek_sound_speed_setting"].attrs["sensor_source"] == "configured"
    assert ds["nortek_sound_speed_setting"].attrs["sensor_source_basis"] == (
        "nortek_user_specified_sound_speed"
    )
    assert "standard_name" not in ds["nortek_sound_speed_setting"].attrs
    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    hints = raw_metadata["blocks"]["mapping_notes"]
    assert "c_sound" not in hints["safe_reader_mappings"]
    assert "nortek_sound_speed_setting" in hints["not_mapped"]["c_sound"]


def test_nortek_raw_reader_does_not_promote_pressure_placeholder(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth", pressure_sensor="no")
    expected["pressure"] = ("time", np.array([0.0, 0.0, 0.0]))

    def fake_read_nortek(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    ds = ssl.read(str(raw_file), userdata=False)

    assert "pressure" not in ds
    assert "nortek_pressure_placeholder" in ds
    assert ds["nortek_pressure_placeholder"].attrs["original_name"] == "pressure"
    assert ds["nortek_pressure_placeholder"].attrs["measurement_type"] == "Placeholder"
    assert ds["nortek_pressure_placeholder"].attrs["sensor_source"] == "placeholder"
    assert ds["nortek_pressure_placeholder"].attrs["sensor_source_basis"] == (
        "nortek_pressure_all_zero"
    )
    assert "standard_name" not in ds["nortek_pressure_placeholder"].attrs
    raw_metadata = json.loads(ds.attrs["raw_metadata"])
    hints = raw_metadata["blocks"]["mapping_notes"]
    assert "pressure" not in hints["safe_reader_mappings"]
    assert "nortek_pressure_placeholder" in hints["not_mapped"]["pressure"]


def test_nortek_raw_reader_marks_zero_correlation_as_placeholder(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_raw_file(tmp_path)
    expected = _dolfyn_like_dataset("earth")
    expected["corr"] = (("beam", "time"), np.zeros((3, 3), dtype=np.uint8))

    def fake_read_nortek(filename, **kwargs):
        return expected

    _install_fake_mhkit(monkeypatch, fake_read_nortek)

    reader = NortekRawReader(
        str(raw_file),
        perform_default_postprocessing=False,
    )
    ds = reader.data

    assert "corr" not in ds
    assert "nortek_correlation_placeholder" in ds
    assert ds["nortek_correlation_placeholder"].attrs["original_name"] == "corr"
    assert ds["nortek_correlation_placeholder"].attrs["measurement_type"] == (
        "Placeholder"
    )
    assert ds["nortek_correlation_placeholder"].attrs["sensor_source"] == (
        "placeholder"
    )
    assert ds["nortek_correlation_placeholder"].attrs["sensor_source_basis"] == (
        "nortek_correlation_all_zero"
    )
    hints = reader._raw_metadata_blocks["mapping_notes"]
    assert "nortek_correlation_placeholder" in hints["not_mapped"]["corr"]


def test_nortek_raw_reader_lists_primary_extension():
    info = next(item for item in ssl.list_readers() if item["key"] == "nortek-raw")

    assert info["extension"] == ".aqd"
    assert info["extensions"] == [".aqd", ".vec", ".wpr"]


def test_nortek_example_smoke_when_mhkit_is_available():
    pytest.importorskip("mhkit")
    example = _nortek_smoke_test_file()

    ds = ssl.read(
        str(example),
        file_format="nortek-raw",
        use_steps=False,
    )

    assert "time" in ds.coords
    assert ds.sizes["time"] > 0
    assert "vel" in ds.data_vars


def test_nortek_example_can_be_written_to_netcdf_when_mhkit_is_available(tmp_path):
    pytest.importorskip("mhkit")
    example = _nortek_smoke_test_file()

    ds = ssl.read(
        str(example),
        file_format="nortek-raw",
    )
    output = tmp_path / "nortek.nc"

    ssl.write(ds, str(output))

    assert output.exists()
    with xr.open_dataset(output) as written:
        assert "time" in written.coords
        assert "vel" in written.data_vars
