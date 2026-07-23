from __future__ import annotations

import io
import json
import os
import struct
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


def _write_nortek2_avgd_file(tmp_path):
    raw_file = tmp_path / "sample_avgd.aqd"
    config_text = (
        'GETCLOCKSTR,TIME="2026-05-03 11:11:25",OFFSET="+00:00",TZ="UTC"\r\n'
        'ID,STR="Aquadopp Deep Water 2 MHz D2VC",SN=401928\r\n'
        'GETAVG,NC=1,CS=0.75,BD=0.50,CY="ENU",DF=7,NPING=12,NB=3\r\n'
    )
    string_payload = b"\x12" + config_text.encode("utf-8") + b"\x00"
    data_payload = bytearray(191)
    data_payload[16] = 3
    struct.pack_into("<I", data_payload, 20, 401928)
    struct.pack_into("<6BH", data_payload, 24, 126, 4, 7, 0, 0, 0, 0)
    for offset, value in (
        (32, 1490.95),
        (36, 2.4270833),
        (40, 1873.0679),
        (44, 1882.5684),
        (48, 46.52165),
        (52, -4.03931),
        (56, -3.87268),
        (80, 0.75),
        (84, 0.5),
        (88, 12.29582),
        (92, 1.88333),
        (96, 4.2),
        (100, 2.0),
        (104, 986.0),
        (108, 1006.0),
        (112, -4252.0),
        (116, -0.07043),
        (120, -0.06732),
        (124, 0.99402),
        (128, 5.03699),
        (148, 0.29405),
        (152, 347.4289),
    ):
        struct.pack_into("<f", data_payload, offset, value)
    struct.pack_into("<3h", data_payload, 160, -64, 287, -21)
    data_payload[166:169] = bytes([105, 92, 93])
    data_payload[169:172] = bytes([97, 93, 93])
    data_payload[172] = 100

    raw_file.write_bytes(
        _nortek2_packet(160, string_payload)
        + _nortek2_packet(38, bytes(data_payload))
    )
    return raw_file


def _nortek2_packet(record_id, payload):
    return struct.pack("<BBBBhhh", 165, 10, record_id, 48, len(payload), 0, 0) + payload


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


def _dolfyn_like_nortek2_dataset():
    return xr.Dataset(
        {
            "vel_avg": (
                ("dir", "range_avg", "time_avg"),
                np.array([[[-0.085]], [[0.256]], [[-0.008]]], dtype=np.float32),
                {"units": "m s-1"},
            ),
            "amp_avg": (
                ("beam", "range_avg", "time_avg"),
                np.array([[[48.5]], [[46.5]], [[46.5]]], dtype=np.float32),
                {"units": "1"},
            ),
            "corr_avg": (
                ("beam", "range_avg", "time_avg"),
                np.array([[[93]], [[84]], [[89]]], dtype=np.uint8),
                {"units": "%"},
            ),
            "temp_avg": ("time_avg", np.array([2.43]), {"units": "degree_C"}),
            "c_sound_avg": ("time_avg", np.array([1491.0]), {"units": "m s-1"}),
            "pressure_avg": ("time_avg", np.array([1873.007]), {"units": "dbar"}),
            "batt_avg": ("time_avg", np.array([11.8]), {"units": "V"}),
            "heading_avg": (
                "time_avg",
                np.array([47.17]),
                {"units": "degree", "standard_name": "platform_orientation"},
            ),
            "pitch_avg": (
                "time_avg",
                np.array([-5.03]),
                {"units": "degree", "standard_name": "platform_pitch"},
            ),
            "roll_avg": (
                "time_avg",
                np.array([-4.25]),
                {"units": "degree", "standard_name": "platform_roll"},
            ),
        },
        coords={
            "dir": ("dir", ["E", "N", "U"]),
            "beam": ("beam", [1, 2, 3]),
            "range_avg": ("range_avg", [1.25]),
            "time_avg": (
                "time_avg",
                np.array(["2026-05-07T00:00:00.001700"], dtype="datetime64[ns]"),
                {"units": "seconds since 1970-01-01 00:00:00 UTC"},
            ),
        },
        attrs={
            "coord_sys": "earth",
            "coord_sys_axes_avg": "ENU",
            "inst_make": "Nortek",
            "inst_model": "Aquadopp Deep Water 2 MHz D2VC",
            "serial_number": "401928",
        },
    )


def _install_fake_mhkit(monkeypatch, read_nortek):
    mhkit = ModuleType("mhkit")
    mhkit.dolfyn = SimpleNamespace(
        io=SimpleNamespace(nortek=SimpleNamespace(read_nortek=read_nortek))
    )
    monkeypatch.setitem(sys.modules, "mhkit", mhkit)


def _install_fake_mhkit_with_nortek2_modules(
    monkeypatch,
    read_signature,
    reader_class=None,
):
    mhkit = ModuleType("mhkit")
    dolfyn = ModuleType("mhkit.dolfyn")
    io_module = ModuleType("mhkit.dolfyn.io")
    nortek = ModuleType("mhkit.dolfyn.io.nortek")
    nortek2 = ModuleType("mhkit.dolfyn.io.nortek2")

    def fail_classic_reader(filename, **kwargs):
        raise AssertionError("classic Nortek decoder should not be called")

    read_signature.__module__ = "mhkit.dolfyn.io.nortek2"
    nortek.read_nortek = fail_classic_reader
    nortek2.read_signature = read_signature
    if reader_class is not None:
        nortek2._Ad2cpReader = reader_class
    io_module.nortek = nortek
    io_module.nortek2 = nortek2
    dolfyn.io = io_module
    mhkit.dolfyn = dolfyn

    monkeypatch.setitem(sys.modules, "mhkit", mhkit)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn", dolfyn)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn.io", io_module)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn.io.nortek", nortek)
    monkeypatch.setitem(sys.modules, "mhkit.dolfyn.io.nortek2", nortek2)
    return nortek2


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


def test_nortek_raw_reader_does_not_register_coordinate_transformations(tmp_path):
    raw_file = _write_raw_file(tmp_path)
    reader = NortekRawReader(str(raw_file), perform_default_postprocessing=False)

    assert reader.pipeline_transformations(_dolfyn_like_dataset("beam")) == []


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
            "rebuild_index": None,
            "dual_profile": None,
            "show_decoder_output": False,
            "apply_aquadopp_compatibility": True,
            "apply_nortek2_aquadopp_compatibility": True,
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


def test_nortek_raw_reader_uses_nortek2_decoder_for_gen2_header(
    monkeypatch,
    tmp_path,
):
    raw_file = tmp_path / "sample.aqd"
    raw_file.write_bytes(b"\xa5\x0a\x00\x00")
    expected = _dolfyn_like_nortek2_dataset()
    calls = []

    def fake_read_signature(
        filename,
        userdata=True,
        nens=None,
        rebuild_index=False,
        debug=False,
        dual_profile=False,
    ):
        print(f"Reading file {filename} ...")
        calls.append(
            {
                "filename": filename,
                "userdata": userdata,
                "nens": nens,
                "rebuild_index": rebuild_index,
                "debug": debug,
                "dual_profile": dual_profile,
            }
        )
        return expected

    _install_fake_mhkit_with_nortek2_modules(monkeypatch, fake_read_signature)

    reader = NortekRawReader(
        str(raw_file),
        userdata=False,
        nens=5,
        rebuild_index=True,
        debug=True,
        dual_profile=False,
        perform_default_postprocessing=False,
    )

    assert "vel_avg" in reader.data
    assert calls == [
        {
            "filename": str(raw_file),
            "userdata": False,
            "nens": 5,
            "rebuild_index": True,
            "debug": True,
            "dual_profile": False,
        }
    ]
    assert reader._raw_metadata_blocks["configuration"]["decoder"] == (
        "mhkit.dolfyn.io.nortek2.read_signature"
    )
    hints = reader._raw_metadata_blocks["mapping_notes"]
    assert hints["safe_reader_mappings"]["temp_avg"] == "temperature"
    assert hints["safe_reader_mappings"]["pressure_avg"] == "pressure"
    assert reader.data["time_avg"].attrs["original_units"] == (
        "seconds since 1970-01-01 00:00:00 UTC"
    )
    assert reader.data["heading_avg"].attrs["standard_name"] == (
        "platform_heading_angle"
    )


def test_nortek_raw_reader_repairs_gen2_aquadopp_average_tail(
    monkeypatch,
    tmp_path,
):
    raw_file = tmp_path / "sample.aqd"
    raw_file.write_bytes(b"\xa5\x0a\x00\x00")

    class FakeAd2cpReader:
        def __init__(self):
            placeholders = b"\x01\x00\x80\x7f" * 3
            velocity = np.array([-85, 256, -8], dtype="<i2").tobytes()
            amplitude = np.array([97, 93, 93], dtype=np.uint8).tobytes()
            correlation = np.array([93, 84, 89], dtype=np.uint8).tobytes()
            self.f = io.BytesIO(
                b"\x00" * 10
                + b"\x00" * 88
                + placeholders
                + velocity
                + amplitude
                + correlation
            )

        def _read_hdr(self, do_cs=False):
            self.f.seek(10)
            return {"sync": 165, "hsz": 10, "id": 22, "sz": 112}

        def _read_burst(self, record_id, data, ensemble_index, echo=False):
            self.f.read(88)
            data["vel"][..., ensemble_index] = np.array([[1], [32640], [1]])
            data["amp"][..., ensemble_index] = np.array([[64], [63], [0]])
            data["corr"][..., ensemble_index] = np.array([[0], [128], [127]])

    def fake_read_signature(filename, **kwargs):
        from mhkit.dolfyn.io import nortek2

        reader = nortek2._Ad2cpReader()
        data = {
            "DatOffset": np.array([76], dtype=np.uint8),
            "vel": np.zeros((3, 1, 1), dtype=np.int16),
            "amp": np.zeros((3, 1, 1), dtype=np.uint8),
            "corr": np.zeros((3, 1, 1), dtype=np.uint8),
        }
        header = reader._read_hdr()
        reader._read_burst(header["id"], data, 0)

        assert reader.f.tell() == 122
        np.testing.assert_array_equal(data["vel"][..., 0], [[-85], [256], [-8]])
        np.testing.assert_array_equal(data["amp"][..., 0], [[97], [93], [93]])
        np.testing.assert_array_equal(data["corr"][..., 0], [[93], [84], [89]])
        return _dolfyn_like_nortek2_dataset()

    _install_fake_mhkit_with_nortek2_modules(
        monkeypatch,
        fake_read_signature,
        FakeAd2cpReader,
    )

    reader = NortekRawReader(str(raw_file), perform_default_postprocessing=False)

    assert "vel_avg" in reader.data
    assert reader._raw_metadata_blocks["configuration"]["compatibility"] == [
        {
            "name": "nortek2_aquadopp_average_record_tail",
            "scope": "in_memory_for_this_read",
            "reason": (
                "Repairs AD2CP/Aquadopp2 average records where DOLfYN's "
                "layout leaves the final velocity, amplitude and correlation "
                "samples unread."
            ),
        }
    ]


def test_nortek_raw_reader_decodes_nortek2_avgd_product_without_dolfyn(
    monkeypatch,
    tmp_path,
):
    raw_file = _write_nortek2_avgd_file(tmp_path)

    def fail_import(name, *args, **kwargs):
        if name.startswith("mhkit"):
            raise AssertionError("DOLfYN should not be imported for ID 38 products")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)

    reader = NortekRawReader(str(raw_file), perform_default_postprocessing=False)
    ds = reader.data

    assert dict(ds.sizes) == {
        "time_avg": 1,
        "range_avg": 1,
        "beam": 3,
        "dir": 3,
        "dirIMU": 3,
    }
    assert reader._raw_metadata_blocks["configuration"]["decoder"] == (
        "seasenselib.nortek2.avgd"
    )
    assert ds.attrs["inst_model"] == "Aquadopp Deep Water 2 MHz D2VC"
    assert ds.attrs["coord_sys"] == "earth"
    assert str(ds["time_avg"].values[0]) == "2026-05-07T00:00:00.000000000"
    assert ds["range_avg"].values[0] == pytest.approx(1.25)
    assert ds["c_sound_avg"].values[0] == pytest.approx(1490.95, abs=1e-4)
    assert ds["temp_avg"].values[0] == pytest.approx(2.4270833, abs=1e-6)
    assert ds["pressure_avg"].values[0] == pytest.approx(1873.0679, abs=1e-4)
    assert ds["batt_avg"].values[0] == pytest.approx(12.29582, abs=1e-5)
    np.testing.assert_allclose(
        ds["vel_avg"].isel(range_avg=0, time_avg=0).values,
        [-0.064, 0.287, -0.021],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ds["amp_avg"].isel(range_avg=0, time_avg=0).values,
        [52.5, 46.0, 46.5],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        ds["corr_avg"].isel(range_avg=0, time_avg=0).values,
        [97, 93, 93],
    )
    assert reader._raw_metadata_blocks["mapping_notes"]["safe_reader_mappings"][
        "temp_avg"
    ] == "temperature"
    assert "vel_avg" in reader._raw_metadata_blocks["mapping_notes"]["not_mapped"]


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
