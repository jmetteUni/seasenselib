from __future__ import annotations

import json

import pytest

from seasenselib.readers.nortek_ascii_reader import NortekAsciiReader


def _write_nortek_ascii_pair(tmp_path, coordinate_system="BEAM"):
    dat_file = tmp_path / "sample.dat"
    hdr_file = tmp_path / "sample.hdr"

    hdr_file.write_text(
        "\n".join(
            [
                "User setup",
                "---------------------------------------------------------------------",
                f"Coordinate system                     {coordinate_system}",
                "",
                "Data file format",
                "---------------------------------------------------------------------",
                r"[C:\deployment\sample.dat]",
                " 1   Month                            (1-12)",
                " 2   Day                              (1-31)",
                " 3   Year",
                " 4   Hour                             (0-23)",
                " 5   Minute                           (0-59)",
                " 6   Second                           (0-59)",
                " 7   Velocity (Beam1|X|East)          (m/s)",
                " 8   Velocity (Beam2|Y|North)         (m/s)",
                " 9   Velocity (Beam3|Z|Up)            (m/s)",
                "10   Amplitude (Beam1)                (counts)",
                "11   Amplitude (Beam2)                (counts)",
                "12   Amplitude (Beam3)                (counts)",
                "13   Pressure                         (dbar)",
                "14   Pressure                         (m)",
                "15   Temperature                      (degrees C)",
                "---------------------------------------------------------------------",
                r"[C:\deployment\sample.dia]",
                " 1   Month                            (1-12)",
                " 2   Day                              (1-31)",
                " 3   Year",
                " 4   Hour                             (0-23)",
                " 5   Minute                           (0-59)",
                " 6   Second                           (0-59)",
                " 7   Velocity (Beam1|X|East)          (m/s)",
            ]
        ),
        encoding="utf-8",
    )

    dat_file.write_text(
        "\n".join(
            [
                "05 05 2026 05 00 00 0.1 0.2 0.3 11 12 13 2.4 2.3 22.7",
                "05 05 2026 05 02 00 0.4 0.5 0.6 14 15 16 2.5 2.4 22.8",
            ]
        ),
        encoding="utf-8",
    )

    return dat_file, hdr_file


@pytest.mark.parametrize(
    ("coordinate_system", "expected_velocity_names"),
    [
        ("BEAM", ["velocity_beam1", "velocity_beam2", "velocity_beam3"]),
        ("XYZ", ["x_velocity", "y_velocity", "z_velocity"]),
        ("ENU", ["east_velocity", "north_velocity", "up_velocity"]),
    ],
)
def test_nortek_ascii_velocity_names_follow_coordinate_system(
    tmp_path,
    coordinate_system,
    expected_velocity_names,
):
    dat_file, hdr_file = _write_nortek_ascii_pair(tmp_path, coordinate_system)

    ds = NortekAsciiReader(
        str(dat_file),
        str(hdr_file),
        perform_default_postprocessing=False,
    ).data

    assert ds.attrs["coordinate_system"] == coordinate_system
    assert [name for name in expected_velocity_names if name in ds.data_vars] == (
        expected_velocity_names
    )
    assert not {"east_velocity", "north_velocity", "up_velocity"}.issubset(
        ds.data_vars
    ) or coordinate_system == "ENU"


def test_nortek_ascii_amplitude_is_always_beam_and_pressure_m_is_depth(tmp_path):
    dat_file, hdr_file = _write_nortek_ascii_pair(tmp_path, "BEAM")

    ds = NortekAsciiReader(
        str(dat_file),
        str(hdr_file),
        perform_default_postprocessing=False,
    ).data

    assert {"amplitude_beam1", "amplitude_beam2", "amplitude_beam3"}.issubset(
        ds.data_vars
    )
    assert ds["amplitude_beam1"].attrs["coordinate_system"] == "BEAM"
    assert "pressure" in ds.data_vars
    assert "depth" in ds.data_vars
    assert "pressure_1" not in ds.data_vars
    assert ds["pressure"].attrs["units"] == "dbar"
    assert ds["depth"].attrs["units"] == "m"


def test_nortek_ascii_header_selects_dat_block_not_dia_block(tmp_path):
    dat_file, hdr_file = _write_nortek_ascii_pair(tmp_path, "BEAM")
    reader = NortekAsciiReader(
        str(dat_file),
        str(hdr_file),
        perform_default_postprocessing=False,
    )

    headers = reader._read_header(str(hdr_file), str(dat_file))

    assert len(headers) == 15
    assert headers[-1] == ("15", "Temperature", "degrees C")


def test_nortek_ascii_raw_metadata_contains_parsed_hdr_metadata(tmp_path):
    dat_file, hdr_file = _write_nortek_ascii_pair(tmp_path, "BEAM")

    ds = NortekAsciiReader(
        str(dat_file),
        str(hdr_file),
    ).data

    assert "raw_metadata" in ds.attrs
    payload = json.loads(ds.attrs["raw_metadata"])
    assert payload["blocks"]["header"] is None
    assert payload["blocks"]["attributes"]["coordinate_system"] == "BEAM"
    assert payload["variables"]["velocity_beam1"] == {
        "column_number": "7",
        "original_name": "Velocity (Beam1|X|East)",
        "units": "m/s",
    }
    assert payload["variables"]["depth"] == {
        "column_number": "14",
        "original_name": "Pressure",
        "units": "m",
    }
