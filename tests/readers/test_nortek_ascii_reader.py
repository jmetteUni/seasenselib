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
                "Transformation matrix                 0.7356 -0.3677 -0.3677",
                "                                      0.0000 -0.6370 0.6370",
                "                                      0.7888 0.7888 0.7888",
                "Magnetometer calibration matrix       0.9910 0.0220 -0.0229",
                "                                      0.0220 1.0000 -0.0062",
                "                                      -0.0180 0.0304 0.9507",
                "Compass hard iron calibration         5 -90 10",
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


def _write_nortek_ascii_pair_with_custom_velocity_labels(
    tmp_path,
    coordinate_system,
    velocity_labels,
):
    dat_file, hdr_file = _write_nortek_ascii_pair(tmp_path, coordinate_system)
    text = hdr_file.read_text(encoding="utf-8")
    text = text.replace(
        "Velocity (Beam1|X|East)",
        velocity_labels[0],
    )
    text = text.replace(
        "Velocity (Beam2|Y|North)",
        velocity_labels[1],
    )
    text = text.replace(
        "Velocity (Beam3|Z|Up)",
        velocity_labels[2],
    )
    hdr_file.write_text(text, encoding="utf-8")
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


def test_nortek_ascii_velocity_names_use_matching_label_component(tmp_path):
    dat_file, hdr_file = _write_nortek_ascii_pair_with_custom_velocity_labels(
        tmp_path,
        "ENU",
        [
            "Velocity (Beam7|Z|Up)",
            "Velocity (Beam2|X|East)",
            "Velocity (Beam9|Y|North)",
        ],
    )

    ds = NortekAsciiReader(
        str(dat_file),
        str(hdr_file),
        perform_default_postprocessing=False,
    ).data

    assert {"east_velocity", "north_velocity", "up_velocity"}.issubset(
        ds.data_vars
    )
    assert "velocity_beam7" not in ds.data_vars
    assert ds["up_velocity"].attrs["original_name"] == "Velocity (Beam7|Z|Up)"

    dat_file, hdr_file = _write_nortek_ascii_pair_with_custom_velocity_labels(
        tmp_path,
        "XYZ",
        [
            "Velocity (Beam7|Z|Up)",
            "Velocity (Beam2|X|East)",
            "Velocity (Beam9|Y|North)",
        ],
    )

    ds = NortekAsciiReader(
        str(dat_file),
        str(hdr_file),
        perform_default_postprocessing=False,
    ).data

    assert {"x_velocity", "y_velocity", "z_velocity"}.issubset(ds.data_vars)
    assert "velocity_beam7" not in ds.data_vars
    assert ds["z_velocity"].attrs["original_name"] == "Velocity (Beam7|Z|Up)"


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
    assert "header" not in payload["blocks"]
    assert payload["blocks"]["attributes"]["coordinate_system"] == "BEAM"
    assert "transformation_matrix" not in payload["blocks"]["attributes"]
    assert payload["blocks"]["calibration"]["transformation_matrix"] == [
        [0.7356, -0.3677, -0.3677],
        [0.0, -0.637, 0.637],
        [0.7888, 0.7888, 0.7888],
    ]
    assert payload["blocks"]["calibration"]["magnetometer_calibration_matrix"] == [
        [0.991, 0.022, -0.0229],
        [0.022, 1.0, -0.0062],
        [-0.018, 0.0304, 0.9507],
    ]
    assert payload["blocks"]["calibration"]["compass_hard_iron_calibration"] == [
        5,
        -90,
        10,
    ]
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
