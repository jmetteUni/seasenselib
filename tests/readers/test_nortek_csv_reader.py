from __future__ import annotations

import numpy as np

import seasenselib as ssl
from seasenselib.readers import get_reader_by_format_key
from seasenselib.readers.nortek_csv_reader import (
    NortekCsvReader,
    load_nortek_csv_data,
)


def _write_nortek_csv(tmp_path):
    csv_file = tmp_path / "Average Velocity DF3.csv"
    csv_file.write_text(
        "\n".join(
            [
                (
                    "dateTime;serialNumber;temperature;pressure;heading;pitch;roll;"
                    "speedOfSound;batteryVoltage;velBeam1#1;ampBeam1#1;"
                    "corrBeam1#1;velBeam2#1;velBeam3#1"
                ),
                (
                    "2026-07-11 12:00:00;A123;8.1;12.5;101.0;1.2;-0.5;"
                    "1450.0;12.1;0.11;55;98;0.21;0.31"
                ),
                (
                    "2026-07-11 12:00:01;A123;8.2;12.6;102.0;1.3;-0.6;"
                    "1450.2;12.0;0.12;56;97;0.22;0.32"
                ),
            ]
        ),
        encoding="utf-8",
    )
    return csv_file


def test_load_nortek_csv_data_preserves_original_helper_logic(tmp_path):
    csv_file = _write_nortek_csv(tmp_path)

    ds = load_nortek_csv_data(csv_file)

    assert ds.sizes["time"] == 2
    assert ds.attrs["instrument_type"] == "Nortek_Aquadopp"
    assert ds.attrs["data_format"] == "Nortek_CSV_Export"
    assert ds.attrs["coordinate_system"] == "BEAM"
    assert ds.attrs["serial_number"] == "A123"
    assert ds["time"].values[0] == np.datetime64("2026-07-11T12:00:00")
    assert ds["temperature"].values.tolist() == [8.1, 8.2]
    assert ds["velocity_beam1"].values.tolist() == [0.11, 0.12]
    assert ds["amplitude_beam1"].values.tolist() == [55, 56]
    assert ds["correlation_beam1"].values.tolist() == [98, 97]
    assert ds["velocity_beam1"].attrs["coordinate_system"] == "BEAM"
    assert ds["speed_of_sound"].attrs["units"] == "m/s"


def test_nortek_csv_reader_wraps_helper_as_seasenselib_reader(tmp_path):
    csv_file = _write_nortek_csv(tmp_path)

    reader = NortekCsvReader(
        str(csv_file),
        perform_default_postprocessing=False,
    )
    ds = reader.data

    assert reader.format_key() == "nortek-csv"
    assert reader.format_name() == "Nortek CSV"
    assert reader.file_extension() is None
    assert reader._get_valid_extensions() == (".csv",)
    assert ds.attrs["filename"] == str(csv_file)
    assert ds["battery_voltage"].values.tolist() == [12.1, 12.0]


def test_nortek_csv_reader_is_discoverable_by_format_key():
    assert get_reader_by_format_key("nortek-csv") is NortekCsvReader


def test_nortek_csv_reader_loads_through_public_api(tmp_path):
    csv_file = _write_nortek_csv(tmp_path)

    ds = ssl.read(
        str(csv_file),
        file_format="nortek-csv",
        use_steps=False,
    )

    assert ds.attrs["data_format"] == "Nortek_CSV_Export"
    assert ds["velocity_beam2"].values.tolist() == [0.21, 0.22]
