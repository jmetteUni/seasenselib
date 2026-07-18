from __future__ import annotations

import json

import pandas as pd
import pytest

from seasenselib.readers.rbr_ascii_reader import (
    RbrAsciiReader,
    _parse_rbr_datetime_series,
)


def _write_rbr_ascii_file(tmp_path):
    path = tmp_path / "sample_rbr.txt"
    path.write_text(
        "\n".join(
            [
                "Model=TR-1050",
                "Firmware=6.20",
                "Serial=013875",
                "HostVersion=(Ruskin version number - 2.24.1.202509250144)",
                "",
                "HostTime=17-Jul-2026 13:45:13.000",
                "LoggerTime=17-Jul-2026 13:44:23.000",
                "LoggingStartTime=02-May-2026 13:00:37.000",
                "LoggingEndTime=01-Aug-2026 06:00:00.000",
                "LoggingSamplingPeriod=00:00:03",
                "NumberOfChannels=1",
                "Channel[1].name=Temperature",
                "Channel[1].calibration=0.003477834548534 -2.54919099523E-4 2.576013252E-6 -7.792057E-8",
                "Channel[1].units=°C (Degrees_C)",
                "",
                "TimeStamp[1].time=02-May-2026 13:00:43.000",
                "TimeStamp[1].sample=1",
                "TimeStamp[1].type=TIME STAMP",
                "ResetStamp[1].time=17-Jul-2026 13:29:07.000",
                "ResetStamp[1].sample=3",
                "ResetStamp[1].type=STOP STAMP",
                "",
                "NumberOfSamples=2",
                "",
                "             Date & Time         Temp",
                " 2026-05-02 13:01:25.000    22.7661903",
                " 2026-05-02 13:01:28.123Z   22.7606206",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_parse_rbr_datetime_series_supports_explicit_formats():
    result = _parse_rbr_datetime_series(
        [
            "2026-05-02 13:01:25.000",
            "2026-05-02 13:01:28.123Z",
            "2026/05/02 13:01:31",
        ]
    )

    assert result[0] == pd.Timestamp("2026-05-02 13:01:25")
    assert result[1] == pd.Timestamp("2026-05-02 13:01:28.123")
    assert result[2] == pd.Timestamp("2026-05-02 13:01:31")


def test_parse_rbr_datetime_series_rejects_ambiguous_day_month_dates():
    with pytest.raises(ValueError):
        _parse_rbr_datetime_series(["02-05-2026 13:01:25.000"])


def test_rbr_ascii_reader_reads_millisecond_timestamps(tmp_path):
    path = _write_rbr_ascii_file(tmp_path)

    ds = RbrAsciiReader(str(path), perform_default_postprocessing=False).data

    assert list(ds.data_vars) == ["Temp"]
    assert ds.time.values[0] == pd.Timestamp("2026-05-02 13:01:25").to_datetime64()
    assert ds.time.values[1] == pd.Timestamp("2026-05-02 13:01:28.123").to_datetime64()
    assert ds["Temp"].attrs["original_name"] == "Temp"
    assert ds["Temp"].attrs["rbr_channel_name"] == "Temperature"
    assert "units" not in ds["Temp"].attrs
    assert ds["Temp"].attrs["rbr_original_units"] == "°C (Degrees_C)"


def test_rbr_ascii_reader_leaves_mapping_to_pipeline(tmp_path):
    path = _write_rbr_ascii_file(tmp_path)

    raw_ds = RbrAsciiReader(str(path), perform_default_postprocessing=False).data
    processed_ds = RbrAsciiReader(str(path)).data

    assert "Temp" in raw_ds.data_vars
    assert "temperature" in processed_ds.data_vars
    assert "Temp" not in processed_ds.data_vars


def test_rbr_ascii_reader_preserves_structured_raw_metadata(tmp_path):
    path = _write_rbr_ascii_file(tmp_path)

    ds = RbrAsciiReader(str(path)).data

    payload = json.loads(ds.attrs["raw_metadata"])
    assert payload["blocks"]["attributes"]["model"] == "TR-1050"
    assert payload["blocks"]["attributes"]["firmware"] == "6.20"
    assert payload["blocks"]["attributes"]["serial"] == "013875"
    assert payload["blocks"]["attributes"]["number_of_samples"] == 2
    assert payload["blocks"]["configuration"]["time_stamps"][0]["sample"] == 1
    assert payload["blocks"]["configuration"]["reset_stamps"][0]["type"] == "STOP STAMP"
    assert payload["blocks"]["calibration"]["Temp"] == [
        0.003477834548534,
        -2.54919099523e-4,
        2.576013252e-6,
        -7.792057e-8,
    ]
    assert payload["variables"]["Temp"] == {
        "original_name": "Temp",
        "rbr_channel_name": "Temperature",
        "channel_number": 1,
        "units": "°C (Degrees_C)",
    }


def test_rbr_ascii_reader_tolerates_non_utf8_header_bytes(tmp_path):
    path = tmp_path / "sample_rbr_latin1.txt"
    path.write_bytes(
        "\n".join(
            [
                "Model=TR-1050",
                "NumberOfChannels=1",
                "Channel[1].name=Temperature",
                "Channel[1].units=",
                "",
                "NumberOfSamples=1",
                "",
                "             Date & Time         Temp",
                " 2026-05-02 13:01:25.000Z    22.7661903",
                "",
            ]
        ).encode("utf-8").replace(b"Channel[1].units=", b"Channel[1].units=\xb0C ")
    )

    ds = RbrAsciiReader(str(path), perform_default_postprocessing=False).data

    assert ds.time.values[0] == pd.Timestamp("2026-05-02 13:01:25").to_datetime64()
    assert ds["Temp"].attrs["rbr_original_units"] == "�C"
