from __future__ import annotations

import json

import numpy as np

from seasenselib.readers import get_reader_by_format_key
from seasenselib.readers.rbr_hex_reader import RbrHexReader, read_rbr_hex


_COEFFICIENTS = [
    0.003476841520171,
    -2.55572836756e-4,
    2.576436978e-6,
    -6.8870462e-8,
]
_RAW_COUNTS = [0x686C8C, 0x686C4C, 0x686F7F]


def _bcd_byte(value: int) -> int:
    return ((value // 10) << 4) | (value % 10)


def _bcd_datetime(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
) -> bytes:
    return bytes(
        [
            _bcd_byte(year - 2000),
            _bcd_byte(month),
            _bcd_byte(day),
            _bcd_byte(hour),
            _bcd_byte(minute),
            _bcd_byte(second),
        ]
    )


def _write_rbr_hex_file(tmp_path):
    binary_header = (
        _bcd_datetime(2026, 5, 2, 13, 7, 52)
        + _bcd_datetime(2026, 8, 1, 6, 0, 0)
        + bytes([0x00, 0x00, 0x03])
        + bytes(33)
    )
    event = b"\x00\x00\x00TIM" + _bcd_datetime(2026, 5, 2, 13, 7, 58)
    samples = b"".join(value.to_bytes(3, "big") for value in _RAW_COUNTS)
    payload = binary_header + event + samples
    path = tmp_path / "sample_rbr.hex"
    path.write_text(
        "\n".join(
            [
                "RBR TR-1050 6.200 000123",
                "Host time     26/07/20 14:46:56",
                "Logger time   26/07/20 14:46:59",
                "Logging start 26/05/02 13:07:52",
                "Logging end   26/08/01 06:00:00",
                "Sample period          00:00:03",
                "Number of channels =  1, number of samples =   3, mode: Stopped",
                "Calibration  1: 0.003476841520171",
                "                -0.000255572836756",
                "                0.000002576436978",
                "                -0.000000068870462 Degrees_C",
                "Number of bytes in header 48",
                "",
                "                               Temp ",
                f"Number of bytes of data {len(event) + len(samples)}",
                "|     |     |     |",
                payload.hex().upper(),
            ]
        ),
        encoding="ascii",
    )
    return path


def test_rbr_hex_reader_metadata_and_discovery():
    assert RbrHexReader.format_key() == "rbr-hex"
    assert RbrHexReader.format_name() == "RBR HEX"
    assert RbrHexReader.file_extension() is None
    assert RbrHexReader.file_extensions() == ()
    assert RbrHexReader._get_valid_extensions() == (".hex",)
    assert get_reader_by_format_key("rbr-hex") is RbrHexReader


def test_read_rbr_hex_preserves_standalone_decode_logic(tmp_path):
    path = _write_rbr_hex_file(tmp_path)

    ds = read_rbr_hex(path)

    assert list(ds.data_vars) == ["channel_1", "temp"]
    assert ds["time"].values.tolist() == [
        np.datetime64("2026-05-02T13:07:58.000").item(),
        np.datetime64("2026-05-02T13:08:01.000").item(),
        np.datetime64("2026-05-02T13:08:04.000").item(),
    ]
    assert ds["channel_1"].values.tolist() == _RAW_COUNTS
    np.testing.assert_allclose(
        ds["temp"].values,
        np.array([22.5363262, 22.53667652, 22.53219368]),
        rtol=0,
        atol=1e-8,
    )
    assert ds.attrs["instrument_model"] == "TR-1050"
    assert ds.attrs["serial_number"] == "123"


def test_rbr_hex_reader_preserves_structured_raw_metadata(tmp_path):
    path = _write_rbr_hex_file(tmp_path)

    ds = RbrHexReader(str(path)).data

    payload = json.loads(ds.attrs["raw_metadata"])
    assert "RBR TR-1050" in payload["blocks"]["header"]
    assert payload["blocks"]["attributes"]["model"] == "TR-1050"
    assert payload["blocks"]["attributes"]["firmware"] == "6.200"
    assert payload["blocks"]["attributes"]["serial"] == "123"
    assert payload["blocks"]["attributes"]["sample_period_seconds"] == 3
    assert payload["blocks"]["configuration"]["binary_events"][0]["marker"] == "TIM"
    assert payload["blocks"]["configuration"]["binary_events"][0]["sample_index"] == 0
    assert payload["blocks"]["calibration"]["temp"]["coefficients"] == _COEFFICIENTS
    assert payload["variables"]["channel_1"]["kind"] == "raw_counts"
    assert payload["variables"]["temp"]["rbr_channel_name"] == "Temp"
    assert payload["variables"]["temp"]["units"] == "degC"
