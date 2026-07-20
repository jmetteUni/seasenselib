"""Reader for RBR TR-1050 style binary HEX files."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import xarray as xr

import seasenselib.parameters as params
from seasenselib.readers.base import AbstractReader


_EVENT_MARKER_LEN = 12
_BINARY_HEADER_LEN = 48


def _bcd(byte: int) -> int:
    """Decode one packed BCD byte."""
    return (byte >> 4) * 10 + (byte & 0xF)


def _bcd_datetime(data: bytes) -> dt.datetime:
    """Decode six BCD bytes as YY MM DD HH MM SS with year base 2000."""
    return dt.datetime(
        2000 + _bcd(data[0]),
        _bcd(data[1]),
        _bcd(data[2]),
        _bcd(data[3]),
        _bcd(data[4]),
        _bcd(data[5]),
    )


def _bcd_timedelta(data: bytes) -> int:
    """Decode three BCD bytes as HH MM SS and return seconds."""
    return _bcd(data[0]) * 3600 + _bcd(data[1]) * 60 + _bcd(data[2])


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _parse_text_header(lines: list[str]) -> dict[str, Any]:
    """Extract metadata from the human-readable RBR HEX header."""
    meta: dict[str, Any] = {
        "model": None,
        "firmware": None,
        "serial": None,
        "n_channels": 1,
        "n_samples": 0,
        "n_bytes_data": 0,
        "channel_names": {},
        "column_names": [],
        "channel_units": {},
        "channel_calibration": {},
        "host_time": None,
        "logger_time": None,
        "logging_start": None,
        "logging_end": None,
        "sample_period_str": None,
        "timestamps": [],
    }

    cal_channel: int | None = None
    cal_coeffs: list[float] = []
    cal_units = ""

    for raw_line in lines:
        is_indented = bool(raw_line) and raw_line[0] in (" ", "\t")
        line = raw_line.strip()

        match = re.match(r"RBR\s+([\w-]+)\s+([\d.]+)\s+0*(\d+)", line)
        if match:
            meta["model"] = match.group(1)
            meta["firmware"] = match.group(2)
            meta["serial"] = match.group(3)
            continue

        match = re.match(r"Host time\s+(\S+\s+\S+)", line)
        if match:
            meta["host_time"] = match.group(1)
            continue

        match = re.match(r"Logger time\s+(\S+\s+\S+)", line)
        if match:
            meta["logger_time"] = match.group(1)
            continue

        match = re.match(r"Logging start\s+(\S+\s+\S+)", line)
        if match:
            meta["logging_start"] = match.group(1)
            continue

        match = re.match(r"Logging end\s+(\S+\s+\S+)", line)
        if match:
            meta["logging_end"] = match.group(1)
            continue

        match = re.match(r"Sample period\s+(\S+)", line)
        if match:
            meta["sample_period_str"] = match.group(1)
            continue

        match = re.match(
            r"Number of channels\s*=\s*(\d+).*number of samples\s*=\s*(\d+)",
            line,
        )
        if match:
            meta["n_channels"] = int(match.group(1))
            meta["n_samples"] = int(match.group(2))
            continue

        match = re.match(r"Number of bytes of data\s+(\d+)", line)
        if match:
            meta["n_bytes_data"] = int(match.group(1))
            continue

        if (
            not meta["column_names"]
            and meta["n_bytes_data"] == 0
            and raw_line.startswith(" ")
            and line
            and all(token.isalpha() for token in line.split())
        ):
            meta["column_names"] = line.split()
            continue

        match = re.match(r"Channel\[(\d+)\]\.name=(.+)", line)
        if match:
            meta["channel_names"][int(match.group(1))] = match.group(2).strip()
            continue

        match = re.match(r"Channel\[(\d+)\]\.units=(.+)", line)
        if match:
            meta["channel_units"][int(match.group(1))] = match.group(2).strip()
            continue

        match = re.match(r"Channel\[(\d+)\]\.calibration=(.+)", line)
        if match:
            cal_channel = int(match.group(1))
            cal_coeffs = [float(item) for item in match.group(2).split()]
            meta["channel_calibration"][cal_channel] = cal_coeffs
            continue

        if cal_channel is not None and is_indented and line:
            float_parts = [float(part) for part in line.split() if _is_float(part)]
            non_float = [part for part in line.split() if not _is_float(part)]
            if non_float:
                cal_units = non_float[-1].replace("_", " ")
            cal_coeffs.extend(float_parts)
            meta["channel_calibration"][cal_channel] = list(cal_coeffs)
            meta["channel_units"][cal_channel] = cal_units
            continue
        if not is_indented:
            cal_channel = None
            cal_units = ""

        match = re.match(r"Calibration\s+(\d+):\s+([-\d.Ee+]+)", line)
        if match:
            cal_channel = int(match.group(1))
            cal_coeffs = [float(match.group(2))]
            meta["channel_calibration"][cal_channel] = list(cal_coeffs)
            continue

        match = re.match(
            r"(?:Timestamp|Reset stamp)\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})"
            r"\s+at sample\s+(\d+)\s+of type:\s+(.+)",
            line,
        )
        if match:
            meta["timestamps"].append(
                {
                    "time": match.group(1),
                    "sample": int(match.group(2)),
                    "type": match.group(3).strip(),
                }
            )

    return meta


def _apply_rbr_cal(raw_counts: np.ndarray, coeffs: list[float]) -> np.ndarray:
    """Apply the original RBR Steinhart-Hart calibration."""
    if len(coeffs) != 4:
        raise ValueError(f"Expected 4 calibration coefficients, got {len(coeffs)}")

    c0, c1, c2, c3 = coeffs
    x = raw_counts.astype(np.float64) / 16777215.0
    ln_r = np.log((1.0 - x) / x)
    inv_tk = c0 + c1 * ln_r + c2 * ln_r**2 + c3 * ln_r**3
    return 1.0 / inv_tk - 273.15


def _parse_binary_header(raw: bytes) -> tuple[dt.datetime, dt.datetime, int]:
    """Parse the fixed 48-byte binary header."""
    log_start = _bcd_datetime(raw[0:6])
    log_end = _bcd_datetime(raw[6:12])
    period_seconds = _bcd_timedelta(raw[12:15])
    return log_start, log_end, period_seconds


def _find_events(data: bytes, record_bytes: int) -> list[tuple[int, str, dt.datetime]]:
    """Return event records as sample index, marker, and timestamp tuples."""
    events: list[tuple[int, str, dt.datetime]] = []
    pos = 0
    sample_count = 0

    while pos < len(data):
        if (
            pos + _EVENT_MARKER_LEN <= len(data)
            and data[pos : pos + 3] == b"\x00\x00\x00"
            and data[pos + 3 : pos + 6].isalpha()
        ):
            marker = data[pos + 3 : pos + 6].decode("ascii")
            try:
                events.append(
                    (sample_count, marker, _bcd_datetime(data[pos + 6 : pos + 12]))
                )
            except (ValueError, OverflowError):
                pass
            pos += _EVENT_MARKER_LEN
        elif pos + record_bytes <= len(data):
            sample_count += 1
            pos += record_bytes
        else:
            break

    return events


def _build_time_array(
    n_samples: int,
    period_seconds: int,
    events: list[tuple[int, str, dt.datetime]],
    fallback_start: dt.datetime,
) -> np.ndarray:
    """Build sample times using TIM events as anchors."""
    times = np.empty(n_samples, dtype="datetime64[ms]")
    period_ms = int(period_seconds * 1000)
    anchors = [
        (index, timestamp)
        for index, marker, timestamp in events
        if marker == "TIM"
    ]

    if not anchors:
        start = np.datetime64(fallback_start, "ms")
        times[:] = start + np.arange(n_samples, dtype="int64") * period_ms
        return times

    for index, (anchor_index, anchor_timestamp) in enumerate(anchors):
        start = np.datetime64(anchor_timestamp, "ms")
        next_anchor_index = (
            anchors[index + 1][0] if index + 1 < len(anchors) else n_samples
        )
        block_len = next_anchor_index - anchor_index
        if block_len > 0:
            times[anchor_index:next_anchor_index] = (
                start + np.arange(block_len, dtype="int64") * period_ms
            )

    first_anchor_index, first_anchor_timestamp = anchors[0]
    if first_anchor_index > 0:
        start = (
            np.datetime64(first_anchor_timestamp, "ms")
            - first_anchor_index * period_ms
        )
        times[:first_anchor_index] = (
            start + np.arange(first_anchor_index, dtype="int64") * period_ms
        )

    return times


def _parse_sample_period_seconds(period_text: str) -> int:
    hours, minutes, seconds = (int(value) for value in period_text.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _safe_variable_name(name: str) -> str:
    return re.sub(r"\W+", "_", name.lower()).strip("_")


def _event_metadata(events: list[tuple[int, str, dt.datetime]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": sample_index,
            "marker": marker,
            "time": timestamp.isoformat(sep=" "),
        }
        for sample_index, marker, timestamp in events
    ]


def _read_rbr_hex_dataset(file_path: str | Path) -> tuple[xr.Dataset, dict[str, Any]]:
    """Read an RBR HEX file and return the dataset plus parsed metadata."""
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    data_start_line = None
    for index, line in enumerate(lines):
        if line.startswith("|") and line.count("|") >= 2:
            data_start_line = index + 1
            break

    if data_start_line is None:
        raise ValueError(f"Could not find hex data section in {path.name}")

    header_lines = lines[:data_start_line]
    raw_header = "\n".join(header_lines).rstrip("\n")
    meta = _parse_text_header(header_lines)

    hex_text = re.sub(r"[^0-9A-Fa-f]", "", "".join(lines[data_start_line:]))
    raw = bytes.fromhex(hex_text)
    if len(raw) < _BINARY_HEADER_LEN:
        raise ValueError(
            f"RBR HEX data section in {path.name} is shorter than the "
            f"{_BINARY_HEADER_LEN}-byte binary header."
        )

    log_start, log_end, period_seconds = _parse_binary_header(raw[:_BINARY_HEADER_LEN])
    if period_seconds == 0 and meta.get("sample_period_str"):
        period_seconds = _parse_sample_period_seconds(meta["sample_period_str"])

    n_channels = meta["n_channels"]
    n_samples = meta["n_samples"]
    record_bytes = 3 * n_channels
    data_section = raw[_BINARY_HEADER_LEN:]
    events = _find_events(data_section, record_bytes)

    channel_data: list[list[int]] = [[] for _ in range(n_channels)]
    pos = 0
    while pos < len(data_section):
        if (
            pos + _EVENT_MARKER_LEN <= len(data_section)
            and data_section[pos : pos + 3] == b"\x00\x00\x00"
            and data_section[pos + 3 : pos + 6].isalpha()
        ):
            pos += _EVENT_MARKER_LEN
        elif pos + record_bytes <= len(data_section):
            for channel_index in range(n_channels):
                offset = pos + channel_index * 3
                value = int.from_bytes(data_section[offset : offset + 3], "big")
                channel_data[channel_index].append(value)
            pos += record_bytes
        else:
            break

    n_actual = len(channel_data[0])
    if n_actual != n_samples:
        warnings.warn(
            f"{path.name}: expected {n_samples} samples, decoded {n_actual}. "
            "Time array built from decoded count.",
            stacklevel=2,
        )
        n_samples = n_actual

    times = _build_time_array(n_samples, period_seconds, events, log_start)
    data_vars: dict[str, xr.Variable] = {}
    channel_metadata: list[dict[str, Any]] = []
    column_names = meta.get("column_names", [])

    for channel_index in range(n_channels):
        channel_number = channel_index + 1
        channel_name = (
            meta["channel_names"].get(channel_number)
            or (
                column_names[channel_index]
                if channel_index < len(column_names)
                else None
            )
            or f"Channel {channel_number}"
        )
        channel_units = meta["channel_units"].get(channel_number, "")
        calibration = meta["channel_calibration"].get(channel_number, [])
        raw_array = np.array(channel_data[channel_index], dtype=np.uint32)
        raw_name = f"channel_{channel_number}"

        data_vars[raw_name] = xr.Variable(
            params.TIME,
            raw_array,
            attrs={
                "long_name": f"{channel_name} raw ADC count",
                "comment": "Raw 24-bit unsigned integer from instrument.",
                "rbr_channel_number": channel_number,
                "rbr_channel_name": channel_name,
                "calibration_coefficients": " ".join(
                    str(value) for value in calibration
                ),
                "units_after_calibration": channel_units,
            },
        )

        physical_name = _safe_variable_name(channel_name)
        if physical_name == raw_name:
            physical_name = f"{raw_name}_phys"

        if len(calibration) == 4:
            physical_data = _apply_rbr_cal(raw_array, calibration)
            calibration_method = (
                "Steinhart-Hart: x=raw/16777215, R=(1-x)/x, "
                "1/T_K=c0+c1*ln(R)+c2*ln(R)^2+c3*ln(R)^3, T=1/T_K-273.15"
            )
            calibration_units = "degC"
        else:
            physical_data = raw_array.astype(np.float64) / 16777215.0
            calibration_method = (
                "raw / 16777215 (calibration not applied: unexpected number "
                "of coefficients)"
            )
            calibration_units = channel_units

        data_vars[physical_name] = xr.Variable(
            params.TIME,
            physical_data,
            attrs={
                "long_name": channel_name,
                "units": calibration_units,
                "rbr_channel_number": channel_number,
                "rbr_channel_name": channel_name,
                "rbr_original_units": channel_units,
                "calibration_method": calibration_method,
                "calibration_coefficients": " ".join(
                    str(value) for value in calibration
                ),
            },
        )

        channel_metadata.append(
            {
                "channel_number": channel_number,
                "channel_name": channel_name,
                "raw_variable_name": raw_name,
                "variable_name": physical_name,
                "units": channel_units,
                "calibrated_units": calibration_units,
                "calibration": calibration,
                "calibration_method": calibration_method,
            }
        )

    attrs = {
        "instrument_model": meta.get("model", "UNK"),
        "instrument_firmware": meta.get("firmware", "UNK"),
        "serial_number": meta.get("serial", "UNK"),
        "logging_start": str(log_start),
        "logging_end": str(log_end),
        "sample_period_seconds": period_seconds,
        "n_channels": n_channels,
        "source_file": path.name,
        "reader": "seasenselib.readers.rbr_hex_reader.read_rbr_hex",
    }
    ds = xr.Dataset(data_vars, coords={params.TIME: times}, attrs=attrs)

    metadata = {
        "raw_header": raw_header,
        "header": meta,
        "binary_header": {
            "logging_start": log_start.isoformat(sep=" "),
            "logging_end": log_end.isoformat(sep=" "),
            "sample_period_seconds": period_seconds,
        },
        "events": _event_metadata(events),
        "channels": channel_metadata,
        "number_of_samples_decoded": n_actual,
    }
    return ds, metadata


def read_rbr_hex(file_path: str | Path) -> xr.Dataset:
    """Read an RBR binary HEX file and return an xarray Dataset."""
    ds, _ = _read_rbr_hex_dataset(file_path)
    return ds


def _rbr_hex_raw_metadata_blocks(metadata: dict[str, Any]) -> dict[str, Any]:
    header = metadata["header"]
    attributes = {
        "model": header.get("model"),
        "firmware": header.get("firmware"),
        "serial": header.get("serial"),
        "host_time": header.get("host_time"),
        "logger_time": header.get("logger_time"),
        "logging_start_text": header.get("logging_start"),
        "logging_end_text": header.get("logging_end"),
        "sample_period_text": header.get("sample_period_str"),
        "number_of_channels": header.get("n_channels"),
        "number_of_samples": header.get("n_samples"),
        "number_of_samples_decoded": metadata.get("number_of_samples_decoded"),
        "number_of_bytes_data": header.get("n_bytes_data"),
        **metadata["binary_header"],
    }
    attributes = {key: value for key, value in attributes.items() if value is not None}

    calibration = {}
    for channel in metadata["channels"]:
        if channel["calibration"]:
            calibration[channel["variable_name"]] = {
                "channel_number": channel["channel_number"],
                "coefficients": channel["calibration"],
                "units_after_calibration": channel["units"],
                "method": channel["calibration_method"],
            }

    configuration = {}
    if header.get("timestamps"):
        configuration["header_events"] = header["timestamps"]
    if metadata.get("events"):
        configuration["binary_events"] = metadata["events"]

    blocks = {"attributes": attributes}
    if calibration:
        blocks["calibration"] = calibration
    if configuration:
        blocks["configuration"] = configuration
    return blocks


def _rbr_hex_raw_variable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    variables = {}
    for channel in metadata["channels"]:
        common = {
            "channel_number": channel["channel_number"],
            "rbr_channel_name": channel["channel_name"],
        }
        variables[channel["raw_variable_name"]] = {
            **common,
            "kind": "raw_counts",
            "units_after_calibration": channel["units"],
        }
        variables[channel["variable_name"]] = {
            **common,
            "original_name": channel["channel_name"],
            "units": channel["calibrated_units"],
            "rbr_original_units": channel["units"],
            "calibration_method": channel["calibration_method"],
        }
    return variables


class RbrHexReader(AbstractReader):
    """Read RBR TR-1050 style binary HEX files."""

    def __init__(
        self,
        input_file: str,
        mapping: dict | None = None,
        **kwargs,
    ):
        self._raw_header = None
        self._raw_metadata_blocks: dict[str, Any] = {}
        self._raw_metadata_variables: dict[str, Any] = {}
        super().__init__(input_file, mapping, **kwargs)
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        return (".hex",)

    def _load_data(self) -> xr.Dataset:
        ds, metadata = _read_rbr_hex_dataset(self.input_file)
        self._raw_header = metadata["raw_header"]
        self._raw_metadata_blocks = _rbr_hex_raw_metadata_blocks(metadata)
        self._raw_metadata_variables = _rbr_hex_raw_variable_metadata(metadata)
        return ds

    @classmethod
    def format_key(cls) -> str:
        return "rbr-hex"

    @classmethod
    def format_name(cls) -> str:
        return "RBR HEX"

    @classmethod
    def file_extension(cls) -> str | None:
        return None

    @classmethod
    def format_mappings(cls) -> dict[str, list[str]]:
        return {
            params.TEMPERATURE: ["temp", "temperature"],
        }
