"""
Module for reading RBR ASCII data files into xarray Datasets.
"""

from __future__ import annotations
import re
import pandas as pd
import xarray as xr
import seasenselib.parameters as params
from .base import AbstractReader


_RBR_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
)


def _parse_rbr_datetime_series(values) -> pd.DatetimeIndex:
    """Parse RBR ASCII timestamps with explicit, non-ambiguous formats."""
    text = pd.Series(values, dtype="string").str.strip()
    text = text.str.replace(r"Z$", "", regex=True)

    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    remaining = text.notna()
    for fmt in _RBR_DATETIME_FORMATS:
        if not remaining.any():
            break
        converted = pd.to_datetime(text[remaining], format=fmt, errors="coerce")
        matched = converted.notna()
        if matched.any():
            matched_index = converted[matched].index
            parsed.loc[matched_index] = converted.loc[matched_index]
            remaining.loc[matched_index] = False

    if remaining.any():
        examples = text[remaining].dropna().head(3).tolist()
        raise ValueError(
            "Could not parse RBR ASCII datetime values with supported formats: "
            f"{examples}"
        )

    return pd.DatetimeIndex(parsed)


class RbrAsciiReader(AbstractReader):
    """ Reads RBR ASCII data from an ASCII file into an xarray Dataset.

    This class reads RBR ASCII data files, extracts the datetime and data columns,
    and organizes the data into an xarray Dataset. It handles the conversion of
    timestamps to datetime objects and assigns metadata according to CF conventions.

    Attributes
    ----------
    data : xr.Dataset
        The xarray Dataset containing the sensor data.
    input_file : str
        The path to the input file containing the RBR ASCII data.
    mapping : dict, optional
        A dictionary mapping names used in the input file to standard names.

    Methods
    -------
    __init__(input_file: str, mapping: dict | None = None):
        Initializes the RbrAsciiReader with the input file and optional mapping.
    _load_data():
        Reads the RBR ASCII data file, processes the data, and creates an xarray Dataset.
    
    Properties
    ----------
    data : xr.Dataset (read-only)
        Returns the xarray Dataset containing the sensor data.
        For backward compatibility, get_data() method is also available but deprecated.
    """

    def __init__(self, input_file: str,
                 mapping: dict | None = None,
                 **kwargs):
        """Initialize RbrAsciiReader.
        
        Parameters
        ----------
        input_file : str
            The path to the input file containing the RBR ASCII data.
        mapping : dict, optional
            A dictionary mapping names used in the input file to standard names.
        **kwargs
            Additional base class parameters:
            
            - input_header_file : str | None
                Path to separate header file (if applicable).
            - perform_default_postprocessing : bool, default=True
                Whether to perform default post-processing.
            - rename_variables : bool, default=True
                Whether to rename variables to standard names.
            - assign_metadata : bool, default=True
                Whether to assign CF-compliant metadata.
            - sort_variables : bool, default=True
                Whether to sort variables alphabetically.
        """
        super().__init__(input_file, mapping, **kwargs)
        self._rbr_header_metadata = {}
        self._raw_metadata_blocks = {}
        self._raw_metadata_variables = {}
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...] | None:
        """Return valid file extensions for RBR ASCII files."""
        return ('.dat', '.txt', '.asc', '.csv')

    @classmethod
    def _is_extension_validation_strict(cls) -> bool:
        """ASCII formats can have various extensions, so warn only."""
        return False

    def _create_xarray_dataset(self, df, columns) -> xr.Dataset:
        """
        Converts a pandas DataFrame to an xarray Dataset.
        Assumes 'Datetime' as the index of the DataFrame, 
        which will be used as the time dimension.
        """

        # Ensure 'Datetime' is the index; if not, set it
        if 'time' not in df.index.names:
            df = df.set_index('time')

        # Rename columns as specified
        #df.rename(columns=params.rename_list, inplace=True)

        # Convert DataFrame to xarray Dataset
        ds = xr.Dataset.from_dataframe(df)

        for column in columns:
            variable = column["variable_name"]
            if variable not in ds:
                continue
            ds[variable].attrs["original_name"] = column["raw_name"]
            if column.get("channel_name"):
                ds[variable].attrs["rbr_channel_name"] = column["channel_name"]
            if column.get("units"):
                ds[variable].attrs["rbr_original_units"] = column["units"]

        # Perform default post-processing
        return ds

    def _parse_data(self, file_path, data_start_index, columns) -> pd.DataFrame:
        """
        Read the data table from an RBR ASCII file.

        The first two fields are the ISO-like date and time columns. The time
        field may include milliseconds and a trailing ``Z`` marker.
        """
        column_names = [column["variable_name"] for column in columns]
        self._validate_data_column_count(file_path, data_start_index, column_names)
        data = pd.read_csv(
            file_path,
            delimiter=r"\s+",
            names=["Date", "Time"] + column_names,
            skiprows=data_start_index,
            encoding="utf-8",
            encoding_errors="replace",
        )

        # Concatenate 'Date' and 'Time' columns to create a 'Datetime'
        # column and convert it to datetime type
        data[params.TIME] = _parse_rbr_datetime_series(
            data["Date"].astype("string") + " " + data["Time"].astype("string")
        )

        # Remove original 'Date' and 'Time' columns
        data.drop(['Date', 'Time'], axis=1, inplace=True) 
        data.set_index(params.TIME, inplace=True)

        return data

    def _validate_data_column_count(self, file_path, data_start_index, column_names):
        """Ensure the parsed table header matches the first data line."""
        expected_count = 2 + len(column_names)
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                if line_number <= data_start_index:
                    continue
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != expected_count:
                    raise ValueError(
                        f"RBR ASCII data line {line_number} has {len(fields)} "
                        f"columns, but the table header defines "
                        f"{expected_count} columns."
                    )
                return

    def _read_header(self, file_path):
        """Parse RBR ASCII header metadata and table column definitions."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            lines = file.readlines()

        data_header_index = self._find_data_header_index(lines)
        if data_header_index is None:
            raise ValueError("Could not find RBR ASCII data header line containing 'Date & Time'.")

        header_lines = lines[:data_header_index]
        table_columns = self._parse_table_columns(lines[data_header_index])
        metadata = self._parse_header_metadata(header_lines)
        columns = self._normalise_table_columns(table_columns, metadata["channels"])

        self._rbr_header_metadata = metadata
        self._raw_metadata_blocks = self._raw_metadata_blocks_from_header(
            metadata,
            columns,
        )
        self._raw_metadata_variables = self._raw_variable_metadata(columns)

        return data_header_index + 1, columns

    def _find_data_header_index(self, lines):
        """Find the RBR data table header row."""
        for index, line in enumerate(lines):
            if re.search(r"\bDate\s*&\s*Time\b", line, flags=re.IGNORECASE):
                return index
        return None

    def _parse_table_columns(self, line):
        """Extract data variable column labels from the RBR table header."""
        match = re.search(
            r"\bDate\s*&\s*Time\b(?P<columns>.*)$",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Could not parse RBR ASCII data header line: {line!r}")

        column_text = match.group("columns").strip()
        if not column_text:
            raise ValueError("RBR ASCII data header does not define data columns.")

        return [item for item in re.split(r"(?:\s{2,}|\t+)", column_text) if item]

    def _parse_header_metadata(self, header_lines):
        """Parse RBR key/value metadata into structured blocks."""
        attributes = {}
        channels = {}
        time_stamps = {}
        reset_stamps = {}

        for line in header_lines:
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                continue

            raw_key, value = stripped.split("=", 1)
            raw_key = raw_key.strip()
            value = value.strip()

            channel_match = re.match(r"^Channel\[(?P<number>\d+)\]\.(?P<field>.+)$", raw_key)
            if channel_match:
                number = int(channel_match.group("number"))
                field = self._normalise_metadata_key(channel_match.group("field"))
                channels.setdefault(number, {})[field] = self._parse_metadata_value(field, value)
                continue

            stamp_match = re.match(
                r"^(?P<kind>TimeStamp|ResetStamp)\[(?P<number>\d+)\]\.(?P<field>.+)$",
                raw_key,
            )
            if stamp_match:
                stamps = time_stamps if stamp_match.group("kind") == "TimeStamp" else reset_stamps
                number = int(stamp_match.group("number"))
                field = self._normalise_metadata_key(stamp_match.group("field"))
                stamps.setdefault(number, {})[field] = self._parse_metadata_value(field, value)
                continue

            key = self._normalise_metadata_key(raw_key)
            attributes[key] = self._parse_metadata_value(key, value)

        return {
            "attributes": attributes,
            "channels": channels,
            "time_stamps": time_stamps,
            "reset_stamps": reset_stamps,
        }

    def _parse_metadata_value(self, key, value):
        """Parse simple numeric metadata while preserving datetime/unit strings."""
        if key == "calibration":
            parsed = self._parse_numeric_values(value)
            return parsed if parsed else value

        if key in {"number_of_channels", "number_of_samples", "sample"}:
            try:
                return int(value)
            except ValueError:
                return value

        return value

    def _parse_numeric_values(self, value):
        """Parse a whitespace-separated list of numeric metadata values."""
        values = []
        for item in value.split():
            try:
                values.append(float(item))
            except ValueError:
                return []
        return values

    def _normalise_metadata_key(self, key):
        """Convert RBR metadata keys into stable snake_case keys."""
        key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip())
        return re.sub(r"[^0-9A-Za-z]+", "_", key).strip("_").lower()

    def _normalise_table_columns(self, table_columns, channels):
        """Build unique dataset column definitions from table and channel metadata."""
        ordered_channel_numbers = sorted(channels)
        columns = []

        for index, raw_name in enumerate(table_columns):
            channel_number = (
                ordered_channel_numbers[index] if index < len(ordered_channel_numbers) else None
            )
            channel = channels.get(channel_number, {}) if channel_number is not None else {}
            channel_name = channel.get("name")
            columns.append(
                {
                    "raw_name": raw_name,
                    "channel_name": channel_name,
                    "variable_name": raw_name,
                    "units": channel.get("units"),
                    "channel_number": channel_number,
                }
            )

        seen = {}
        for column in columns:
            variable_name = column["variable_name"]
            if variable_name in seen:
                seen[variable_name] += 1
                column["variable_name"] = f"{variable_name}_{seen[variable_name]}"
            else:
                seen[variable_name] = 0

        return columns

    def _raw_metadata_blocks_from_header(self, metadata, columns):
        """Build structured raw metadata blocks for the pipeline."""
        blocks = {
            "attributes": metadata["attributes"],
        }

        configuration = {}
        if metadata["time_stamps"]:
            configuration["time_stamps"] = [
                metadata["time_stamps"][key]
                for key in sorted(metadata["time_stamps"])
            ]
        if metadata["reset_stamps"]:
            configuration["reset_stamps"] = [
                metadata["reset_stamps"][key]
                for key in sorted(metadata["reset_stamps"])
            ]
        if configuration:
            blocks["configuration"] = configuration

        calibration = {}
        for column in columns:
            channel_number = column.get("channel_number")
            channel = metadata["channels"].get(channel_number, {})
            if not channel or "calibration" not in channel:
                continue
            calibration[column["variable_name"]] = channel["calibration"]
        if calibration:
            blocks["calibration"] = calibration

        return blocks

    def _raw_variable_metadata(self, columns):
        """Build raw metadata for parsed data columns."""
        variables = {}
        for column in columns:
            metadata = {
                "original_name": column["raw_name"],
            }
            if column.get("channel_name"):
                metadata["rbr_channel_name"] = column["channel_name"]
            if column.get("channel_number"):
                metadata["channel_number"] = column["channel_number"]
            if column.get("units"):
                metadata["units"] = column["units"]
            variables[column["variable_name"]] = metadata
        return variables

    def _load_data(self) -> xr.Dataset:
        """Load the RBR ASCII data and return an xarray Dataset.
        
        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        data_start_index, columns = self._read_header(self.input_file)
        data = self._parse_data(self.input_file, data_start_index, columns)
        ds = self._create_xarray_dataset(data, columns)
        return ds

    @classmethod
    def format_key(cls) -> str:
        return 'rbr-ascii'

    @classmethod
    def format_name(cls) -> str:
        return 'RBR ASCII'

    @classmethod
    def file_extension(cls) -> str | None:
        return None
