"""
Reader wrapper and helper functions for Nortek CSV exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import xarray as xr

from .base import AbstractReader


def _parse_nortek_csv_columns(df: pd.DataFrame) -> Dict:
    """
    Extract data variables from Nortek CSV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Nortek CSV data

    Returns
    -------
    Dict
        Dictionary of data variables for xarray Dataset
    """
    data_vars = {}

    # Environmental data
    for csv_col, var_name in [
        ("temperature", "temperature"),
        ("pressure", "pressure"),
        ("heading", "heading"),
        ("pitch", "pitch"),
        ("roll", "roll"),
        ("speedOfSound", "speed_of_sound"),
        ("batteryVoltage", "battery_voltage"),
    ]:
        if csv_col in df.columns:
            data_vars[var_name] = (["time"], df[csv_col].values)

    # Velocity, amplitude, correlation data for 3 beams
    for i in [1, 2, 3]:
        for data_type, prefix in [
            ("vel", "velocity"),
            ("amp", "amplitude"),
            ("corr", "correlation"),
        ]:
            csv_col = f"{data_type}Beam{i}#1"
            var_name = f"{prefix}_beam{i}"
            if csv_col in df.columns:
                data_vars[var_name] = (["time"], df[csv_col].values)

    return data_vars


def _add_nortek_variable_attributes(ds: xr.Dataset) -> xr.Dataset:
    """
    Add units and metadata attributes to Nortek dataset variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add attributes to

    Returns
    -------
    xr.Dataset
        Dataset with variable attributes added
    """
    # Environmental variable attributes
    attr_map = {
        "temperature": {"units": "degrees_C", "long_name": "Water Temperature"},
        "pressure": {"units": "dbar", "long_name": "Pressure"},
        "heading": {"units": "degrees", "long_name": "Heading"},
        "pitch": {"units": "degrees", "long_name": "Pitch"},
        "roll": {"units": "degrees", "long_name": "Roll"},
        "speed_of_sound": {"units": "m/s", "long_name": "Speed of Sound"},
        "battery_voltage": {"units": "V", "long_name": "Battery Voltage"},
    }

    for var_name, attrs in attr_map.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs.update(attrs)

    # Beam data attributes
    for i in [1, 2, 3]:
        vel_var = f"velocity_beam{i}"
        amp_var = f"amplitude_beam{i}"
        corr_var = f"correlation_beam{i}"

        if vel_var in ds.data_vars:
            ds[vel_var].attrs.update(
                {
                    "units": "m/s",
                    "long_name": f"Velocity Beam {i}",
                    "coordinate_system": "BEAM",
                }
            )
        if amp_var in ds.data_vars:
            ds[amp_var].attrs.update(
                {"units": "counts", "long_name": f"Amplitude Beam {i}"}
            )
        if corr_var in ds.data_vars:
            ds[corr_var].attrs.update(
                {"units": "%", "long_name": f"Correlation Beam {i}"}
            )

    return ds


def load_nortek_csv_data(
    file_path: Union[str, Path], header_file: Optional[str] = None
) -> xr.Dataset:
    """
    Load Nortek CSV data exported from AquaPro software.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV data file (e.g., "Average Velocity DF3.csv")
    header_file : str, optional
        Path to Units.csv file for metadata (optional)

    Returns
    -------
    xr.Dataset
        Dataset with Nortek CSV data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # Read CSV and parse time
    df = pd.read_csv(file_path, delimiter=";")
    df["datetime"] = pd.to_datetime(df["dateTime"])
    times = df["datetime"].values

    # Extract data variables
    data_vars = _parse_nortek_csv_columns(df)

    # Create dataset
    ds = xr.Dataset(data_vars, coords={"time": times})

    # Add global metadata
    ds.attrs.update(
        {
            "instrument_type": "Nortek_Aquadopp",
            "filename": str(file_path),
            "data_format": "Nortek_CSV_Export",
            "coordinate_system": "BEAM",
        }
    )

    # Extract serial number
    if "serialNumber" in df.columns:
        ds.attrs["serial_number"] = str(df["serialNumber"].iloc[0])

    # Add variable attributes
    ds = _add_nortek_variable_attributes(ds)

    print(f"  Nortek CSV: loaded {len(times)} samples from {file_path.name}")

    return ds


class NortekCsvReader(AbstractReader):
    """
    Read Nortek CSV data exported from AquaPro software.

    This class is a SeaSenseLib wrapper around the original Nortek CSV helper
    functions. The parsing logic is kept in ``load_nortek_csv_data`` and the
    class only adapts it to the common reader interface.
    """

    def __init__(
        self,
        input_file: str,
        mapping: dict | None = None,
        input_header_file: str | None = None,
        **kwargs,
    ):
        """Initialize NortekCsvReader.

        Parameters
        ----------
        input_file : str
            Path to the Nortek CSV file.
        mapping : dict, optional
            Variable name mapping dictionary.
        input_header_file : str, optional
            Optional Units.csv path. Kept for API compatibility with the
            original helper signature.
        **kwargs
            Additional base class parameters.
        """
        super().__init__(
            input_file,
            mapping,
            input_header_file=input_header_file,
            **kwargs,
        )
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        """Return valid file extensions for Nortek CSV exports."""
        return (".csv",)

    def _load_data(self) -> xr.Dataset:
        """Load the Nortek CSV file and return an xarray Dataset."""
        return load_nortek_csv_data(
            self.input_file,
            header_file=self.input_header_file,
        )

    @classmethod
    def format_key(cls) -> str:
        return "nortek-csv"

    @classmethod
    def format_name(cls) -> str:
        return "Nortek CSV"

    @classmethod
    def file_extension(cls) -> str | None:
        return None
