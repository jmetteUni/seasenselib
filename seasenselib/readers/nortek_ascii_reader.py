"""
Module for reading Nortek ASCII data files into xarray Datasets.
"""

from __future__ import annotations
import re
import pandas as pd
import xarray as xr
import seasenselib.parameters as params
from .base import AbstractReader


class NortekAsciiReader(AbstractReader):
    """ Reads Nortek ASCII data from a .dat file into a xarray Dataset. 
    
    This class reads Nortek ASCII data files, extracts column names and units from a .hdr file, 
    and organizes the data into an xarray Dataset. It handles duplicate column names by making 
    them unique, converts timestamps to datetime objects, and assigns metadata according to 
    CF conventions.
    
    Attributes
    ----------
    data : xr.Dataset
        The xarray Dataset containing the sensor data.
    dat_file_path : str
        The path to the .dat file containing the Nortek ASCII data.
    header_file_path : str 
        The path to the .hdr file containing the header information for the Nortek ASCII data.
    
    Methods
    -------
    __init__(dat_file_path, header_file_path):
        Initializes the NortekAsciiReader with the paths to the .dat and .hdr files.
    _load_data():
        Reads the .dat and .hdr files, processes the data, and creates an xarray Dataset.
    
    Properties
    ----------
    data : xr.Dataset (read-only)
        Returns the xarray Dataset containing the sensor data.
        For backward compatibility, get_data() method is also available but deprecated.
    
    file_type : str
        A string indicating the type of file being read, in this case, 'Nortek ASCII'.
    """

    def __init__(self, dat_file_path: str,
                 header_file_path: str,
                 mapping: dict | None = None,
                 **kwargs):
        """Initialize NortekAsciiReader.
        
        Parameters
        ----------
        dat_file_path : str
            Path to the .dat file.
        header_file_path : str
            Path to the .hdr file.
        mapping : dict, optional
            Variable name mapping dictionary.
        **kwargs
            Additional base class parameters:
            
            - perform_default_postprocessing : bool, default=True
                Whether to perform default post-processing.
            - rename_variables : bool, default=True
                Whether to rename variables to standard names.
            - assign_metadata : bool, default=True
                Whether to assign CF-compliant metadata.
            - sort_variables : bool, default=True
                Whether to sort variables alphabetically.
        """
        super().__init__(dat_file_path, mapping, input_header_file=header_file_path, **kwargs)
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        """Return valid file extensions for Nortek ASCII data files."""
        return ('.dat', '.txt', '.asc')

    @classmethod
    def _is_extension_validation_strict(cls) -> bool:
        """ASCII formats can have various extensions, so warn only."""
        return False

    def _read_header(self, hdr_file_path):
        """Reads the .hdr file to extract column names and units."""
        headers = []
        with open(hdr_file_path, 'r') as file:
            capture = False
            for line in file:
                if line.strip() == "Data file format":
                    capture = True
                    continue
                if capture:
                    if line.strip() == '':
                        break
                    if line.strip() and not line.startswith('---') and not line.startswith('['):
                        # Use regex to split the line considering whitespace count
                        parts = re.split(r'\s{2,}', line.strip())

                        if len(parts) >= 2:
                            col_number = parts[0]
                            if parts[-1].startswith('(') and parts[-1].endswith(')'):
                                unit = parts[-1].strip('()')
                                col_name = ' '.join(parts[1:-1])
                            else:
                                unit = 'unknown'
                                col_name = ' '.join(parts[1:])
                        else:
                            # Fallback if no unit is provided and the line is not correctly parsed
                            col_number = parts[0].split()[0]
                            col_name = ' '.join(parts[0].split()[1:])
                            unit = 'unknown'

                        headers.append((col_number, col_name, unit))
        return headers

    def _parse_data(self, dat_file_path, headers):
        """Parses the .dat file using headers information."""
        columns = [name for _, name, _ in headers]  # Extract just the names from headers

        # Handle duplicate column names by making them unique
        unique_columns = []
        seen = {}
        for col in columns:
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            unique_columns.append(col)

        data = pd.read_csv(dat_file_path, sep=r'\s+', names=unique_columns)
        return data

    def _create_xarray_dataset(self, df, headers):
        """Converts the DataFrame to an xarray Dataset, renaming columns and assigning units."""
        # Convert columns to datetime
        df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])

        # Set datetime as the index
        df.set_index('time', inplace=True)

        # Rename columns as specified
        df.rename(columns=params.rename_list, inplace=True)

        # Convert the DataFrame to an xarray Dataset
        ds = xr.Dataset.from_dataframe(df)

        # Renaming and CF meta data enrichment
        for header in headers:
            _, variable, unit = header

            # Rename
            if variable in params.rename_list.keys():
                variable = params.rename_list[variable]

            # Set unit
            ds[variable].attrs['unit'] = unit

        # Assign meta information for all attributes of the xarray Dataset
        for key in (list(ds.data_vars.keys()) + list(ds.coords.keys())):
            super()._assign_metadata_for_key_to_xarray_dataset( ds, key)

        return ds

    def _load_data(self) -> xr.Dataset:
        """Load the Nortek ASCII data and return an xarray Dataset.
        
        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        headers = self._read_header(self.input_header_file)
        data = self._parse_data(self.input_file, headers)
        ds = self._create_xarray_dataset(data, headers)
        return ds

    @classmethod
    def format_key(cls) -> str:
        return 'nortek-ascii'

    @classmethod
    def format_name(cls) -> str:
        return 'Nortek ASCII'

    @classmethod
    def file_extension(cls) -> str | None:
        return None
    
    @classmethod
    def format_mappings(cls) -> dict:
        """Get Nortek ASCII format-specific variable name mappings.
        
        Returns:
            Dictionary mapping canonical parameter names to Nortek-specific
            variable name patterns commonly found in ASCII export files.
        """
        return {
            params.EAST_VELOCITY: ['Velocity (Beam1|X|East)', 'Eastward velocity'],
            params.NORTH_VELOCITY: ['Velocity (Beam2|Y|North)', 'Northward velocity'],
            params.UP_VELOCITY: ['Velocity (Beam3|Z|Up)', 'Upward velocity'],
            params.EAST_AMPLITUDE: ['Amplitude (Beam1)', 'Eastward amplitude'],
            params.NORTH_AMPLITUDE: ['Amplitude (Beam2)', 'Northward amplitude'],
            params.UP_AMPLITUDE: ['Amplitude (Beam3)', 'Upward amplitude'],
            params.SPEED_OF_SOUND: ['Soundspeed', 'Speed of Sound'],
            params.PRESSURE: ['Pressure'],
            params.TEMPERATURE: ['Temperature'],
        }
