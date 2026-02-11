"""
Module for reading RCM data from MATLAB .mat files.
"""

from __future__ import annotations
import pandas as pd
import xarray as xr
from seasenselib.readers.base import AbstractReader
import seasenselib.parameters as params

class RcmMatlabReader(AbstractReader):
    """Reader which converts RCM data stored in MATLAB .mat files into xarray dataset."""

    def __init__(self, input_file: str,
                 mapping: dict | None = None,
                 **kwargs):
        """Initialize RcmMatlabReader.
        
        Parameters
        ----------
        input_file : str
            Path to the MAT file.
        mapping : dict, optional
            Variable name mapping dictionary.
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
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...] | None:
        """Return valid file extensions for MATLAB files."""
        return ('.mat',)

    def _parse_data(self, mat_file_path):
        import scipy.io

        # read adcp file 
        data = scipy.io.loadmat(mat_file_path)

        #prepare for alteration
        def mat_to_dict(data):
            return {key: data[key].flatten()
                    if hasattr(data[key], 'flatten')
                    else data[key]
                    for key in data.keys()
            }
        data = mat_to_dict(data)

        # convert julian time to datetime
        data['time'] = pd.to_datetime(data['t'] - 719529, unit='D')
        
        # remove original julian time 
        data.pop('t')

        #create pandas dataframe 
        df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))

        # set time as index
        df.set_index('time', inplace=True)

        return df

    def _create_xarray_dataset(self, df):
        """Create xarray dataset from pandas dataframe.
        
        Returns raw dataset with format-specific variable names.
        Variable mapping and metadata enrichment handled by stages.
        """
        ds = xr.Dataset.from_dataframe(df)
        
        # Add minimal units from raw data (stages will enrich with CF metadata)
        ds['u'].attrs = {'units': 'm/s'}
        ds['v'].attrs = {'units': 'm/s'}
        ds['temp'].attrs = {'units': '°C'}
        ds['cond'].attrs = {'units': 'S/m'}
        ds['pres'].attrs = {'units': 'dbar'}
        
        # Add source information
        ds.attrs['source'] = 'Recording Current Meter - Aanderaa'
        ds.attrs['instrument'] = 'RCM'
        
        return ds

    def _load_data(self) -> xr.Dataset:
        """Load data from the MATLAB file and return an xarray Dataset."""
        data = self._parse_data(self.input_file)
        return self._create_xarray_dataset(data)

    @classmethod
    def format_mappings(cls) -> dict[str, list]:
        """Return RCM-specific variable name mappings.
        
        Returns
        -------
        dict[str, list]
            Dictionary mapping standard names to RCM format-specific aliases.
        """
        return {
            params.EAST_VELOCITY: ['u'],
            params.NORTH_VELOCITY: ['v'],
            params.TEMPERATURE: ['temp'],
            params.CONDUCTIVITY: ['cond'],
            params.PRESSURE: ['pres'],
            'vdir': ['vdir'],
            'vmag': ['vmag'],
        }

    @classmethod
    def format_key(cls) -> str:
        return 'rcm-matlab'

    @classmethod
    def format_name(cls) -> str:
        return 'RCM Matlab'

    @classmethod
    def file_extension(cls) -> str | None:
        return None
