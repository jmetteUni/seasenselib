"""
Module for reading RBR .rsk files into xarray Datasets.
"""

from __future__ import annotations
import pandas as pd
import xarray as xr
import seasenselib.parameters as params
from .base import AbstractReader


class RbrRskReader(AbstractReader):
    """
    Reads sensor data from a RBR .rsk file into a xarray Dataset.

    Attributes
    ----------
    data : xr.Dataset
        The xarray Dataset containing the sensor data.
    input_file : str
        The path to the input file containing the RBR legacy data.
    mapping : dict, optional
        A dictionary mapping names used in the input file to standard names.
    """

    def __init__(self, input_file: str,
                 mapping: dict | None = None,
                 **kwargs):
        """Initialize RbrRskReader.

        Parameters
        ----------
        input_file : str
            The path to the input file containing the data.
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
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        """Return valid file extensions for RSK files."""
        return ('.rsk',)

    def _load_data(self) -> xr.Dataset:
        """ Reads a RSK file and converts it to a xarray Dataset. 

        This method uses the pyrsktools library to read the RSK file, extracts
        channel information and measurement data, processes the timestamps, and
        organizes the data into a xarray Dataset. It also assigns metadata 
        according to the CF conventions to the dataset variables.
        
        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """

        from pyrsktools import RSK

        # Open the RSK file and read the data
        rsk = RSK(self.input_file)
        rsk.open()
        rsk.readdata()
        rsk.close()

        # Convert array to xarray Dataset
        ds = xr.Dataset(
            data_vars={name: (['time'], rsk.data[name]) for name in rsk.channelNames},
            coords={params.TIME: pd.to_datetime(rsk.data['timestamp'], unit='s')}
        )

        # Assign metadata to the dataset variables
        # For this, iterate over rsk.channels and look for channel name = longName.
        # Then assign _dbName, shortName, channelID, feModuleType, feModuleVersion,
        # units, label, shortName as attributes to the dataset variables.
        for channel in rsk.channels:
            if channel.longName in ds:
                attrs = {
                    'rsk_channel_id': channel.channelID,
                    'rsk_long_name': channel.longName,
                    'rsk_short_name': channel.shortName,
                    'rsk_label': channel.label,
                    'rsk_dbName': channel._dbName,
                    'rsk_units': channel.units,
                    'rsk_units_plain_text': channel.unitsPlainText,
                    'rsk_fe_module_type': channel.feModuleType,
                    'rsk_fe_module_version': channel.feModuleVersion,
                    'rsk_is_measured': channel.isMeasured,
                    'rsk_is_derived': channel.isDerived,
                }
                ds[channel.longName].attrs.update(attrs)

        # Store instrument info for metadata extraction
        self._instrument_info = rsk.instrument
        self._db_info = rsk.dbInfo

        # Add instrument information as global attributes
        if self._instrument_info:
            attrs = {
                'instrument_model': self._instrument_info.model,
                'instrument_serial': self._instrument_info.serialID,
                'instrument_firmware_version': self._instrument_info.firmwareVersion,
                'instrument_firmware_type': self._instrument_info.firmwareType,
            }
            ds.attrs.update(attrs)
            if getattr(self._instrument_info, "partNumber", None):
                ds.attrs['instrument_part_number'] = self._instrument_info.partNumber

        # Add database information as global attributes
        if self._db_info:
            ds.attrs['rsk_version'] = self._db_info.version
            ds.attrs['rsk_type'] = self._db_info.type

        # Perform default post-processing
        return ds

    @classmethod
    def format_key(cls) -> str:
        return 'rbr-rsk-default'

    @classmethod
    def format_name(cls) -> str:
        return 'RBR RSK Default'

    @classmethod
    def file_extension(cls) -> str | None:
        return None
    
    @classmethod
    def format_mappings(cls) -> dict:
        """Get RBR RSK format-specific variable name mappings.
        
        Returns:
            Dictionary mapping canonical parameter names to RBR-specific
            variable name patterns commonly found in RSK files.
        """
        return {
            params.TEMPERATURE: ['temperature_00', 'temperature_01', 'temperature'],
            params.CONDUCTIVITY: ['conductivity_00', 'conductivity_01', 'conductivity'],
            params.PRESSURE: ['sea_pressure', 'pressure'],
            params.SALINITY: ['salinity_00', 'salinity_01', 'salinity'],
            params.DEPTH: ['depth'],
        }
