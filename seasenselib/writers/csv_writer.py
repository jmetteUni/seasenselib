"""
Module for writing sensor data to CSV files.
"""

import logging
from seasenselib.writers.base import AbstractWriter
import seasenselib.parameters as params

logger = logging.getLogger(__name__)

class CsvWriter(AbstractWriter):
    """ Writes sensor data from a xarray Dataset to a CSV file. 
    
    This class is used to save sensor data in a CSV format, which is a common format for
    tabular data. The provided data is expected to be in an xarray Dataset format.  

    Example usage:
        writer = CsvWriter(data)
        writer.write("output_file.csv")

    Attributes:
    ------------
    data : xr.Dataset
        The xarray Dataset containing the sensor data to be written to a CSV file.

    Methods:
    ------------
    __init__(data: xr.Dataset):
        Initializes the CsvWriter with the provided xarray Dataset.
    write(file_name: str, coordinate = params.TIME):
        Writes the xarray Dataset to a CSV file with the specified file name.
        The coordinate parameter is validated to exist in the dataset but does not
        affect which data is written; the full dataset is always exported.
    file_extension: str
        The default file extension for this writer, which is '.csv'.
    """

    def write(self, file_name: str, coordinate=params.TIME, **kwargs):
        """ Writes the xarray Dataset to a CSV file with the specified file name and coordinate.

        Parameters:
        -----------
        file_name (str):
            The name of the output CSV file where the data will be saved.
        coordinate (str):
            A coordinate or dimension that must be present in the dataset. Default is params.TIME.
            This parameter is validated but does not affect the output; the full dataset
            is always written to CSV.
        **kwargs:
            Additional keyword arguments (unused in this implementation).
        """

        if coordinate not in self.data.coords and coordinate not in self.data.dims:
            raise ValueError(f"Coordinate '{coordinate}' not found in the dataset.")

        # Convert the selected data to a pandas dataframe
        df = self.data.to_dataframe()

        # Write the dataframe to the CSV file
        logger.info("Writing CSV file to '%s'", file_name)
        df.to_csv(file_name, index=True)
        logger.info("Wrote CSV file to '%s'", file_name)

    @staticmethod
    def file_extension() -> str:
        """Get the default file extension for this writer.
        
        Returns:
        --------
        str
            The file extension for CSV files, which is '.csv'.
        """
        return '.csv'

    @staticmethod
    def format_key() -> str:
        """Get the format key for this writer.
        
        Returns:
        --------
        str
            The format key 'csv'.
        """
        return 'csv'

    @staticmethod
    def format_name() -> str:
        """Get the human-readable format name.
        
        Returns:
        --------
        str
            The format name 'CSV'.
        """
        return 'CSV'
