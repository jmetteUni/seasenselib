"""
Module for abstract base class for writing sensor data from xarray Datasets.

This module defines the `AbstractWriter` class, which serves as a base class for
all writer implementations in the SeaSenseLib package. Concrete writer classes should
inherit from this class and implement the `write` method to handle the specifics of
writing data to various formats (e.g., NetCDF, CSV, Excel).
"""

from abc import ABC, abstractmethod
import logging
import warnings
import xarray as xr

logger = logging.getLogger(__name__)


class AbstractWriter(ABC):
    """Abstract base class for writing sensor data from xarray Datasets.
    
    This class provides a common interface for all writer implementations.
    All concrete writer classes should inherit from this class and implement
    the write method.
    
    This class supports the context manager protocol for automatic resource cleanup:
    
    >>> with SomeWriter(dataset) as writer:
    ...     writer.write('output.nc')
    >>> # resources automatically cleaned up
    
    Attributes:
    -----------
    data : xr.Dataset (read-only)
        The xarray Dataset containing the sensor data to be written.
    
    Methods:
    --------
    __init__(data: xr.Dataset):
        Initializes the writer with the provided xarray Dataset.
    __enter__() -> AbstractWriter:
        Context manager entry point.
    __exit__(exc_type, exc_val, exc_tb) -> None:
        Context manager exit - releases resources.
    file_extension: str
        The default file extension for this writer (to be implemented by subclasses).
    format_name() -> str:
        Get the format name for this writer (to be implemented by subclasses).
    format_key() -> str:
        Get the format key for this writer (to be implemented by subclasses).
    write(file_name: str, **kwargs):
        Writes the xarray Dataset to a file (to be implemented by subclasses).

    Raises:
    -------
    NotImplementedError:
        If the subclass does not implement the `write` method or the `file_extension` property.
    TypeError:
        If the provided data is not an xarray Dataset.
    """

    def __init__(self, data: xr.Dataset):
        """Initialize the writer with the provided xarray Dataset.
        
        Parameters:
        -----------
        data : xr.Dataset
            The xarray Dataset containing the sensor data to be written.
            
        Raises:
        -------
        TypeError:
            If the provided data is not an xarray Dataset.
        """

        if not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset.")

        self._data = data
        self._initialized = True  # Flag to control setter behavior
        logger.info("Initialized writer %s", self.__class__.__name__)

    @staticmethod
    @abstractmethod
    def format_name() -> str:
        """Get the format name for this writer.

        This property must be implemented by all subclasses.

        Returns:
        --------
        str
            The format (e.g., 'netCDF', 'CSV').

        Raises:
        -------
        NotImplementedError:
            If the subclass does not implement this property.
        """
        raise NotImplementedError("Writer classes must define a format name")

    @staticmethod
    @abstractmethod
    def format_key() -> str:
        """Get the format key for this writer.

        This property must be implemented by all subclasses.
        
        Returns:
        --------
        str
            The format key (e.g., 'netcdf', 'csv').

        Raises:
        -------
        NotImplementedError:
            If the subclass does not implement this property.
        """
        raise NotImplementedError("Writer classes must define a format key")

    @staticmethod
    @abstractmethod
    def file_extension() -> str:
        """Get the default file extension for this writer.
        
        This property must be implemented by all subclasses.
        The extension must be unique over all registered writers.
        If the writer does not specify a unique file extension, just return `None`.
        
        Returns:
        --------
        str
            The file extension (e.g., '.nc', '.csv').

        Raises:
        -------
        NotImplementedError:
            If the subclass does not implement this property.
        """
        raise NotImplementedError("Writer classes must define a file extension")

    @property
    def data(self) -> xr.Dataset:
        """Get the xarray Dataset (read-only).
        
        Returns:
        --------
        xr.Dataset
            The xarray Dataset containing the sensor data.
        """
        return self._data

    @data.setter
    def data(self, value: xr.Dataset) -> None:
        """Set the xarray Dataset with validation (deprecated).
        
        .. deprecated:: 1.5
            Setting data after construction is deprecated and will be
            removed in version 2.0. Create a new writer instance instead.
        
        Parameters:
        -----------
        value : xr.Dataset
            The xarray Dataset containing the sensor data.
            
        Raises:
        -------
        TypeError:
            If the provided data is not an xarray Dataset.
        """
        if not isinstance(value, xr.Dataset):
            raise TypeError("Data must be an xarray Dataset.")
        
        # Warn if trying to set after initialization
        if hasattr(self, '_initialized') and self._initialized:
            warnings.warn(
                "Setting data after construction is deprecated and will be "
                "removed in version 2.0. Create a new writer instance instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
        self._data = value

    def __enter__(self) -> 'AbstractWriter':
        """Context manager entry point.
        
        Returns:
        --------
        AbstractWriter
            Returns self for use in with statement.
            
        Examples:
        ---------
        >>> with SomeWriter(dataset) as writer:
        ...     writer.write('output.nc')
        """
        logger.debug("Entering writer context for %s", self.__class__.__name__)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - releases resources.
        
        Parameters:
        -----------
        exc_type : type
            Exception type if an exception was raised.
        exc_val : BaseException
            Exception value if an exception was raised.
        exc_tb : TracebackType
            Traceback if an exception was raised.
        """
        self._data = None
        logger.debug("Exiting writer context for %s", self.__class__.__name__)

    def __repr__(self) -> str:
        """String representation of the writer.
        
        Returns:
        --------
        str
            Human-readable string showing class name and format.
        """
        return f"{self.__class__.__name__}(format='{self.format_key()}')"

    @abstractmethod
    def write(self, file_name: str, **kwargs):
        """Write the xarray Dataset to a file.
        
        Parameters:
        -----------
        file_name : str
            The name of the output file where the data will be saved.
        **kwargs
            Additional keyword arguments specific to the writer implementation.
        """
        raise NotImplementedError("Subclasses must implement the write method")
