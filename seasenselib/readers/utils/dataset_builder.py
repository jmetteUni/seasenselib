"""
Dataset building utilities for xarray Datasets.

This module provides static methods for creating and populating xarray Datasets
with proper structure, coordinates, and data arrays.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import xarray as xr

import seasenselib.parameters as params


class DatasetBuilder:
    """
    Utility class for building xarray Datasets.
    
    All methods are static and handle the creation and population of
    xarray Datasets with proper coordinate systems and data structures.
    
    Examples
    --------
    >>> from seasenselib.readers.utils import DatasetBuilder
    >>> import numpy as np
    >>> 
    >>> time = np.array([...])
    >>> depth = np.array([...])
    >>> ds = DatasetBuilder.create_template(time, depth, 54.0, 10.0)
    >>> DatasetBuilder.assign_data(ds, 'temperature', temp_data)
    """
    
    @staticmethod
    def create_template(
        time_array: np.ndarray,
        depth_array: Optional[np.ndarray],
        latitude: float,
        longitude: float,
        depth_name: str = None
    ) -> xr.Dataset:
        """
        Create an xarray Dataset template with coordinates.
        
        Creates a Dataset with time, latitude, longitude coordinates and
        optionally depth as a data variable along the time dimension.
        
        Parameters
        ----------
        time_array : np.ndarray
            Array of datetime values for the time coordinate.
        depth_array : np.ndarray or None
            Array of depth values, or None if depth is not available.
        latitude : float
            Latitude coordinate value.
        longitude : float
            Longitude coordinate value.
        depth_name : str, optional
            Name for the depth variable. Defaults to params.DEPTH.
            
        Returns
        -------
        xr.Dataset
            Empty xarray Dataset with coordinates set up.
            
        Examples
        --------
        >>> import numpy as np
        >>> from datetime import datetime
        >>> time = np.array([datetime(2024, 1, 1), datetime(2024, 1, 2)])
        >>> depth = np.array([10.0, 15.0])
        >>> ds = DatasetBuilder.create_template(time, depth, 54.0, 10.0)
        >>> 'time' in ds.coords
        True
        """
        if depth_name is None:
            depth_name = params.DEPTH
            
        coords = dict(
            time=time_array,
            latitude=latitude,
            longitude=longitude,
        )

        # Only add depth coordinate if depth_array is not None
        if depth_array is not None:
            coords[depth_name] = ([params.TIME], depth_array)

        now_local = datetime.now().astimezone()
        now_utc = now_local.astimezone(timezone.utc)

        return xr.Dataset(
            data_vars=dict(),
            coords=coords,
            attrs=dict(
                latitude=latitude,
                longitude=longitude,
                CreateTime=now_local.isoformat(),
                CreateTime_UTC=now_utc.isoformat().replace("+00:00", "Z"),
                DataType='TimeSeries',
            )
        )

    @staticmethod
    def assign_data(ds: xr.Dataset, key: str, data: Any) -> None:
        """
        Assign a data array to the dataset.
        
        Creates a new DataArray with the time dimension and assigns it
        to the dataset under the given key.
        
        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to add data to.
        key : str
            The variable name for the data.
        data : array-like
            The data values to assign.
            
        Examples
        --------
        >>> ds = DatasetBuilder.create_template(time, depth, 54.0, 10.0)
        >>> DatasetBuilder.assign_data(ds, 'temperature', [20.1, 20.2, 20.3])
        >>> 'temperature' in ds.data_vars
        True
        """
        ds[key] = xr.DataArray(data, dims=params.TIME)
        ds[key].attrs = {}
