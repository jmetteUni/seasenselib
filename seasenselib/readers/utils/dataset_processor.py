"""
Dataset processing utilities for xarray Datasets.

**DEPRECATED**: This module is deprecated and will be removed in v0.6.0.
Use the stage system instead: from seasenselib.pipeline import default_pipeline

This module provides static methods for transforming xarray Datasets,
including sorting variables, renaming parameters, deriving oceanographic
parameters, and assigning global attributes.
"""

from __future__ import annotations
import platform
import re
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Optional, Callable

import xarray as xr

import seasenselib.parameters as params

MODULE_NAME = 'seasenselib'


class DatasetProcessor:
    """
    Utility class for xarray Dataset transformations.
    
    **DEPRECATED**: This class is deprecated and will be removed in v0.6.0.
    Use the stage system instead:
    
    .. code-block:: python
    
        from seasenselib.pipeline import default_pipeline
        
        # Instead of DatasetProcessor methods:
        pipeline = default_pipeline()
        processed_ds = pipeline.process(raw_ds)
    
    All methods are static and operate on xarray Datasets directly.
    These methods handle:
    - Variable sorting
    - Parameter renaming according to standard mappings
    - Deriving oceanographic parameters (density, potential temperature)
    - Assigning global CF-compliant attributes
    
    Examples
    --------
    >>> from seasenselib.readers.utils import DatasetProcessor
    >>> import xarray as xr
    >>> 
    >>> ds = xr.Dataset(...)
    >>> ds = DatasetProcessor.sort_variables(ds)
    >>> ds = DatasetProcessor.rename_parameters(ds)
    """
    
    def __init__(self):
        """
        Initialize DatasetProcessor.
        
        **DEPRECATED**: This class is deprecated and will be removed in v0.6.0.
        Use the stage system instead: from seasenselib.pipeline import default_pipeline
        """
        warnings.warn(
            "DatasetProcessor is deprecated and will be removed in v0.6.0. "
            "Use the stage system instead: from seasenselib.pipeline import default_pipeline",
            DeprecationWarning,
            stacklevel=2
        )
    
    @staticmethod
    def sort_variables(ds: xr.Dataset) -> xr.Dataset:
        """
        Sort variables in an xarray Dataset alphabetically by name.
        
        Variables with the same base name (e.g., temperature_1, temperature_2)
        are grouped together due to alphabetical sorting.
        
        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to be sorted.
            
        Returns
        -------
        xr.Dataset
            The xarray Dataset with variables sorted by their names.
            
        Examples
        --------
        >>> ds = xr.Dataset({'z': [1], 'a': [2], 'b': [3]})
        >>> sorted_ds = DatasetProcessor.sort_variables(ds)
        >>> list(sorted_ds.data_vars)
        ['a', 'b', 'z']
        """
        # Sort all variables and coordinates by name
        all_names = sorted(list(ds.data_vars) + list(ds.coords))

        # Create a new Dataset with sorted variables and coordinates
        ds_sorted = ds[all_names]

        # Ensure that the attributes are preserved
        ds_sorted.attrs = ds.attrs.copy()

        return ds_sorted

    @staticmethod
    def rename_parameters(ds: xr.Dataset) -> xr.Dataset:
        """
        Rename variables in an xarray Dataset according to standard mappings.
        
        **DEPRECATED**: Use MappingStage from the stage system instead.
        
        Handles aliases with or without trailing numbering and ensures unique 
        standard names with numbering. If a standard name only occurs once, 
        it will not have a numbering suffix.
        
        Uses ``seasenselib.parameters.default_mappings`` for the alias-to-standard
        name mapping.
        
        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset with variables to rename.
            
        Returns
        -------
        xr.Dataset
            The xarray Dataset with renamed variables.
            
        Examples
        --------
        >>> # If default_mappings has {'temperature': ['temp', 't']}
        >>> ds = xr.Dataset({'temp': [20.0], 'sal': [35.0]})
        >>> renamed = DatasetProcessor.rename_parameters(ds)
        >>> 'temperature' in renamed.data_vars
        True
        """
        warnings.warn(
            "DatasetProcessor.rename_parameters() is deprecated. "
            "Use MappingStage from the stage system instead.",
            DeprecationWarning,
            stacklevel=2
        )
        ds_vars = list(ds.variables)
        rename_dict = {}

        # Build a reverse mapping: alias_lower -> standard_name
        alias_to_standard = {}
        for standard_name, aliases in params.default_mappings.items():
            for alias in aliases:
                alias_to_standard[alias.lower()] = standard_name

        # First, collect all matches: (standard_name, original_var, suffix)
        matches = []
        for var in ds_vars:
            if not isinstance(var, str):
                continue
            var_lower = var.lower()
            matched = False
            for alias_lower, standard_name in alias_to_standard.items():
                # Match alias with optional _<number> at the end
                m = re.match(rf"^{re.escape(alias_lower)}(_?\d{{1,2}})?$", var_lower)
                if m:
                    suffix = m.group(1) or ""
                    matches.append((standard_name, var, suffix))
                    matched = True
                    break
            if not matched:
                continue

        # Group by standard_name
        grouped = defaultdict(list)
        for standard_name, var, suffix in matches:
            grouped[standard_name].append((var, suffix))

        # Assign new names: only add numbering if there are multiple
        for standard_name, vars_with_suffixes in grouped.items():
            if len(vars_with_suffixes) == 1:
                # Only one variable: use plain standard name
                rename_dict[vars_with_suffixes[0][0]] = standard_name
            else:
                # Multiple variables: always add numbering (_1, _2, ...)
                for idx, (var, suffix) in enumerate(vars_with_suffixes, 1):
                    rename_dict[var] = f"{standard_name}_{idx}"

        return ds.rename(rename_dict)

    @staticmethod
    def derive_oceanographic_parameters(
        ds: xr.Dataset,
        assign_metadata_callback: Optional[Callable[[xr.Dataset, str], None]] = None
    ) -> xr.Dataset:
        """
        Derive oceanographic parameters from temperature, pressure, and salinity.
        
        **DEPRECATED**: Use DerivationStage from the stage system instead.
        
        Calculates derived parameters like density and potential temperature
        using the Gibbs SeaWater (GSW) oceanographic toolbox when temperature, 
        pressure, and salinity data are available.
        
        For multiple sensors (e.g., temperature_1, temperature_2), uses the first
        available sensor or the base parameter name if only one exists.
        
        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing the sensor data.
        assign_metadata_callback : callable, optional
            A callback function to assign metadata to derived parameters.
            Should accept (ds, key) arguments.
            
        Returns
        -------
        xr.Dataset
            The xarray Dataset with derived parameters added.
            
        Notes
        -----
        Requires temperature, salinity, and pressure to be present in the dataset.
        If any are missing, returns the dataset unchanged.
        """
        warnings.warn(
            "DatasetProcessor.derive_oceanographic_parameters() is deprecated. "
            "Use DerivationStage from the stage system instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Find the appropriate temperature variable
        temperature_var = None
        if params.TEMPERATURE in ds.data_vars:
            temperature_var = params.TEMPERATURE
        elif f"{params.TEMPERATURE}_1" in ds.data_vars:
            temperature_var = f"{params.TEMPERATURE}_1"
        
        # Find the appropriate salinity variable
        salinity_var = None
        if params.SALINITY in ds.data_vars:
            salinity_var = params.SALINITY
        elif f"{params.SALINITY}_1" in ds.data_vars:
            salinity_var = f"{params.SALINITY}_1"
        
        # Pressure should typically be singular, but check both possibilities
        pressure_var = None
        if params.PRESSURE in ds.data_vars:
            pressure_var = params.PRESSURE
        elif f"{params.PRESSURE}_1" in ds.data_vars:
            pressure_var = f"{params.PRESSURE}_1"
        
        # Check if we have all required parameters for oceanographic calculations
        if temperature_var and salinity_var and pressure_var:
            import gsw
            
            # Derive density using GSW
            ds[params.DENSITY] = ([params.TIME], gsw.density.rho(
                ds[salinity_var].values, 
                ds[temperature_var].values, 
                ds[pressure_var].values))
            
            # Derive potential temperature using GSW
            ds[params.POTENTIAL_TEMPERATURE] = ([params.TIME], gsw.pt0_from_t(
                ds[salinity_var].values, 
                ds[temperature_var].values, 
                ds[pressure_var].values))
            
            # Assign metadata for derived parameters if callback provided
            if assign_metadata_callback:
                assign_metadata_callback(ds, params.DENSITY)
                assign_metadata_callback(ds, params.POTENTIAL_TEMPERATURE)
                
        return ds

    @staticmethod
    def assign_default_global_attributes(
        ds: xr.Dataset,
        input_file: str,
        format_name: str,
        reader_class_name: str
    ) -> xr.Dataset:
        """
        Assign default global attributes to the xarray Dataset.
        
        Sets CF-compliant global attributes including history, conventions,
        and processor information.
        
        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to which the global attributes will be assigned.
        input_file : str
            Path to the input file.
        format_name : str
            Human-readable format name (e.g., "Sea-Bird CNV").
        reader_class_name : str
            Name of the reader class used.
            
        Returns
        -------
        xr.Dataset
            The xarray Dataset with global attributes assigned.
        """
        module_name = MODULE_NAME
        module_version = version(MODULE_NAME)
        python_version = platform.python_version()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Assemble history entry
        history_entry = (
            f"{timestamp}: created from {format_name} file ({input_file}) "
            f"using {module_name} v{module_version} ({reader_class_name} class) "
            f"under Python {python_version}"
        )

        ds.attrs['history'] = history_entry
        ds.attrs['Conventions'] = 'CF-1.13'

        # Information about the processor of the xarray dataset
        ds.attrs['processor_name'] = module_name
        ds.attrs['processor_version'] = module_version
        ds.attrs['processor_reader_class'] = reader_class_name
        ds.attrs['processor_python_version'] = python_version
        ds.attrs['processor_input_filename'] = input_file
        ds.attrs['processor_input_file_type'] = format_name

        return ds
