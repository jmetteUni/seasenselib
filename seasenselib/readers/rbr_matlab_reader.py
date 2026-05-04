"""
Facade for reading RBR MATLAB .mat files, automatically selecting the correct reader
based on the root variable in the MATLAB structure.

If the root variable is "RBR", delegates to RbrMatlabLegacyReader.
If the root variable is "rsk", delegates to RbrMatlabRsktoolsReader.
Otherwise, raises an error.
"""

from __future__ import annotations
import xarray as xr
from .base import AbstractReader
from .rbr_matlab_legacy_reader import RbrMatlabLegacyReader
from .rbr_matlab_rsktools_reader import RbrMatlabRsktoolsReader

class RbrMatlabReader(AbstractReader):
    """
    Facade for reading RBR Matlab .mat files, automatically selecting the correct reader
    based on the root variable in the MATLAB structure.
    
    Note
    ----
    File validation occurs twice: once in this facade and once in the delegate reader.
    This is intentional design for defense-in-depth and to ensure delegate readers
    work correctly when instantiated directly. The validation overhead is
    negligible compared to file loading time.
    """
    def __init__(self, input_file: str,
                 mapping: dict | None = None,
                 **kwargs):
        """Initialize RbrMatlabReader.
        
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
        # Avoid running the pipeline twice: delegate reader handles postprocessing.
        delegate_use_steps = kwargs.pop("use_steps", True)
        super().__init__(input_file, mapping, use_steps=False, **kwargs)
        self._reader_format_name = None
        self._reader_format_key = None
        self._validate_file()
        # Store kwargs to pass to delegate reader
        self._delegate_use_steps = delegate_use_steps
        self._kwargs = dict(kwargs)

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        """Return valid file extensions for MATLAB files."""
        return ('.mat',)

    def _load_data(self) -> xr.Dataset:
        """Select the appropriate reader and load data.
        
        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """

        import scipy.io

        # Load Matlab file to inspect root variable
        mat = scipy.io.loadmat(self.input_file, squeeze_me=True, struct_as_record=False)

        # Select the appropriate reader based on root variable
        # Pass through all kwargs to honor configuration options
        if "RBR" in mat:
            reader = RbrMatlabLegacyReader(
                self.input_file,
                mapping=self.mapping,
                use_steps=self._delegate_use_steps,
                **self._kwargs
            )
        elif "rsk" in mat:
            reader = RbrMatlabRsktoolsReader(
                self.input_file,
                mapping=self.mapping,
                use_steps=self._delegate_use_steps,
                **self._kwargs
            )
        else:
            raise ValueError("Neither 'RBR' nor 'rsk' struct found in .mat file.")

        # Store reader metadata
        self._reader_format_name = reader.format_name()
        self._reader_format_key = reader.format_key()
        
        return reader.data

    @classmethod
    def format_key(cls) -> str:
        return 'rbr-matlab'

    @classmethod
    def format_name(cls) -> str:
        return 'RBR Matlab'

    @classmethod
    def file_extension(cls) -> str | None:
        return None
