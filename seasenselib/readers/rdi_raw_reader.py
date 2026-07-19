"""Reader wrapper for RDI raw binary ADCP files via MHKiT DOLfYN."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

import xarray as xr

import seasenselib.parameters as params
from .base import AbstractReader


_SUPPORTED_EXTENSIONS = (".000", ".pd0", ".enr", ".ens", ".enx")

_DOLFYN_ATTR_DEFAULTS: dict[str, dict[str, str]] = {
    "c_sound": {
        "units": "m s-1",
        "long_name": "Speed of Sound",
        "standard_name": "speed_of_sound_in_sea_water",
    },
    "depth": {
        "units": "m",
        "long_name": "Depth",
        "standard_name": "depth",
        "positive": "down",
    },
    "pressure": {
        "units": "dbar",
        "long_name": "Pressure",
        "standard_name": "sea_water_pressure",
    },
    "salinity": {
        "units": "psu",
        "long_name": "Salinity",
        "standard_name": "sea_water_salinity",
    },
    "temp": {
        "units": "degree_C",
        "long_name": "Temperature",
        "standard_name": "sea_water_temperature",
    },
    "vel": {
        "units": "m s-1",
        "long_name": "Water Velocity",
    },
}


def _json_text(value: Any) -> str:
    """Return compact JSON text for attrs that should survive netCDF writing."""
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _read_rdi_import() -> tuple[Callable[..., xr.Dataset], str]:
    """Import the documented MHKiT DOLfYN RDI reader lazily."""
    try:
        from mhkit import dolfyn
    except ModuleNotFoundError as exc:
        if exc.name != "mhkit":
            raise
        try:
            import dolfyn  # type: ignore[no-redef]
        except ModuleNotFoundError as fallback_exc:
            raise ImportError(
                "RdiRawReader requires MHKiT with DOLfYN support. "
                'Install it with: pip install "mhkit[dolfyn]"'
            ) from fallback_exc
        source = "dolfyn.io.rdi.read_rdi"
    else:
        source = "mhkit.dolfyn.io.rdi.read_rdi"

    try:
        read_rdi = dolfyn.io.rdi.read_rdi
    except AttributeError as exc:
        raise ImportError(
            "The installed DOLfYN package does not expose io.rdi.read_rdi(). "
            'Please install a recent MHKiT build with: pip install "mhkit[dolfyn]"'
        ) from exc

    return read_rdi, source


class RdiRawReader(AbstractReader):
    """Read Teledyne RD Instruments (RDI) raw binary ADCP files with DOLfYN.

    The reader delegates binary decoding to the MHKiT DOLfYN RDI parser and
    keeps the returned xarray structure intact. SeaSenseLib only adds reader
    provenance, raw-metadata hints, and high-confidence variable mappings such
    as ``temp`` -> ``temperature`` and ``c_sound`` -> ``speed_of_sound``.

    Velocity is intentionally preserved as DOLfYN's vector variable ``vel``.
    Its component meaning depends on ``ds.attrs["coord_sys"]`` (for example
    beam, inst, ship, earth, or principal), so automatic CF component variables
    would be a scientific decision rather than a safe metadata cleanup.
    """

    def __init__(
        self,
        input_file: str,
        userdata: bool | str | None = None,
        nens: int | tuple[int, int] | None = None,
        debug: int | None = None,
        vmdas_search: bool = False,
        winriver: bool = False,
        search_num: int | None = None,
        mapping: dict | None = None,
        **kwargs,
    ):
        """Initialize the RDI raw ADCP reader.

        Parameters
        ----------
        input_file
            Path to an RDI raw binary ADCP file, commonly ``.000``.
        userdata
            Passed to DOLfYN. Use ``True`` to search for a sibling
            ``*.userdata.json`` file, ``False`` to skip it, or a string path
            to use a specific metadata file. ``None`` keeps DOLfYN's default.
        nens
            Number of ensembles to read, or a start/stop tuple for DOLfYN
            versions that support sliced reads.
        debug
            DOLfYN debug level. ``None`` keeps DOLfYN's default.
        vmdas_search
            Ask DOLfYN to search for VMDAS navigation blocks when offsets are
            unreliable.
        winriver
            Force DOLfYN's WinRiver parsing path. DOLfYN normally detects this.
        search_num
            Optional DOLfYN search window for versions that expose it.
        mapping
            Additional SeaSenseLib variable mappings.
        **kwargs
            Base reader options such as ``perform_default_postprocessing`` and
            ``use_steps``.
        """
        self._userdata = userdata
        self._nens = nens
        self._debug = debug
        self._vmdas_search = vmdas_search
        self._winriver = winriver
        self._search_num = search_num
        self._dolfyn_reader_source = ""
        self._raw_metadata_blocks: dict[str, Any] = {}
        self._raw_metadata_variables: dict[str, Any] = {}
        super().__init__(input_file, mapping, **kwargs)
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        """Return supported RDI raw binary extensions."""
        return _SUPPORTED_EXTENSIONS

    def _load_data(self) -> xr.Dataset:
        """Load the RDI raw data with DOLfYN and return an xarray Dataset."""
        read_rdi, source = _read_rdi_import()
        self._dolfyn_reader_source = source
        ds = self._call_read_rdi(read_rdi)
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"DOLfYN returned {type(ds)!r}; expected xarray.Dataset."
            )

        self._annotate_dolfyn_dataset(ds)
        self._raw_metadata_blocks = self._build_raw_metadata_blocks(ds)
        self._raw_metadata_variables = self._build_raw_variable_metadata(ds)
        return ds

    def _call_read_rdi(self, read_rdi: Callable[..., xr.Dataset]) -> xr.Dataset:
        """Call DOLfYN's RDI reader while tolerating small API changes."""
        kwargs: dict[str, Any] = {
            "userdata": self._userdata,
            "nens": self._nens,
            "vmdas_search": self._vmdas_search,
            "winriver": self._winriver,
        }
        if self._search_num is not None:
            kwargs["search_num"] = self._search_num

        signature = inspect.signature(read_rdi)
        parameters = signature.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        )

        if self._debug is not None:
            if "debug" in parameters:
                kwargs["debug"] = self._debug
            elif "debug_level" in parameters:
                kwargs["debug_level"] = self._debug
            elif accepts_kwargs:
                kwargs["debug"] = self._debug

        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}

        return read_rdi(self.input_file, **kwargs)

    def _annotate_dolfyn_dataset(self, ds: xr.Dataset) -> None:
        """Add SeaSenseLib provenance and conservative metadata defaults."""
        ds.attrs.setdefault("source_format", self.format_key())
        ds.attrs.setdefault("source_format_name", self.format_name())
        ds.attrs.setdefault("raw_data_reader", self._dolfyn_reader_source)
        ds.attrs.setdefault(
            "rdi_raw_reader_note",
            (
                "Binary decoding is delegated to DOLfYN. Velocity is preserved "
                "as DOLfYN variable 'vel'; component names depend on coord_sys."
            ),
        )
        ds.attrs.setdefault(
            "rdi_mapping_hints",
            _json_text(self._mapping_hints(ds)),
        )
        ds.attrs.setdefault(
            "rdi_reader_options",
            _json_text(self._read_options()),
        )

        for variable_name, attrs in _DOLFYN_ATTR_DEFAULTS.items():
            if variable_name not in ds:
                continue
            for attr_name, attr_value in attrs.items():
                ds[variable_name].attrs.setdefault(attr_name, attr_value)

        for variable_name in ds.data_vars:
            variable_attrs = ds[variable_name].attrs
            variable_attrs.setdefault("original_name", variable_name)
            if "units" in variable_attrs:
                variable_attrs.setdefault("original_units", variable_attrs["units"])

        if "range" in ds.coords:
            ds["range"].attrs.setdefault("units", "m")
            ds["range"].attrs.setdefault("long_name", "Profile Range")
            ds["range"].attrs.setdefault(
                "comment",
                "Distance from transducer to cell center, as decoded by DOLfYN.",
            )

    def _build_raw_metadata_blocks(self, ds: xr.Dataset) -> dict[str, Any]:
        """Build opaque raw metadata blocks for the finalization stage."""
        return {
            "configuration": {
                "dolfyn_reader": self._dolfyn_reader_source,
                "reader_options": self._read_options(),
            },
            "mapping_hints": self._mapping_hints(ds),
        }

    def _build_raw_variable_metadata(self, ds: xr.Dataset) -> dict[str, Any]:
        """Summarize DOLfYN variables without copying data values."""
        variables: dict[str, Any] = {}
        for name in list(ds.coords) + list(ds.data_vars):
            array = ds[name]
            attrs = dict(array.attrs)
            variable_metadata = {
                "dims": list(array.dims),
                "attrs": attrs,
            }
            if "original_name" in attrs:
                variable_metadata["original_name"] = attrs["original_name"]
            if "units" in attrs:
                variable_metadata["units"] = attrs["units"]
            if "original_units" in attrs:
                variable_metadata["original_units"] = attrs["original_units"]
            variables[name] = variable_metadata
        return variables

    def _postprocess_after_pipeline(self, ds: xr.Dataset) -> xr.Dataset:
        """Expose raw-variable metadata under mapped variable names as well."""
        if "raw_metadata" not in ds.attrs or not self._processing_metadata:
            return ds

        variable_mappings = self._processing_metadata.get("variable_mappings")
        if not isinstance(variable_mappings, dict) or not variable_mappings:
            return ds

        try:
            raw_metadata = json.loads(ds.attrs["raw_metadata"])
        except (TypeError, json.JSONDecodeError):
            return ds

        variables = raw_metadata.get("variables")
        if not isinstance(variables, dict):
            return ds

        for original_name, mapped_name in variable_mappings.items():
            if original_name not in variables or mapped_name in variables:
                continue
            source_metadata = variables[original_name]
            if not isinstance(source_metadata, dict):
                continue
            mapped_metadata = dict(source_metadata)
            mapped_metadata.setdefault("original_name", original_name)
            if mapped_name in ds:
                attrs = ds[mapped_name].attrs
                if "units" in attrs:
                    mapped_metadata["units"] = attrs["units"]
                if "original_units" in attrs:
                    mapped_metadata.setdefault("original_units", attrs["original_units"])
            variables[mapped_name] = mapped_metadata

        ds.attrs["raw_metadata"] = json.dumps(
            raw_metadata,
            ensure_ascii=False,
            default=str,
        )
        return ds

    def _read_options(self) -> dict[str, Any]:
        """Return DOLfYN read options for provenance."""
        return {
            "userdata": self._userdata,
            "nens": self._nens,
            "debug": self._debug,
            "vmdas_search": self._vmdas_search,
            "winriver": self._winriver,
            "search_num": self._search_num,
        }

    def _mapping_hints(self, ds: xr.Dataset) -> dict[str, Any]:
        """Describe safe and conditional mappings for downstream users."""
        coord_sys = str(ds.attrs.get("coord_sys", "")).lower() or None
        velocity_confidence = "conditional" if coord_sys == "earth" else "uncertain"
        return {
            "safe_reader_mappings": {
                "temp": params.TEMPERATURE,
                "c_sound": params.SPEED_OF_SOUND,
                "pressure": params.PRESSURE,
                "depth": params.DEPTH,
                "salinity": params.SALINITY,
            },
            "velocity": {
                "source_variable": "vel",
                "coordinate_system": coord_sys,
                "confidence": velocity_confidence,
                "note": (
                    "DOLfYN stores velocity as a vector. When coord_sys is "
                    "'earth', components are east/north/up plus error velocity. "
                    "For beam, inst, ship, or principal coordinates, CF "
                    "east/north/up splitting should wait until rotation or "
                    "deployment metadata has been reviewed."
                ),
            },
            "quality_variables": {
                "amp": "Acoustic amplitude/intensity; DOLfYN standard_name is kept.",
                "corr": "Beam correlation; DOLfYN standard_name is kept.",
                "prcnt_gd": "Percent-good quality variable; DOLfYN standard_name is kept.",
            },
        }

    @classmethod
    def format_mappings(cls) -> dict[str, list[str]]:
        """Return conservative DOLfYN-to-SeaSenseLib variable mappings."""
        return {
            params.TEMPERATURE: ["temp"],
            params.SPEED_OF_SOUND: ["c_sound"],
            params.PRESSURE: ["pressure"],
            params.DEPTH: ["depth"],
            params.SALINITY: ["salinity"],
        }

    @classmethod
    def format_key(cls) -> str:
        return "rdi-raw"

    @classmethod
    def format_name(cls) -> str:
        return "RDI ADCP raw"

    @classmethod
    def file_extension(cls) -> str | None:
        return ".000"

    @classmethod
    def file_extensions(cls) -> tuple[str, ...]:
        return _SUPPORTED_EXTENSIONS


__all__ = ["RdiRawReader"]
