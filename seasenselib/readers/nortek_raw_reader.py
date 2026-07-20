"""Reader wrapper for Nortek raw binary files."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib
import inspect
import io
import json
import logging
from typing import Any, Callable, Iterator

import numpy as np
import xarray as xr

import seasenselib.parameters as params
from .base import AbstractReader


logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (".aqd", ".vec", ".wpr")
_NORTEK_SOUND_SPEED_SETTING = "nortek_sound_speed_setting"
_NORTEK_PRESSURE_PLACEHOLDER = "nortek_pressure_placeholder"
_NORTEK_CORRELATION_PLACEHOLDER = "nortek_correlation_placeholder"
_EXPERIMENTAL_NOTE = (
    "Experimental reader: Nortek raw support is available for early validation "
    "and may be refined as additional Nortek binary variants are tested."
)
_SENSOR_SOURCE_DESCRIPTIONS = {
    "sensor": "Value is interpreted as coming from an instrument sensor.",
    "configured": "Value is interpreted as an instrument or user setting.",
    "placeholder": "Field is present but is not interpreted as a valid value.",
    "unknown": "Source cannot be determined from available metadata.",
}
_NORTEK_SENSOR_BASIS_DESCRIPTIONS = {
    "nortek_user_specified_sound_speed": (
        "Nortek user configuration flag for sound speed."
    ),
    "nortek_pressure_sensor_header": (
        "Nortek header pressure-sensor availability field."
    ),
    "nortek_pressure_nonzero": (
        "Decoded Nortek pressure field contains non-zero values."
    ),
    "nortek_pressure_all_zero": (
        "Decoded Nortek pressure field contains only zero values."
    ),
    "nortek_compass_header": "Nortek header compass availability field.",
    "nortek_tilt_sensor_header": (
        "Nortek header tilt-sensor availability field."
    ),
    "nortek_correlation_all_zero": (
        "Decoded Nortek correlation field contains only zero values."
    ),
}

_AQUADOPP_TEMPLATE_FIELDS = (
    "time",
    "error",
    "batt",
    "c_sound",
    "heading",
    "pitch",
    "roll",
    "status",
    "temp",
)

_NORTEK_ATTR_DEFAULTS: dict[str, dict[str, str]] = {
    "amp": {
        "units": "1",
        "long_name": "Acoustic Signal Amplitude",
        "comment": "Beam acoustic amplitude decoded from the Nortek raw data.",
    },
    "batt": {
        "units": "V",
        "long_name": "Battery Voltage",
        "measurement_type": "Measured",
    },
    "c_sound": {
        "units": "m s-1",
        "long_name": "Speed of Sound",
        "standard_name": "speed_of_sound_in_sea_water",
    },
    _NORTEK_SOUND_SPEED_SETTING: {
        "units": "m s-1",
        "long_name": "Nortek Sound Speed Setting",
        "measurement_type": "Configured",
        "comment": (
            "Sound speed decoded from the Nortek data block. The user "
            "configuration indicates a user-specified sound-speed setting, "
            "so SeaSenseLib does not treat this as a measured water property."
        ),
    },
    "corr": {
        "units": "%",
        "long_name": "Acoustic Signal Correlation",
    },
    _NORTEK_CORRELATION_PLACEHOLDER: {
        "units": "%",
        "long_name": "Nortek Correlation Placeholder",
        "measurement_type": "Placeholder",
        "comment": (
            "The decoded correlation field contains only zeros. For classic "
            "Aquadopp single-point blocks this field can be created by the "
            "backend template even though the raw block does not contain "
            "correlation samples."
        ),
    },
    "error": {
        "units": "1",
        "long_name": "Nortek Error Code",
        "comment": "Instrument error code decoded from the Nortek raw data.",
    },
    "heading": {
        "units": "degree",
        "long_name": "Heading",
        "standard_name": "platform_heading_angle",
        "measurement_type": "Measured",
    },
    "pitch": {
        "units": "degree",
        "long_name": "Pitch",
        "standard_name": "platform_pitch_angle",
        "measurement_type": "Measured",
    },
    "pressure": {
        "units": "dbar",
        "long_name": "Pressure",
        "standard_name": "sea_water_pressure",
        "measurement_type": "Measured",
    },
    _NORTEK_PRESSURE_PLACEHOLDER: {
        "units": "dbar",
        "long_name": "Nortek Pressure Placeholder",
        "measurement_type": "Placeholder",
        "comment": (
            "A pressure field was decoded, but the raw values or instrument "
            "configuration do not support treating it as measured pressure."
        ),
    },
    "roll": {
        "units": "degree",
        "long_name": "Roll",
        "standard_name": "platform_roll_angle",
        "measurement_type": "Measured",
    },
    "status": {
        "units": "1",
        "long_name": "Nortek Status Code",
        "comment": "Instrument status code decoded from the Nortek raw data.",
    },
    "temp": {
        "units": "degree_C",
        "long_name": "Temperature",
        "standard_name": "sea_water_temperature",
        "measurement_type": "Measured",
        "comment": "Temperature decoded from the Nortek raw data block.",
    },
    "vel": {
        "units": "m s-1",
        "long_name": "Water Velocity",
    },
}


def _read_nortek_import() -> tuple[Callable[..., xr.Dataset], str]:
    """Import the MHKiT DOLfYN Nortek reader lazily."""
    try:
        from mhkit import dolfyn
    except ModuleNotFoundError as exc:
        if exc.name != "mhkit":
            raise
        try:
            import dolfyn  # type: ignore[no-redef]
        except ModuleNotFoundError as fallback_exc:
            raise ImportError(
                "NortekRawReader requires MHKiT with DOLfYN support. "
                'Install it with: pip install "mhkit[dolfyn]"'
            ) from fallback_exc
        source_prefix = "dolfyn"
    else:
        source_prefix = "mhkit.dolfyn"

    read_nortek = getattr(dolfyn.io, "read_nortek", None)
    source = f"{source_prefix}.io.read_nortek"
    if not callable(read_nortek):
        try:
            read_nortek = dolfyn.io.nortek.read_nortek
        except AttributeError as exc:
            raise ImportError(
                "The installed DOLfYN package does not expose "
                "io.nortek.read_nortek(). Please install a recent MHKiT build "
                'with: pip install "mhkit[dolfyn]"'
            ) from exc
        source = f"{source_prefix}.io.nortek.read_nortek"

    return read_nortek, source


@contextmanager
def _patched_aquadopp_template(
    filename: str,
    read_nortek: Callable[..., xr.Dataset],
) -> Iterator[dict[str, str] | None]:
    """Temporarily fill missing Aquadopp 0x01 template fields if needed."""
    if not filename.lower().endswith(".aqd"):
        yield None
        return

    module_name = getattr(read_nortek, "__module__", "")
    if not module_name:
        yield None
        return

    try:
        nortek_module = importlib.import_module(module_name)
        defs_base = (
            module_name
            if module_name.endswith(".io")
            else module_name.rsplit(".", 1)[0]
        )
        defs_module = importlib.import_module(f"{defs_base}.nortek_defs")
    except (ImportError, ValueError):
        yield None
        return

    vec_data = getattr(defs_module, "vec_data", None)
    vec_sys = getattr(defs_module, "vec_sys", None)
    if not isinstance(vec_data, dict) or not isinstance(vec_sys, dict):
        yield None
        return

    missing = [
        field
        for field in _AQUADOPP_TEMPLATE_FIELDS
        if field not in vec_data and field in vec_sys
    ]
    if not missing:
        yield None
        return

    patched = dict(vec_data)
    for field in missing:
        patched[field] = vec_sys[field]

    module_defs = getattr(nortek_module, "defs", None)
    if not hasattr(module_defs, "vec_data"):
        yield None
        return

    original_defs = defs_module.vec_data
    original_nortek_defs = module_defs.vec_data
    defs_module.vec_data = patched
    module_defs.vec_data = patched
    try:
        yield {
            "name": "aquadopp_single_point_template",
            "scope": "in_memory_for_this_read",
            "reason": (
                "Adds missing timestamp and environmental variable definitions "
                "for classic Aquadopp 0x01 blocks when the backend template is "
                "incomplete."
            ),
        }
    finally:
        defs_module.vec_data = original_defs
        module_defs.vec_data = original_nortek_defs


class NortekRawReader(AbstractReader):
    """Read Nortek raw binary files with MHKiT DOLfYN.

    The reader delegates binary decoding to DOLfYN and keeps the returned
    xarray structure intact. SeaSenseLib adds compact provenance, raw metadata
    hints, and conservative mappings for clearly identified scalar variables.

    This reader is marked experimental because support is still being validated
    across Nortek raw variants.

    Velocity is intentionally preserved as vector variable ``vel``. Its
    component meaning depends on ``ds.attrs["coord_sys"]`` and the ``dir``
    coordinate, so automatic CF component variables would be a scientific
    interpretation step rather than a safe reader cleanup.
    """

    def __init__(
        self,
        input_file: str,
        userdata: bool | str | None = None,
        nens: int | tuple[int, int] | None = None,
        debug: bool | None = None,
        do_checksum: bool | None = None,
        show_decoder_output: bool = False,
        apply_aquadopp_compatibility: bool = True,
        mapping: dict | None = None,
        **kwargs,
    ):
        """Initialize the Nortek raw reader.

        Parameters
        ----------
        input_file
            Path to a Nortek raw binary file. This reader currently advertises
            classic Nortek ``.aqd``, ``.vec`` and ``.wpr`` files.
        userdata
            Passed to DOLfYN. Use ``True`` to search for a sibling
            ``*.userdata.json`` file, ``False`` to skip it, or a string path
            to use a specific metadata file. ``None`` keeps DOLfYN's default.
        nens
            Number of ensembles to read, or a start/stop tuple for backend
            versions that support sliced reads.
        debug
            DOLfYN debug flag. ``None`` keeps DOLfYN's default.
        do_checksum
            Ask DOLfYN to verify Nortek block checksums. ``None`` keeps the
            backend default.
        show_decoder_output
            If True, let the backend decoder write progress messages to
            stdout. The default keeps SeaSenseLib reads quiet.
        apply_aquadopp_compatibility
            Apply a small in-memory compatibility patch for DOLfYN builds where
            classic Aquadopp 0x01 blocks miss timestamp/environmental template
            definitions. The raw binary values are still decoded by DOLfYN.
        mapping
            Additional SeaSenseLib variable mappings.
        **kwargs
            Base reader options such as ``perform_default_postprocessing`` and
            ``use_steps``.
        """
        self._userdata = userdata
        self._nens = nens
        self._debug = debug
        self._do_checksum = do_checksum
        self._show_decoder_output = show_decoder_output
        self._apply_aquadopp_compatibility = apply_aquadopp_compatibility
        self._decoder_source = ""
        self._compatibility_notes: list[dict[str, str]] = []
        self._raw_metadata_blocks: dict[str, Any] = {}
        self._raw_metadata_variables: dict[str, Any] = {}
        super().__init__(input_file, mapping, **kwargs)
        self._validate_file()

    @classmethod
    def _get_valid_extensions(cls) -> tuple[str, ...]:
        """Return supported Nortek raw binary extensions."""
        return _SUPPORTED_EXTENSIONS

    def _load_data(self) -> xr.Dataset:
        """Load the Nortek raw data and return an xarray Dataset."""
        read_nortek, source = _read_nortek_import()
        self._decoder_source = source
        ds = self._call_read_nortek(read_nortek)
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"DOLfYN returned {type(ds)!r}; expected xarray.Dataset."
            )

        ds = self._annotate_decoded_dataset(ds)
        self._raw_metadata_blocks = self._build_raw_metadata_blocks(ds)
        self._raw_metadata_variables = self._build_raw_variable_metadata(ds)
        return ds

    def _call_read_nortek(
        self,
        read_nortek: Callable[..., xr.Dataset],
    ) -> xr.Dataset:
        """Call DOLfYN's Nortek reader while tolerating small API changes."""
        kwargs = self._read_kwargs(read_nortek)

        if self._show_decoder_output:
            return self._read_with_optional_compatibility(read_nortek, kwargs)

        captured_stdout = io.StringIO()
        try:
            with redirect_stdout(captured_stdout):
                return self._read_with_optional_compatibility(read_nortek, kwargs)
        finally:
            output = captured_stdout.getvalue().strip()
            if output:
                logger.debug("Suppressed Nortek decoder stdout:\n%s", output)

    def _read_kwargs(self, read_nortek: Callable[..., xr.Dataset]) -> dict[str, Any]:
        """Build backend keyword arguments accepted by this DOLfYN version."""
        kwargs: dict[str, Any] = {}
        for name, value in (
            ("userdata", self._userdata),
            ("nens", self._nens),
            ("debug", self._debug),
            ("do_checksum", self._do_checksum),
        ):
            if value is not None:
                kwargs[name] = value

        signature = inspect.signature(read_nortek)
        parameters = signature.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return kwargs

    def _read_with_optional_compatibility(
        self,
        read_nortek: Callable[..., xr.Dataset],
        kwargs: dict[str, Any],
    ) -> xr.Dataset:
        """Run the backend reader, optionally applying the Aquadopp template fix."""
        if not self._apply_aquadopp_compatibility:
            return read_nortek(self.input_file, **kwargs)

        with _patched_aquadopp_template(self.input_file, read_nortek) as note:
            if note is not None:
                self._compatibility_notes.append(note)
            return read_nortek(self.input_file, **kwargs)

    def _annotate_decoded_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Add SeaSenseLib provenance and conservative metadata defaults."""
        ds = self._rename_non_measurement_fields(ds)
        self._normalize_time_attrs(ds)

        for variable_name, attrs in _NORTEK_ATTR_DEFAULTS.items():
            if variable_name not in ds:
                continue
            for attr_name, attr_value in attrs.items():
                ds[variable_name].attrs.setdefault(attr_name, attr_value)

        self._normalize_orientation_standard_names(ds)
        self._annotate_environmental_sources(ds)
        self._annotate_orientation_sources(ds)

        for variable_name in list(ds.coords) + list(ds.data_vars):
            variable_attrs = ds[variable_name].attrs
            variable_attrs.setdefault("original_name", variable_name)
            if "units" in variable_attrs:
                variable_attrs.setdefault("original_units", variable_attrs["units"])

        if "range" in ds.coords:
            ds["range"].attrs.setdefault("units", "m")
            ds["range"].attrs.setdefault("long_name", "Profile Range")
            ds["range"].attrs.setdefault(
                "comment",
                "Distance from transducer to cell center, as decoded from raw data.",
            )
        if "vel" in ds:
            coord_sys = str(ds.attrs.get("coord_sys", "")).strip().lower()
            if coord_sys:
                ds["vel"].attrs.setdefault("coordinate_system", coord_sys)
        return ds

    def _rename_non_measurement_fields(self, ds: xr.Dataset) -> xr.Dataset:
        """Avoid presenting configured or placeholder fields as measurements."""
        rename_map = {}
        sound_speed_setting = (
            "c_sound" in ds
            and self._is_truthy(ds.attrs.get("user_specified_sound_speed"))
        )
        pressure_placeholder = False
        pressure_placeholder_basis = ""
        if "pressure" in ds:
            pressure_all_zero = self._is_zero_placeholder(ds["pressure"])
            pressure_available = self._pressure_sensor_available(ds)
            pressure_placeholder = pressure_all_zero or not pressure_available
            if pressure_all_zero:
                pressure_placeholder_basis = "nortek_pressure_all_zero"
            elif not pressure_available:
                pressure_placeholder_basis = "nortek_pressure_sensor_header"
        correlation_placeholder = (
            "corr" in ds and self._is_zero_placeholder(ds["corr"])
        )

        if sound_speed_setting:
            rename_map["c_sound"] = _NORTEK_SOUND_SPEED_SETTING
        if pressure_placeholder:
            rename_map["pressure"] = _NORTEK_PRESSURE_PLACEHOLDER
        if correlation_placeholder:
            rename_map["corr"] = _NORTEK_CORRELATION_PLACEHOLDER

        renamed = ds.rename(rename_map) if rename_map else ds
        for original_name, renamed_name in rename_map.items():
            attrs = renamed[renamed_name].attrs
            attrs.setdefault("original_name", original_name)
            if "units" in attrs:
                attrs.setdefault("original_units", attrs["units"])
            attrs.pop("standard_name", None)
            attrs.update(_NORTEK_ATTR_DEFAULTS[renamed_name])
            if renamed_name == _NORTEK_SOUND_SPEED_SETTING:
                self._set_source_attrs(
                    attrs,
                    "configured",
                    "nortek_user_specified_sound_speed",
                )
            elif renamed_name == _NORTEK_PRESSURE_PLACEHOLDER:
                self._set_source_attrs(
                    attrs,
                    "placeholder",
                    pressure_placeholder_basis or "nortek_pressure_all_zero",
                )
            elif renamed_name == _NORTEK_CORRELATION_PLACEHOLDER:
                self._set_source_attrs(
                    attrs,
                    "placeholder",
                    "nortek_correlation_all_zero",
                )
        return renamed

    @staticmethod
    def _normalize_time_attrs(ds: xr.Dataset) -> None:
        """Move datetime serialization attrs away from xarray-reserved keys."""
        for coord_name in ds.coords:
            coord = ds[coord_name]
            if not np.issubdtype(coord.dtype, np.datetime64):
                continue
            attrs = coord.attrs
            if "units" in attrs:
                attrs.setdefault("original_units", attrs.pop("units"))
            if "calendar" in attrs:
                attrs.setdefault("original_calendar", attrs.pop("calendar"))

    @staticmethod
    def _normalize_orientation_standard_names(ds: xr.Dataset) -> None:
        """Use SeaSenseLib's orientation standard names while keeping originals."""
        for variable_name, canonical in (
            ("heading", params.HEADING),
            ("pitch", params.PITCH),
            ("roll", params.ROLL),
        ):
            if variable_name not in ds:
                continue
            expected = params.metadata.get(canonical, {}).get("standard_name")
            if not expected:
                continue
            attrs = ds[variable_name].attrs
            current = attrs.get("standard_name")
            if current and current != expected:
                attrs.setdefault("original_standard_name", current)
            attrs["standard_name"] = expected

    @staticmethod
    def _set_source_attrs(
        attrs: dict[str, Any],
        source: str,
        basis: str,
    ) -> None:
        """Set compact source annotations shared with other raw readers."""
        attrs["sensor_source"] = source
        attrs["sensor_source_basis"] = basis

    @staticmethod
    def _annotate_environmental_sources(ds: xr.Dataset) -> None:
        """Add source annotations where Nortek metadata gives clear evidence."""
        if "pressure" in ds:
            attrs = ds["pressure"].attrs
            if NortekRawReader._pressure_sensor_available(ds):
                basis = (
                    "nortek_pressure_sensor_header"
                    if ds.attrs.get("pressure_sensor") is not None
                    else "nortek_pressure_nonzero"
                )
                NortekRawReader._set_source_attrs(attrs, "sensor", basis)

    @staticmethod
    def _annotate_orientation_sources(ds: xr.Dataset) -> None:
        """Add concise source comments when Nortek sensor flags are available."""
        compass = NortekRawReader._is_truthy(ds.attrs.get("compass"))
        tilt = NortekRawReader._is_truthy(ds.attrs.get("tilt_sensor"))
        if "heading" in ds and ds.attrs.get("compass") is not None:
            attrs = ds["heading"].attrs
            if compass:
                NortekRawReader._set_source_attrs(
                    attrs,
                    "sensor",
                    "nortek_compass_header",
                )
                attrs.setdefault(
                    "comment",
                    "Heading decoded from the Nortek compass sensor.",
                )
            else:
                attrs["measurement_type"] = "Unknown"
                NortekRawReader._set_source_attrs(
                    attrs,
                    "unknown",
                    "nortek_compass_header",
                )
                attrs.setdefault(
                    "comment",
                    "Heading was decoded, but the Nortek header does not "
                    "confirm an available compass sensor.",
                )
        for name in ("pitch", "roll"):
            if name not in ds or ds.attrs.get("tilt_sensor") is None:
                continue
            attrs = ds[name].attrs
            if tilt:
                NortekRawReader._set_source_attrs(
                    attrs,
                    "sensor",
                    "nortek_tilt_sensor_header",
                )
                attrs.setdefault(
                    "comment",
                    f"{name.capitalize()} decoded from the Nortek tilt sensor.",
                )
            else:
                attrs["measurement_type"] = "Unknown"
                NortekRawReader._set_source_attrs(
                    attrs,
                    "unknown",
                    "nortek_tilt_sensor_header",
                )
                attrs.setdefault(
                    "comment",
                    f"{name.capitalize()} was decoded, but the Nortek header "
                    "does not confirm an available tilt sensor.",
                )

    @staticmethod
    def _pressure_sensor_available(ds: xr.Dataset) -> bool:
        """Return False only when the header explicitly says no pressure sensor."""
        value = ds.attrs.get("pressure_sensor")
        if value is None:
            return True
        return NortekRawReader._is_truthy(value)

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        """Interpret common Nortek/DOLfYN truthy attribute values."""
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "y", "on", "available", "present"}

    @staticmethod
    def _is_zero_placeholder(array: xr.DataArray) -> bool:
        """Return True when a decoded field only contains zero-like values."""
        values = np.asarray(array.values)
        if values.size == 0:
            return False
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return False
        return bool(np.all(finite == 0))

    def _build_raw_metadata_blocks(self, ds: xr.Dataset) -> dict[str, Any]:
        """Build compact raw metadata blocks for the finalization stage."""
        blocks: dict[str, Any] = {
            "configuration": {
                "decoder": self._decoder_source,
                "status": "experimental",
                "note": _EXPERIMENTAL_NOTE,
                "reader_options": self._read_options(),
            },
            "sensor_sources": self._sensor_source_summary(ds),
            "mapping_notes": self._mapping_notes(ds),
        }
        if self._compatibility_notes:
            blocks["configuration"]["compatibility"] = list(self._compatibility_notes)
        return blocks

    def _build_raw_variable_metadata(self, ds: xr.Dataset) -> dict[str, Any]:
        """Summarize variables without copying data values."""
        variables: dict[str, Any] = {}
        for name in list(ds.coords) + list(ds.data_vars):
            array = ds[name]
            attrs = dict(array.attrs)
            variable_metadata = {"dims": list(array.dims)}
            for attr_name in (
                "original_name",
                "units",
                "original_units",
                "original_standard_name",
                "measurement_type",
                "sensor_source",
                "sensor_source_basis",
                "coordinate_system",
            ):
                if attr_name in attrs:
                    variable_metadata[attr_name] = attrs[attr_name]
            variables[name] = variable_metadata
        return variables

    def _sensor_source_summary(self, ds: xr.Dataset) -> dict[str, Any]:
        """Return compact Nortek source annotations for raw metadata."""
        fields = {}
        for name in ds.data_vars:
            attrs = ds[name].attrs
            if "sensor_source" not in attrs:
                continue
            fields[name] = {
                "source": attrs["sensor_source"],
                "basis": attrs.get("sensor_source_basis", "unknown"),
            }

        return {
            "note": (
                "sensor_source values are compact SeaSenseLib annotations. "
                "sensor_source_basis values with nortek_* are evidence from "
                "decoded Nortek header fields or conservative value checks. "
                "These annotations are not external standard terms."
            ),
            "definitions": {
                "sensor_source": _SENSOR_SOURCE_DESCRIPTIONS,
                "sensor_source_basis": _NORTEK_SENSOR_BASIS_DESCRIPTIONS,
            },
            "raw_fields": {
                "pressure_sensor": ds.attrs.get("pressure_sensor"),
                "compass": ds.attrs.get("compass"),
                "tilt_sensor": ds.attrs.get("tilt_sensor"),
                "user_specified_sound_speed": ds.attrs.get(
                    "user_specified_sound_speed"
                ),
            },
            "fields": fields,
        }

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
        """Return raw read options for provenance."""
        return {
            "userdata": self._userdata,
            "nens": self._nens,
            "debug": self._debug,
            "do_checksum": self._do_checksum,
            "show_decoder_output": self._show_decoder_output,
            "apply_aquadopp_compatibility": self._apply_aquadopp_compatibility,
        }

    def _mapping_notes(self, ds: xr.Dataset) -> dict[str, Any]:
        """Describe conservative mapping choices."""
        coord_sys = str(ds.attrs.get("coord_sys", "")).lower() or None
        safe_mappings = {}
        for source, canonical in (
            ("temp", params.TEMPERATURE),
            ("c_sound", params.SPEED_OF_SOUND),
            ("pressure", params.PRESSURE),
            ("batt", params.BATTERY_VOLTAGE),
        ):
            if source in ds:
                safe_mappings[source] = canonical

        not_mapped: dict[str, str] = {}
        for original, renamed in (
            ("c_sound", _NORTEK_SOUND_SPEED_SETTING),
            ("pressure", _NORTEK_PRESSURE_PLACEHOLDER),
            ("corr", _NORTEK_CORRELATION_PLACEHOLDER),
        ):
            if renamed in ds:
                not_mapped[original] = f"Kept as {renamed}; not a safe measurement mapping."
        if "vel" in ds:
            not_mapped["vel"] = (
                "Kept as vector variable vel; component splitting depends on "
                "coord_sys and the decoded dir coordinate."
            )
        if "amp" in ds:
            not_mapped["amp"] = (
                "Kept as vector variable amp; beam/component semantics depend "
                "on the decoded coordinate system."
            )

        return {
            "safe_reader_mappings": safe_mappings,
            "not_mapped": not_mapped,
            "velocity": {
                "source_variable": "vel",
                "coordinate_system": coord_sys,
                "cf_component_mapping": (
                    "not_applied"
                    if coord_sys != "earth"
                    else "possible_after_review"
                ),
            },
        }

    @classmethod
    def format_mappings(cls) -> dict[str, list[str]]:
        """Return conservative Nortek-to-SeaSenseLib variable mappings."""
        return {
            params.TEMPERATURE: ["temp"],
            params.SPEED_OF_SOUND: ["c_sound"],
            params.BATTERY_VOLTAGE: ["batt"],
        }

    @classmethod
    def format_key(cls) -> str:
        return "nortek-raw"

    @classmethod
    def format_name(cls) -> str:
        return "Nortek Raw (experimental)"

    @classmethod
    def file_extension(cls) -> str | None:
        return ".aqd"

    @classmethod
    def file_extensions(cls) -> tuple[str, ...]:
        return _SUPPORTED_EXTENSIONS


__all__ = ["NortekRawReader"]
