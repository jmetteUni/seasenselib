"""Nortek velocity coordinate-system transformations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, List

import numpy as np
import xarray as xr

import seasenselib.parameters as params
from ...interfaces import ITransformation, TransformationRecord


logger = logging.getLogger(__name__)


_VALID_COORDINATE_SYSTEMS = {"BEAM", "XYZ", "ENU"}
_COORDINATE_ALIASES = {
    "BEAM": "BEAM",
    "BEAMS": "BEAM",
    "XYZ": "XYZ",
    "INST": "XYZ",
    "INSTRUMENT": "XYZ",
    "ENU": "ENU",
    "EARTH": "ENU",
}
_BEAM_RE = re.compile(r"^velocity_beam(?P<component>[123])(?P<suffix>_cell\d+)?$")
_XYZ_RE = re.compile(r"^(?P<component>[xyz])_velocity(?P<suffix>_cell\d+)?$")
_ENU_RE = re.compile(
    r"^(?P<component>east|north|up)_velocity(?P<suffix>_cell\d+)?$"
)


@dataclass(frozen=True)
class _VelocityGroup:
    suffix: str
    source_variables: tuple[str, str, str]
    target_variables: tuple[str, str, str]


@dataclass(frozen=True)
class _MatrixInfo:
    matrix: np.ndarray
    source: str
    scale: str


@dataclass(frozen=True)
class _PointingDownInfo:
    values: xr.DataArray | bool
    source: str


class NortekCoordinateTransformation(ITransformation):
    """Transform Nortek velocity components between BEAM, XYZ, and ENU."""

    def __init__(
        self,
        target_coordinate_system: str | None = None,
        source_coordinate_system: str | None = None,
        transformation_matrix: Any = None,
        pointing_down: bool | str | None = None,
        keep_source: bool = False,
        overwrite: bool = False,
    ):
        self.target_coordinate_system = _normalize_coordinate_system(
            target_coordinate_system,
            allow_none=True,
        )
        self.source_coordinate_system = _normalize_coordinate_system(
            source_coordinate_system,
            allow_none=True,
        )
        self.transformation_matrix = transformation_matrix
        self.pointing_down = _optional_bool(pointing_down)
        self.keep_source = bool(keep_source)
        self.overwrite = bool(overwrite)

    def name(self) -> str:
        return "nortek_coordinate_system"

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the handler from pipeline configuration."""
        if not isinstance(config, dict):
            return

        target = config.get("target_coordinate_system", config.get("target"))
        source = config.get("source_coordinate_system", config.get("source"))
        if target is not None:
            self.target_coordinate_system = _normalize_coordinate_system(target)
        if source is not None:
            self.source_coordinate_system = _normalize_coordinate_system(source)
        if "transformation_matrix" in config:
            self.transformation_matrix = config["transformation_matrix"]
        if "pointing_down" in config:
            self.pointing_down = _optional_bool(config["pointing_down"])
        if "keep_source" in config:
            self.keep_source = bool(config["keep_source"])
        if "overwrite" in config:
            self.overwrite = bool(config["overwrite"])

    def can_transform(
        self,
        dataset: xr.Dataset,
        context: Dict[str, Any] | None = None,
    ) -> bool:
        return self.target_coordinate_system is not None

    def transform(
        self,
        dataset: xr.Dataset,
        context: Dict[str, Any] | None = None,
    ) -> tuple[xr.Dataset, List[TransformationRecord | Dict[str, Any]]]:
        target = self.target_coordinate_system
        if target is None:
            return dataset, []

        source = self.source_coordinate_system or _dataset_coordinate_system(dataset)
        if source is None:
            raise ValueError(
                "Nortek coordinate transformation requires a source coordinate "
                "system in the handler config or dataset metadata."
            )
        if source == target:
            logger.debug("Nortek velocity is already in %s coordinates", target)
            return dataset, []

        groups = _velocity_groups(dataset, source, target)
        if not groups:
            raise ValueError(
                f"No complete {source} velocity triplet found for Nortek "
                f"coordinate transformation to {target}."
            )

        matrix_info = (
            self._transformation_matrix(dataset, context)
            if _requires_beam_matrix(source, target)
            else None
        )
        if matrix_info is not None:
            _validate_invertible(matrix_info.matrix, matrix_info.source)

        transformed = dataset.copy()
        output_variables: list[str] = []
        input_variables: list[str] = []
        pointing_sources: set[str] = set()
        orientation_used = _requires_orientation(source, target)

        for group in groups:
            template = _broadcast_components(dataset, group.source_variables)[0]
            pointing_info = (
                self._pointing_down(dataset, template)
                if matrix_info is not None
                else None
            )
            if pointing_info is not None:
                pointing_sources.add(pointing_info.source)

            values = _transform_group(
                dataset,
                group,
                source,
                target,
                matrix_info.matrix if matrix_info else None,
                pointing_info.values if pointing_info else False,
            )
            _assign_transformed_variables(
                transformed,
                template,
                group,
                values,
                source,
                target,
                self.overwrite,
            )
            input_variables.extend(group.source_variables)
            output_variables.extend(group.target_variables)

        if not self.keep_source:
            transformed = transformed.drop_vars(sorted(set(input_variables)))

        transformed.attrs["coordinate_system"] = target
        transformed.attrs["coordinate_system_original"] = source

        parameters = {
            "source_coordinate_system": source,
            "target_coordinate_system": target,
            "input_variables": sorted(set(input_variables)),
            "output_variables": sorted(set(output_variables)),
            "keep_source": self.keep_source,
            "overwrite": self.overwrite,
            "coordinate_system_convention": (
                "BEAM uses Nortek beam velocities; XYZ uses instrument "
                "coordinates; ENU uses east, north, up earth coordinates."
            ),
        }
        if matrix_info is not None:
            parameters["transformation_matrix_beam_to_xyz"] = matrix_info.matrix.tolist()
            parameters["transformation_matrix_source"] = matrix_info.source
            parameters["transformation_matrix_scale"] = matrix_info.scale
            parameters["pointing_down_source"] = ", ".join(sorted(pointing_sources))
        if orientation_used:
            parameters["orientation_variables"] = ["heading", "pitch", "roll"]
            parameters["orientation_formula"] = (
                "Nortek support formula: R = H(heading - 90 deg) * "
                "P(pitch, roll). No magnetic declination correction is "
                "applied; heading values must already use the intended "
                "reference frame."
            )

        record = TransformationRecord(
            transformation=self.name(),
            description=(
                f"Transformed Nortek velocity components from {source} to "
                f"{target} coordinates."
            ),
            variables=sorted(set(output_variables)),
            parameters=parameters,
        )
        return transformed, [record]

    def _transformation_matrix(
        self,
        dataset: xr.Dataset,
        context: Dict[str, Any] | None,
    ) -> _MatrixInfo:
        matrix, source = _find_transformation_matrix(
            self.transformation_matrix,
            dataset,
            context or {},
        )
        if matrix is None:
            raise ValueError(
                "Nortek BEAM coordinate transformation requires the instrument "
                "BEAM-to-XYZ transformation matrix."
            )
        matrix, scale = _scale_matrix_if_needed(matrix)
        return _MatrixInfo(matrix=matrix, source=source, scale=scale)

    def _pointing_down(
        self,
        dataset: xr.Dataset,
        template: xr.DataArray,
    ) -> _PointingDownInfo:
        if self.pointing_down is not None:
            return _PointingDownInfo(
                values=bool(self.pointing_down),
                source="handler.pointing_down",
            )

        attr_value = _first_present_attr(
            dataset,
            ("nortek_pointing_down", "pointing_down"),
        )
        if attr_value is not None:
            parsed = _optional_bool(attr_value)
            if parsed is not None:
                return _PointingDownInfo(
                    values=parsed,
                    source="dataset.attrs.pointing_down",
                )

        status = _status_bit0(dataset, template)
        if status is not None:
            return _PointingDownInfo(
                values=status,
                source="status_code_bit0",
            )

        raise ValueError(
            "Nortek BEAM coordinate transformation requires instrument "
            "orientation. Pass pointing_down=true/false, provide a "
            "nortek_pointing_down attribute, or include status_code values "
            "where bit 0 encodes the orientation."
        )


def _normalize_coordinate_system(value: Any, allow_none: bool = False) -> str | None:
    if value is None or value == "":
        if allow_none:
            return None
        raise ValueError("Coordinate system is required.")
    text = str(value).strip().upper()
    normalized = _COORDINATE_ALIASES.get(text)
    if normalized is None:
        raise ValueError(
            f"Unsupported coordinate system {value!r}; expected one of "
            f"{', '.join(sorted(_VALID_COORDINATE_SYSTEMS))}."
        )
    return normalized


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "down"}:
        return True
    if text in {"0", "false", "no", "n", "off", "up"}:
        return False
    if text in {"none", "null", "auto"}:
        return None
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _dataset_coordinate_system(dataset: xr.Dataset) -> str | None:
    for key in ("coordinate_system", "nortek_coordinate_system", "coord_sys"):
        value = dataset.attrs.get(key)
        if value is None:
            continue
        try:
            return _normalize_coordinate_system(value)
        except ValueError:
            continue

    for coordinate_system in ("BEAM", "XYZ", "ENU"):
        if _velocity_groups(dataset, coordinate_system, coordinate_system):
            return coordinate_system
    return None


def _velocity_groups(
    dataset: xr.Dataset,
    source: str,
    target: str,
) -> list[_VelocityGroup]:
    source_map = _component_suffixes(dataset, source)
    groups = []
    for suffix in sorted(source_map):
        source_variables = source_map[suffix]
        if len(source_variables) != 3:
            continue
        groups.append(
            _VelocityGroup(
                suffix=suffix,
                source_variables=source_variables,
                target_variables=_target_variables(target, suffix),
            )
        )
    return groups


def _component_suffixes(
    dataset: xr.Dataset,
    coordinate_system: str,
) -> dict[str, tuple[str, str, str]]:
    pattern, order = {
        "BEAM": (_BEAM_RE, {"1": 0, "2": 1, "3": 2}),
        "XYZ": (_XYZ_RE, {"x": 0, "y": 1, "z": 2}),
        "ENU": (_ENU_RE, {"east": 0, "north": 1, "up": 2}),
    }[coordinate_system]
    groups: dict[str, list[str | None]] = {}

    for name in dataset.data_vars:
        match = pattern.match(name)
        if not match:
            continue
        suffix = match.group("suffix") or ""
        component = match.group("component").lower()
        entries = groups.setdefault(suffix, [None, None, None])
        entries[order[component]] = name

    return {
        suffix: tuple(entries)  # type: ignore[arg-type]
        for suffix, entries in groups.items()
        if all(entries)
    }


def _target_variables(coordinate_system: str, suffix: str) -> tuple[str, str, str]:
    if coordinate_system == "BEAM":
        return (
            f"velocity_beam1{suffix}",
            f"velocity_beam2{suffix}",
            f"velocity_beam3{suffix}",
        )
    if coordinate_system == "XYZ":
        return (
            f"x_velocity{suffix}",
            f"y_velocity{suffix}",
            f"z_velocity{suffix}",
        )
    return (
        f"east_velocity{suffix}",
        f"north_velocity{suffix}",
        f"up_velocity{suffix}",
    )


def _requires_beam_matrix(source: str, target: str) -> bool:
    return source == "BEAM" or target == "BEAM"


def _requires_orientation(source: str, target: str) -> bool:
    return source == "ENU" or target == "ENU"


def _find_transformation_matrix(
    explicit_matrix: Any,
    dataset: xr.Dataset,
    context: Dict[str, Any],
) -> tuple[np.ndarray | None, str]:
    if explicit_matrix is not None:
        return _matrix_from_any(explicit_matrix), "handler.transformation_matrix"

    raw_blocks = context.get("raw_metadata_blocks")
    matrix = _matrix_from_raw_blocks(raw_blocks)
    if matrix is not None:
        return matrix, "context.raw_metadata_blocks.calibration.transformation_matrix"

    matrix = _matrix_from_attrs(dataset.attrs)
    if matrix is not None:
        return matrix, "dataset.attrs.nortek_T_Mij"

    raw_metadata = dataset.attrs.get("raw_metadata")
    if raw_metadata:
        try:
            payload = json.loads(str(raw_metadata))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            matrix = _matrix_from_raw_blocks(payload.get("blocks"))
            if matrix is not None:
                return matrix, "dataset.attrs.raw_metadata.calibration.transformation_matrix"

    return None, ""


def _matrix_from_raw_blocks(raw_blocks: Any) -> np.ndarray | None:
    if not isinstance(raw_blocks, dict):
        return None
    calibration = raw_blocks.get("calibration")
    if not isinstance(calibration, dict):
        return None
    matrix = calibration.get("transformation_matrix")
    if matrix is None:
        return None
    return _matrix_from_any(matrix)


def _matrix_from_attrs(attrs: Dict[str, Any]) -> np.ndarray | None:
    values = []
    for row in range(1, 4):
        row_values = []
        for column in range(1, 4):
            key = f"nortek_T_M{row}{column}"
            if key not in attrs:
                return None
            row_values.append(attrs[key])
        values.append(row_values)
    return _matrix_from_any(values)


def _matrix_from_any(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = [
            [value[f"M{row}{column}"] for column in range(1, 4)]
            for row in range(1, 4)
        ]
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            "Nortek transformation matrix must have shape 3x3; "
            f"got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Nortek transformation matrix contains non-finite values.")
    return matrix


def _scale_matrix_if_needed(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    max_abs = float(np.max(np.abs(matrix)))
    if max_abs > 8.0:
        return matrix / 4096.0, "divided_by_4096"
    return matrix, "as_provided"


def _validate_invertible(matrix: np.ndarray, source: str) -> None:
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-12:
        raise ValueError(
            "Nortek transformation matrix is singular or nearly singular "
            f"({source})."
        )


def _first_present_attr(dataset: xr.Dataset, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in dataset.attrs:
            return dataset.attrs[name]
    return None


def _status_bit0(dataset: xr.Dataset, template: xr.DataArray) -> xr.DataArray | None:
    status_name = next(
        (name for name in ("status_code", "status") if name in dataset),
        None,
    )
    if status_name is None:
        return None

    status, _ = xr.broadcast(dataset[status_name], template)
    values = np.asarray(status.values).reshape(-1)
    parsed = [_parse_status_value(value) for value in values]
    if any(value is None for value in parsed):
        return None

    bit0 = np.asarray([bool(value & 1) for value in parsed], dtype=bool)
    return xr.DataArray(
        bit0.reshape(status.shape),
        dims=status.dims,
        coords=status.coords,
    )


def _parse_status_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    if re.fullmatch(r"[01]{2,}", text):
        return int(text, 2)
    try:
        return int(float(text))
    except ValueError:
        return None


def _broadcast_components(
    dataset: xr.Dataset,
    source_variables: tuple[str, str, str],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    return xr.broadcast(*(dataset[name] for name in source_variables))


def _transform_group(
    dataset: xr.Dataset,
    group: _VelocityGroup,
    source: str,
    target: str,
    matrix: np.ndarray | None,
    pointing_down: xr.DataArray | bool,
) -> np.ndarray:
    components = _broadcast_components(dataset, group.source_variables)
    template = components[0]
    flat = np.stack([np.asarray(array.values, dtype=float) for array in components])
    flat = flat.reshape(3, -1)

    rotation = (
        _rotation_matrices(dataset, template)
        if _requires_orientation(source, target)
        else None
    )
    beam_matrix = (
        _effective_beam_matrix(matrix, pointing_down, template)
        if matrix is not None
        else None
    )

    if source == "BEAM":
        xyz = _matmul_matrix(beam_matrix, flat)
    elif source == "ENU":
        xyz = _solve_matrix(rotation, flat)
    else:
        xyz = flat

    if target == "BEAM":
        transformed = _solve_matrix(beam_matrix, xyz)
    elif target == "ENU":
        transformed = _matmul_matrix(rotation, xyz)
    else:
        transformed = xyz

    return transformed.reshape((3, *template.shape))


def _effective_beam_matrix(
    matrix: np.ndarray,
    pointing_down: xr.DataArray | bool,
    template: xr.DataArray,
) -> np.ndarray:
    if isinstance(pointing_down, bool):
        effective = matrix.copy()
        if pointing_down:
            effective[1, :] *= -1
            effective[2, :] *= -1
        return effective

    pointing, _ = xr.broadcast(pointing_down, template)
    flat = np.asarray(pointing.values, dtype=bool).reshape(-1)
    matrices = np.broadcast_to(matrix, (flat.size, 3, 3)).copy()
    matrices[flat, 1, :] *= -1
    matrices[flat, 2, :] *= -1
    return matrices


def _rotation_matrices(dataset: xr.Dataset, template: xr.DataArray) -> np.ndarray:
    missing = [name for name in ("heading", "pitch", "roll") if name not in dataset]
    if missing:
        raise ValueError(
            "Nortek ENU coordinate transformation requires heading, pitch, "
            f"and roll variables; missing: {', '.join(missing)}."
        )

    heading = _angle_values_degrees(dataset["heading"], template)
    pitch = _angle_values_degrees(dataset["pitch"], template)
    roll = _angle_values_degrees(dataset["roll"], template)

    hdg = np.radians(heading - 90.0)
    pch = np.radians(pitch)
    rll = np.radians(roll)

    rotation = np.empty((heading.size, 3, 3), dtype=float)
    cos_h = np.cos(hdg)
    sin_h = np.sin(hdg)
    cos_p = np.cos(pch)
    sin_p = np.sin(pch)
    cos_r = np.cos(rll)
    sin_r = np.sin(rll)

    h_matrix = np.zeros((heading.size, 3, 3), dtype=float)
    h_matrix[:, 0, 0] = cos_h
    h_matrix[:, 0, 1] = sin_h
    h_matrix[:, 1, 0] = -sin_h
    h_matrix[:, 1, 1] = cos_h
    h_matrix[:, 2, 2] = 1.0

    p_matrix = np.zeros((heading.size, 3, 3), dtype=float)
    p_matrix[:, 0, 0] = cos_p
    p_matrix[:, 0, 1] = -sin_p * sin_r
    p_matrix[:, 0, 2] = -cos_r * sin_p
    p_matrix[:, 1, 1] = cos_r
    p_matrix[:, 1, 2] = -sin_r
    p_matrix[:, 2, 0] = sin_p
    p_matrix[:, 2, 1] = sin_r * cos_p
    p_matrix[:, 2, 2] = cos_p * cos_r

    rotation[:] = np.einsum("nij,njk->nik", h_matrix, p_matrix)
    return rotation


def _angle_values_degrees(array: xr.DataArray, template: xr.DataArray) -> np.ndarray:
    broadcasted, _ = xr.broadcast(array, template)
    values = np.asarray(broadcasted.values, dtype=float).reshape(-1)
    units = str(array.attrs.get("units", "")).strip().lower()
    if units in {"rad", "radian", "radians"}:
        values = np.degrees(values)
    return values


def _matmul_matrix(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix @ values
    return np.einsum("nij,jn->in", matrix, values)


def _solve_matrix(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return np.linalg.solve(matrix, values)
    return np.linalg.solve(matrix, values.T[..., None])[..., 0].T


def _assign_transformed_variables(
    dataset: xr.Dataset,
    template: xr.DataArray,
    group: _VelocityGroup,
    values: np.ndarray,
    source: str,
    target: str,
    overwrite: bool,
) -> None:
    conflicts = [
        name
        for name in group.target_variables
        if name in dataset and name not in group.source_variables
    ]
    if conflicts and not overwrite:
        raise ValueError(
            "Nortek coordinate transformation target variables already exist: "
            f"{', '.join(conflicts)}. Set overwrite=true to replace them."
        )

    for index, variable_name in enumerate(group.target_variables):
        attrs = _target_attrs(variable_name, group.source_variables, source, target)
        dataset[variable_name] = xr.DataArray(
            values[index],
            dims=template.dims,
            coords=template.coords,
            attrs=attrs,
        )


def _target_attrs(
    variable_name: str,
    source_variables: tuple[str, str, str],
    source: str,
    target: str,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "units": "m/s",
        "coordinate_system": target,
        "transformation_source_coordinate_system": source,
        "transformation_source_variables": ", ".join(source_variables),
    }
    if target == "ENU":
        attrs.update(params.metadata.get(variable_name.split("_cell", 1)[0], {}))
    elif target == "XYZ":
        axis = variable_name.split("_", 1)[0].upper()
        attrs["long_name"] = f"{axis} velocity in instrument XYZ coordinates"
    else:
        beam = variable_name.split("_cell", 1)[0].replace("velocity_beam", "")
        attrs["long_name"] = f"Velocity Beam {beam}"
    return attrs


def transformed_coordinate_system_from_metadata(
    metadata: Dict[str, Any] | None,
) -> str | None:
    """Return the last target coordinate system recorded by this handler."""
    if not isinstance(metadata, dict):
        return None
    transformations = metadata.get("transformations")
    if not isinstance(transformations, list):
        return None
    target = None
    for record in transformations:
        if not isinstance(record, dict):
            continue
        if record.get("transformation") != "nortek_coordinate_system":
            continue
        parameters = record.get("parameters")
        if not isinstance(parameters, dict):
            continue
        target = parameters.get("target_coordinate_system") or target
    return str(target) if target else None


__all__ = [
    "NortekCoordinateTransformation",
    "transformed_coordinate_system_from_metadata",
]
