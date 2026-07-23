from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from seasenselib.pipeline.base import StageContext
from seasenselib.pipeline.config import PipelineConfig
from seasenselib.pipeline.transformation import TransformationStage
from seasenselib.pipeline.transformation.handlers import NortekCoordinateTransformation
from seasenselib.readers.netcdf_reader import NetCdfReader
from seasenselib.readers.nortek_csv_reader import NortekCsvReader
from seasenselib.writers.netcdf_writer import NetCdfWriter


_T = np.array(
    [
        [0.7356, -0.3677, -0.3677],
        [0.0, -0.6370, 0.6370],
        [0.7888, 0.7888, 0.7888],
    ]
)


def _matrix_metadata(matrix=None):
    return {
        "raw_metadata_blocks": {
            "calibration": {
                "transformation_matrix": (matrix if matrix is not None else _T).tolist()
            }
        }
    }


def _beam_dataset():
    return xr.Dataset(
        {
            "velocity_beam1": ("time", np.array([0.10, 0.20, np.nan])),
            "velocity_beam2": ("time", np.array([0.30, 0.40, 0.50])),
            "velocity_beam3": ("time", np.array([0.50, 0.60, 0.70])),
            "heading": ("time", np.array([120.0, 121.0, 122.0])),
            "pitch": ("time", np.array([5.0, 5.0, 5.0])),
            "roll": ("time", np.array([-2.0, -2.0, -2.0])),
            "status_code": ("time", np.array(["00000000", "00000000", "00000000"])),
        },
        coords={"time": np.arange(3)},
        attrs={"coordinate_system": "BEAM"},
    )


def test_nortek_beam_to_xyz_matches_transformation_matrix():
    ds = _beam_dataset()
    transformer = NortekCoordinateTransformation(target_coordinate_system="XYZ")

    result, records = transformer.transform(ds, _matrix_metadata())

    source = np.vstack(
        [
            ds["velocity_beam1"].values,
            ds["velocity_beam2"].values,
            ds["velocity_beam3"].values,
        ]
    )
    expected = _T @ source
    np.testing.assert_allclose(result["x_velocity"].values, expected[0])
    np.testing.assert_allclose(result["y_velocity"].values, expected[1])
    np.testing.assert_allclose(result["z_velocity"].values, expected[2])
    assert {"velocity_beam1", "velocity_beam2", "velocity_beam3"}.isdisjoint(
        result.data_vars
    )
    assert result.attrs["coordinate_system"] == "XYZ"
    assert records[0].parameters["source_coordinate_system"] == "BEAM"
    assert records[0].parameters["target_coordinate_system"] == "XYZ"


def test_nortek_beam_to_enu_roundtrips_to_beam():
    ds = _beam_dataset()
    ds["velocity_beam1"] = ds["velocity_beam1"].fillna(0.70)
    metadata = _matrix_metadata()

    enu, _ = NortekCoordinateTransformation("ENU").transform(ds, metadata)
    back, _ = NortekCoordinateTransformation("BEAM").transform(enu, metadata)

    for name in ("velocity_beam1", "velocity_beam2", "velocity_beam3"):
        np.testing.assert_allclose(back[name].values, ds[name].values, equal_nan=True)
    assert {"east_velocity", "north_velocity", "up_velocity"}.isdisjoint(
        back.data_vars
    )
    assert back.attrs["coordinate_system"] == "BEAM"


def test_nortek_integer_matrix_is_scaled_by_4096():
    ds = _beam_dataset()
    metadata = _matrix_metadata(_T * 4096.0)

    result, records = NortekCoordinateTransformation("XYZ").transform(ds, metadata)

    source = np.vstack(
        [
            ds["velocity_beam1"].values,
            ds["velocity_beam2"].values,
            ds["velocity_beam3"].values,
        ]
    )
    np.testing.assert_allclose(result["x_velocity"].values, (_T @ source)[0])
    assert records[0].parameters["transformation_matrix_scale"] == "divided_by_4096"


def test_nortek_enu_transform_requires_orientation_variables():
    ds = _beam_dataset().drop_vars("heading")

    with pytest.raises(ValueError, match="requires heading, pitch, and roll"):
        NortekCoordinateTransformation("ENU").transform(ds, _matrix_metadata())


def test_nortek_coordinate_handler_supports_cell_suffixes():
    ds = _beam_dataset().rename(
        {
            "velocity_beam1": "velocity_beam1_cell2",
            "velocity_beam2": "velocity_beam2_cell2",
            "velocity_beam3": "velocity_beam3_cell2",
        }
    )

    result, _ = NortekCoordinateTransformation("XYZ").transform(ds, _matrix_metadata())

    assert {"x_velocity_cell2", "y_velocity_cell2", "z_velocity_cell2"}.issubset(
        result.data_vars
    )
    assert "velocity_beam1_cell2" not in result


def test_configured_transformation_stage_runs_nortek_handler():
    ds = _beam_dataset()
    stage = TransformationStage()
    stage.configure(
        {
            "handlers": ["nortek_coordinate_system"],
            "nortek_coordinate_system": {
                "target_coordinate_system": "XYZ",
            },
        }
    )

    result = stage.process(StageContext(ds, _matrix_metadata()))

    assert result.metadata["handlers_applied"] == [
        "transformation:nortek_coordinate_system"
    ]
    assert result.dataset.attrs["processor_transformations_count"] == 1
    payload = json.loads(result.dataset.attrs["processor_transformations"])
    assert payload[0]["parameters"]["target_coordinate_system"] == "XYZ"


def _write_nortek_csv_files(tmp_path):
    csv_file = tmp_path / "Average Velocity DF3.csv"
    csv_file.write_text(
        "\n".join(
            [
                (
                    "dateTime;serialNumber;temperature;pressure;heading;pitch;roll;"
                    "coordinateSystem;velBeam1#1;velBeam2#1;velBeam3#1"
                ),
                "2026-07-11 12:00:00;A123;8.1;12.5;101.0;1.2;-0.5;BEAM;0.11;0.21;0.31",
                "2026-07-11 12:00:01;A123;8.2;12.6;102.0;1.3;-0.6;BEAM;0.12;0.22;0.32",
            ]
        ),
        encoding="utf-8",
    )

    header_file = tmp_path / "String Data.csv"
    header_file.write_text(
        "\n".join(
            [
                "idx;string",
                (
                    '0;ID,STR="Aquadopp",SN=400115|'
                    'GETAVG,NC=1,CY="BEAM",NB=3|'
                    'GETXFAVG,ROWS=3,COLS=3,'
                    'M11=0.7356,M12=-0.3677,M13=-0.3677,'
                    'M21=0.0000,M22=-0.6370,M23=0.6370,'
                    'M31=0.7888,M32=0.7888,M33=0.7888|'
                    'GETHW,FW=10003'
                ),
            ]
        ),
        encoding="utf-8",
    )
    return csv_file, header_file


def test_nortek_csv_to_netcdf_enu_then_back_to_beam_roundtrip(tmp_path):
    csv_file, header_file = _write_nortek_csv_files(tmp_path)
    original = NortekCsvReader(
        str(csv_file),
        input_header_file=str(header_file),
        perform_default_postprocessing=False,
    ).data

    transformed = NortekCsvReader(
        str(csv_file),
        input_header_file=str(header_file),
        target_coordinate_system="ENU",
        pointing_down=False,
    ).data
    assert transformed.attrs["coordinate_system"] == "ENU"
    assert {"east_velocity", "north_velocity", "up_velocity"}.issubset(
        transformed.data_vars
    )

    output = tmp_path / "transformed.nc"
    NetCdfWriter(transformed).write(str(output))

    config = PipelineConfig()
    config.add_stage(
        "transformation",
        config={
            "handlers": ["nortek_coordinate_system"],
            "nortek_coordinate_system": {
                "target_coordinate_system": "BEAM",
                "pointing_down": False,
            },
        },
    )
    back = NetCdfReader(str(output), pipeline_config=config).data

    for name in ("velocity_beam1", "velocity_beam2", "velocity_beam3"):
        assert name in back
        np.testing.assert_allclose(back[name].values, original[name].values)
    assert {"east_velocity", "north_velocity", "up_velocity"}.isdisjoint(
        back.data_vars
    )
