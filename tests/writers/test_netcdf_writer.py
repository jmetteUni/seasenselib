from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from seasenselib.core.exceptions import WriterError
from seasenselib.writers.netcdf_writer import NetCdfWriter


def test_netcdf_writer_rejects_slashes_before_creating_file(tmp_path):
    ds = xr.Dataset({"cond0S/m": ("time", np.array([1.0, 2.0]))})
    output = tmp_path / "bad.nc"

    with pytest.raises(WriterError, match="--sanitize-netcdf-names"):
        NetCdfWriter(ds).write(str(output))

    assert not output.exists()


def test_netcdf_writer_preserves_existing_file_when_validation_fails(tmp_path):
    existing = xr.Dataset({"temperature": ("time", np.array([10.0]))})
    output = tmp_path / "existing.nc"
    existing.to_netcdf(output)
    original_bytes = output.read_bytes()

    invalid = xr.Dataset({"cond0S/m": ("time", np.array([1.0, 2.0]))})

    with pytest.raises(WriterError, match="NetCDF output cannot be created"):
        NetCdfWriter(invalid).write(str(output))

    assert output.read_bytes() == original_bytes


def test_netcdf_writer_can_sanitize_slashes_in_names(tmp_path):
    ds = xr.Dataset(
        {"cond0S/m": ("sample/id", np.array([1.0, 2.0]))},
        coords={"sample/id": np.array([0, 1])},
    )
    output = tmp_path / "sanitized.nc"

    NetCdfWriter(ds).write(str(output), sanitize_names=True)

    assert "cond0S/m" in ds.data_vars
    with xr.open_dataset(output) as written:
        assert "sample_id" in written.dims
        assert "sample_id" in written.coords
        assert "cond0S_m" in written.data_vars
        assert written["cond0S_m"].attrs["original_name"] == "cond0S/m"
        assert written["sample_id"].attrs["original_name"] == "sample/id"


def test_netcdf_writer_sanitized_name_collision_fails_cleanly(tmp_path):
    ds = xr.Dataset(
        {
            "cond0S/m": ("time", np.array([1.0, 2.0])),
            "cond0S_m": ("time", np.array([3.0, 4.0])),
        }
    )
    output = tmp_path / "collision.nc"

    with pytest.raises(WriterError, match="duplicate name 'cond0S_m'"):
        NetCdfWriter(ds).write(str(output), sanitize_names=True)

    assert not output.exists()


def test_netcdf_writer_writes_valid_dataset_with_cleaned_attrs(tmp_path):
    ds = xr.Dataset(
        {"conductivity": ("time", np.array([1.0, 2.0]))},
        attrs={"config": {"sensor": "SBE37"}, "optional": None},
    )
    ds["conductivity"].attrs["metadata"] = {"units_source": "CNV"}
    output = tmp_path / "valid.nc"

    NetCdfWriter(ds).write(str(output))

    with xr.open_dataset(output) as written:
        assert list(written.data_vars) == ["conductivity"]
        assert written["conductivity"].values.tolist() == [1.0, 2.0]
        assert written.attrs["config"] == '{"sensor": "SBE37"}'
        assert written.attrs["optional"] == ""
        assert (
            written["conductivity"].attrs["metadata"] == '{"units_source": "CNV"}'
        )
