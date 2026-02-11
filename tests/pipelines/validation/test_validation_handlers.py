import numpy as np
import pytest
import xarray as xr

from seasenselib.pipeline.base import StageContext
from seasenselib.pipeline.validation.handlers.cf_validator import CFValidator
from seasenselib.pipeline.validation.handlers.unit_validator import UnitValidator
from seasenselib.pipeline.validation.handlers.validation_runner import ValidationRunner
from seasenselib.pipeline.interfaces import IValidator, ValidationError


def test_cf_validator_valid_dataset():
    ds = xr.Dataset(
        {"temperature": (["time"], [10.0, 11.0])},
        coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")}
    )
    ds.attrs["Conventions"] = "CF-1.13"
    ds["temperature"].attrs["units"] = "K"
    ds["temperature"].attrs["standard_name"] = "sea_water_temperature"
    ds["time"].attrs["units"] = "days since 1970-01-01"

    validator = CFValidator()
    issues = validator.validate(ds)
    assert issues == []


def test_cf_validator_reports_missing():
    ds = xr.Dataset({"temperature": (["time"], [10.0, 11.0])})
    validator = CFValidator()
    issues = validator.validate(ds)
    assert any(i.severity == "error" for i in issues)
    assert any(i.severity == "warning" for i in issues)
    assert any(i.severity == "info" for i in issues)


def test_unit_validator_reports_missing_and_deprecated():
    ds = xr.Dataset({"temperature": (["time"], [10.0, 11.0])})
    validator = UnitValidator()
    issues = validator.validate(ds)
    assert any(i.severity == "warning" for i in issues)

    ds["temperature"].attrs["units"] = "degC"
    issues = validator.validate(ds)
    assert any(i.severity == "info" for i in issues)


class DummyValidator(IValidator):
    def __init__(self, issues):
        self._issues = issues

    def name(self) -> str:
        return "dummy"

    def validate(self, dataset: xr.Dataset):
        return self._issues


def test_validation_runner_collects_and_strict():
    ds = xr.Dataset({"temperature": (["time"], [1.0, 2.0])})
    errors = [ValidationError("bad", severity="error")]
    warnings = [ValidationError("warn", severity="warning")]
    infos = [ValidationError("info", severity="info")]

    runner = ValidationRunner(
        validators=[DummyValidator(errors + warnings + infos)],
        strict=False
    )
    result = runner.process(StageContext(ds, {}))
    validation = result.metadata["validation"]
    assert validation["errors"]
    assert validation["warnings"]
    assert validation["infos"]

    runner_strict = ValidationRunner(
        validators=[DummyValidator(errors)],
        strict=True
    )
    with pytest.raises(ValueError):
        runner_strict.process(StageContext(ds, {}))
