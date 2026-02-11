import numpy as np
import pytest
import xarray as xr

from seasenselib.pipeline.unit_handling.handlers.unit_normalizer import UnitNormalizer
from seasenselib.pipeline.unit_handling.handlers.unit_converter import UnitConverter
import seasenselib.pipeline.unit_handling.handlers.unit_converter as converter_mod


def test_unit_normalizer_basic_conversion():
    ds = xr.Dataset({"temperature": (["time"], [10.0, 11.0])})
    ds["temperature"].attrs["units"] = "degC"
    normalizer = UnitNormalizer(strict=False, auto_convert=True)

    result, issues, conversions = normalizer.normalize(ds)
    assert result["temperature"].attrs["units"] == "degC"
    assert issues == []
    assert conversions


def test_unit_normalizer_strict_missing_units():
    ds = xr.Dataset({"temperature": (["time"], [10.0, 11.0])})
    normalizer = UnitNormalizer(strict=True, auto_convert=False)
    with pytest.raises(ValueError):
        normalizer.normalize(ds)


def test_unit_normalizer_missing_units_non_strict():
    ds = xr.Dataset({"temperature": (["time"], [10.0, 11.0])})
    normalizer = UnitNormalizer(strict=False, auto_convert=False)
    _, issues, conversions = normalizer.normalize(ds)
    assert issues
    assert conversions == []


def test_unit_converter_no_units_no_conversion():
    ds = xr.Dataset({"temperature": (["time"], [10.0, 11.0])})
    converter = UnitConverter(expected_units={"temperature": "K"})
    result, conversions = converter.convert(ds)
    assert result is ds
    assert conversions == []


def test_unit_converter_with_pint_if_available():
    if not converter_mod._HAS_PINT:
        pytest.skip("pint not available")

    ds = xr.Dataset({"temperature": (["time"], np.array([0.0, 10.0]))})
    ds["temperature"].attrs["units"] = "degC"
    converter = UnitConverter(
        expected_units={"temperature": "K"},
        conversion_mode="duplicate_keep_original",
        original_suffix="_orig",
    )
    result, conversions = converter.convert(ds)

    assert "temperature_orig" in result
    assert result["temperature"].attrs["units"] == "K"
    assert result["temperature"].attrs.get("units_original") == "degC"
    assert conversions
