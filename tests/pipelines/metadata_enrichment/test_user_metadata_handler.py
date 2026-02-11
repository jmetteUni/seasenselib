import numpy as np
import xarray as xr

from seasenselib.pipeline.metadata_enrichment.handlers.user_metadata_handler import UserMetadataHandler
from seasenselib.pipeline.interfaces import MetadataRegistry


def _make_dataset():
    return xr.Dataset(
        {
            "temperature": ("time", [10.0, 11.0]),
        },
        coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")},
    )


def test_user_metadata_applies_globals_and_variables_and_filters_reserved():
    ds = _make_dataset()
    handler = UserMetadataHandler()
    handler.set_user_metadata({
        "global": {
            "cruise": "Test Cruise",
            "raw_format": "should_be_ignored",
        },
        "variables": {
            "temperature": {
                "units": "degC",
                "raw_note": "ignore",
            },
            "unknown": {
                "note": "missing",
            },
        },
    })

    result = handler.enrich(ds, MetadataRegistry())

    assert result.attrs["cruise"] == "Test Cruise"
    assert "raw_format" not in result.attrs
    assert result["temperature"].attrs["units"] == "degC"
    assert "raw_note" not in result["temperature"].attrs

    warnings = handler.warnings
    assert any("reserved global attribute 'raw_format'" in w for w in warnings)
    assert any("reserved attribute 'temperature.raw_note'" in w for w in warnings)
    assert any("unknown variable 'unknown'" in w for w in warnings)

    assert handler.applied is True
    assert "cruise" in handler.applied_global_keys
    assert "temperature" in handler.applied_variable_keys


def test_user_metadata_noop_when_missing():
    ds = _make_dataset()
    handler = UserMetadataHandler()
    handler.set_user_metadata(None)

    result = handler.enrich(ds, MetadataRegistry())

    assert result is ds
    assert handler.applied is False
