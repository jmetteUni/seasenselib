import numpy as np
import xarray as xr

from seasenselib.pipeline.metadata_enrichment.handlers.cf_convention import CFConvention
from seasenselib.pipeline.metadata_enrichment.handlers.acdd_convention import ACDDConvention
from seasenselib.pipeline.interfaces import MetadataRegistry


def test_cf_convention_enrich_adds_metadata():
    ds = xr.Dataset(
        {"temperature": (["time"], [10.0, 11.0, 12.0])},
        coords={"time": np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64')}
    )
    conv = CFConvention()
    enriched = conv.enrich(ds, MetadataRegistry())

    assert enriched["temperature"].attrs.get("standard_name") == "sea_water_temperature"
    assert "Conventions" in enriched.attrs
    assert "CF-1.13" in enriched.attrs["Conventions"]
    assert enriched["time"].attrs.get("axis") == "T"


def test_cf_convention_validate_reports_missing_attrs():
    ds = xr.Dataset({"temperature": (["time"], [1.0, 2.0])})
    conv = CFConvention()
    issues = conv.validate(ds)
    assert issues


def test_acdd_convention_enrich_auto_coverage():
    ds = xr.Dataset(
        {"temperature": (["time"], [10.0, 11.0, 12.0])},
        coords={
            "time": np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64'),
            "latitude": 54.0,
            "longitude": 10.0,
        },
    )
    conv = ACDDConvention()
    enriched = conv.enrich(ds, MetadataRegistry())

    assert "Conventions" in enriched.attrs
    assert "ACDD-1.3" in enriched.attrs["Conventions"]
    assert enriched.attrs.get("geospatial_lat_min") == 54.0
    assert enriched.attrs.get("geospatial_lon_max") == 10.0
    assert str(enriched.attrs.get("time_coverage_start", "")).startswith("2020-01-01")
    assert enriched.attrs.get("standard_name_vocabulary") == "CF-1.13"
    assert enriched.attrs.get("processing_level") == "L1"


def test_acdd_convention_validate_warns_on_missing():
    ds = xr.Dataset()
    conv = ACDDConvention()
    issues = conv.validate(ds)
    assert issues

