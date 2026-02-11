import numpy as np
import xarray as xr

from seasenselib.pipeline.metadata_enrichment.handlers.acdd_auto_metadata import AcddAutoMetadata
from seasenselib.pipeline.interfaces import MetadataRegistry


def _make_dataset():
    ds = xr.Dataset(
        {
            "temperature": ("time", [10.0, 11.0, 12.0]),
            "pressure": ("time", [1.0, 2.0, 3.0]),
        },
        coords={
            "time": np.array(["2020-01-01T00:00:00", "2020-01-01T00:10:00", "2020-01-01T00:20:00"], dtype="datetime64"),
            "latitude": 53.4968,
            "longitude": 8.1815,
            "depth": ("time", [0.0, 5.0, 10.0]),
        },
    )
    ds.attrs["raw_format"] = "sbe-cnv"
    ds.attrs["raw_filename"] = "example.cnv"
    return ds


def test_preserve_existing_fields():
    ds = _make_dataset()
    ds.attrs["title"] = "My Title"
    ds.attrs["summary"] = "My Summary"
    ds.attrs["keywords"] = "custom, keywords"

    handler = AcddAutoMetadata()
    result = handler.enrich(ds, MetadataRegistry())

    assert result.attrs["title"] == "My Title"
    assert result.attrs["summary"] == "My Summary"
    assert result.attrs["keywords"] == "custom, keywords"
    assert "acdd_autogen_fields" not in result.attrs


def test_title_includes_date_and_point_coords():
    ds = _make_dataset()
    handler = AcddAutoMetadata()
    result = handler.enrich(ds, MetadataRegistry())

    title = result.attrs.get("title", "")
    assert "Level-1 dataset" in title
    assert "2020-01-01" in title
    assert "53.4968N" in title
    assert "8.1815E" in title
    assert "53.4968N, 8.1815E" in title


def test_title_uses_extent_when_coords_arrays():
    ds = _make_dataset()
    ds = ds.assign_coords(
        latitude=("time", [53.40, 53.50, 53.60]),
        longitude=("time", [8.10, 8.20, 8.30]),
    )
    handler = AcddAutoMetadata()
    result = handler.enrich(ds, MetadataRegistry())

    title = result.attrs.get("title", "")
    assert "within" in title
    assert "53.40–53.60" in title
    assert "8.10–8.30" in title


def test_summary_includes_optional_coverage_lines():
    ds = _make_dataset()
    handler = AcddAutoMetadata()
    result = handler.enrich(ds, MetadataRegistry())

    summary = result.attrs.get("summary", "")
    assert "Time coverage" in summary
    assert "Spatial coverage" in summary
    assert "Depth range" in summary


def test_keywords_include_format_and_variable_tokens():
    ds = _make_dataset()
    ds["temperature"].attrs["standard_name"] = "sea_water_temperature"

    handler = AcddAutoMetadata()
    result = handler.enrich(ds, MetadataRegistry())

    keywords = result.attrs.get("keywords", "")
    assert "oceanography" in keywords
    assert "level-1" in keywords
    assert "cnv" in keywords
    assert "temperature" in keywords


def test_keywords_deterministic_order():
    ds = _make_dataset()
    handler = AcddAutoMetadata()
    result1 = handler.enrich(ds, MetadataRegistry())
    result2 = handler.enrich(ds, MetadataRegistry())

    assert result1.attrs.get("keywords") == result2.attrs.get("keywords")
