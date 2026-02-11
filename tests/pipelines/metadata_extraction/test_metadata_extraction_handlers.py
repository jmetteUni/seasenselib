import xarray as xr

from seasenselib.pipeline.base import StageContext
from seasenselib.pipeline.metadata_extraction.handlers.attribute_extractor import AttributeMetadataExtractor
from seasenselib.pipeline.metadata_extraction.handlers.global_attribute_extractor import GlobalAttributeMetadataExtractor
from seasenselib.pipeline.metadata_extraction.handlers.extraction_runner import MetadataExtractionRunner


def test_attribute_metadata_extractor_positive_and_coords():
    ds = xr.Dataset({"temp": (["time"], [1.0, 2.0])})
    ds["temp"].attrs.update({
        "units": "degC",
        "long_name": "Temperature",
        "standard_name": "sea_water_temperature",
        "valid_min": 0,
        "valid_max": 40,
        "_FillValue": -9999,
        "comment": "test",
    })
    ds = ds.assign_coords(depth=("time", [1.0, 2.0]))
    ds["depth"].attrs["units"] = "m"

    extractor = AttributeMetadataExtractor()
    registry = extractor.extract(ds)

    assert registry.get("temp.units") == "degC"
    assert registry.get("temp.long_name") == "Temperature"
    assert registry.get("temp.standard_name") == "sea_water_temperature"
    assert registry.get("temp.valid_min") == 0
    assert registry.get("temp.valid_max") == 40
    assert registry.get("temp._FillValue") == -9999
    assert registry.get("temp.comment") == "test"
    assert registry.get("depth.units") == "m"


def test_attribute_metadata_extractor_non_dataset():
    extractor = AttributeMetadataExtractor()
    registry = extractor.extract("not-a-dataset")
    assert registry.to_dict() == {}


def test_global_attribute_metadata_extractor_keys():
    ds = xr.Dataset({"temp": (["time"], [1.0, 2.0])})
    ds.attrs["title"] = "My Dataset"
    ds.attrs["instrument"] = "CTD"

    extractor = GlobalAttributeMetadataExtractor()
    registry = extractor.extract(ds)

    assert registry.get("global.title") == "My Dataset"
    assert registry.get("acdd.title") == "My Dataset"
    assert registry.get("instrument.instrument") == "CTD"


def test_metadata_extraction_runner_merges():
    ds = xr.Dataset({"temp": (["time"], [1.0, 2.0])})
    ds["temp"].attrs["units"] = "degC"
    ds.attrs["title"] = "Test"

    runner = MetadataExtractionRunner()
    context = StageContext(ds, {})
    result = runner.process(context)

    registry = result.metadata.get("_metadata_registry")
    assert registry is not None
    assert registry.get("temp.units") == "degC"
    assert registry.get("global.title") == "Test"
