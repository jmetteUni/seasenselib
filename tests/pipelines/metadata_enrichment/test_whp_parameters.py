import xarray as xr

from seasenselib.pipeline.metadata_enrichment.handlers.whp_parameters import WHPParameters
from seasenselib.pipeline.interfaces import MetadataRegistry


def test_whp_parameters_sets_codes_for_known_variables():
    ds = xr.Dataset(
        {
            "temperature": ("time", [1.0, 2.0]),
            "pressure": ("time", [10.0, 11.0]),
            "salinity": ("time", [33.0, 34.0]),
            "conductivity": ("time", [3.0, 3.1]),
        }
    )
    handler = WHPParameters()
    enriched = handler.enrich(ds, MetadataRegistry())

    assert enriched["temperature"].attrs.get("whp_parameter") == "CTDTMP"
    assert enriched["pressure"].attrs.get("whp_parameter") == "CTDPRS"
    assert enriched["salinity"].attrs.get("whp_parameter") == "CTDSAL"
    assert enriched["conductivity"].attrs.get("whp_parameter") == "CTDCOND"


def test_whp_parameters_respects_existing_attribute_and_numbered_names():
    ds = xr.Dataset(
        {
            "temperature_1": ("time", [1.0, 2.0]),
            "temperature_2": ("time", [2.0, 3.0]),
        }
    )
    ds["temperature_1"].attrs["whp_parameter"] = "CUSTOM"

    handler = WHPParameters()
    enriched = handler.enrich(ds, MetadataRegistry())

    assert enriched["temperature_1"].attrs.get("whp_parameter") == "CUSTOM"
    assert enriched["temperature_2"].attrs.get("whp_parameter") == "CTDTMP"
