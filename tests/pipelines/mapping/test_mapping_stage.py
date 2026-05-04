import xarray as xr

from seasenselib.pipeline import StageContext
from seasenselib.pipeline.mapping import MappingStage
import seasenselib.parameters as params


def test_mapping_stage_basic():
    stage = MappingStage()
    ds = xr.Dataset({
        "t090C": (["time"], [10, 11, 12]),
        "sal00": (["time"], [35, 35.1, 35.2]),
    })
    result = stage.process(StageContext(ds, {}))

    assert params.TEMPERATURE in result.dataset
    assert params.SALINITY in result.dataset


def test_mapping_stage_preserve_original():
    stage = MappingStage()
    stage.configure({"preserve_original": True})
    ds = xr.Dataset({"t090C": (["time"], [10, 11, 12])})

    result = stage.process(StageContext(ds, {}))
    assert result.dataset[params.TEMPERATURE].attrs.get("original_name") == "t090C"
