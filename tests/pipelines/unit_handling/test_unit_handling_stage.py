import xarray as xr

from seasenselib.pipeline import StageContext
from seasenselib.pipeline.unit_handling import UnitHandlingStage
import seasenselib.parameters as params


def test_unit_handling_stage_normalizes_units():
    stage = UnitHandlingStage()
    ds = xr.Dataset({params.TEMPERATURE: (["time"], [10, 11, 12])})
    ds[params.TEMPERATURE].attrs["units"] = "degC"

    result = stage.process(StageContext(ds, {}))
    assert result.dataset[params.TEMPERATURE].attrs["units"] == "degC"
