import xarray as xr

from seasenselib.pipeline.base import StageContext
from seasenselib.pipeline.mapping.handlers.dict_mapping_strategy import DictMappingStrategy
from seasenselib.pipeline.mapping.handlers.regex_mapping_strategy import RegexMappingStrategy
from seasenselib.pipeline.mapping.handlers.user_mapping_strategy import UserMappingStrategy
from seasenselib.pipeline.mapping.handlers.strategy_runner import MappingStrategyRunner
from seasenselib.pipeline.mapping.handlers.mapping_runner import MappingRunner


def test_dict_mapping_strategy_case_insensitive():
    strategy = DictMappingStrategy({"temperature": ["t090C", "TEMP"]})
    assert strategy.map("t090c") == "temperature"
    assert strategy.map("TEMP") == "temperature"
    assert strategy.map("unknown") is None


def test_user_mapping_strategy_case_insensitive():
    strategy = UserMappingStrategy({"MyTemp": "temperature"})
    assert strategy.map("mytemp") == "temperature"
    assert strategy.map("MYTEMP") == "temperature"
    assert strategy.map("other") is None


def test_regex_mapping_strategy_basic():
    strategy = RegexMappingStrategy([(r"^temp\d+$", "temperature")])
    assert strategy.map("TEMP1") == "temperature"
    assert strategy.map("salinity") is None


def test_mapping_strategy_runner_order_and_empty():
    first = UserMappingStrategy({"a": "one"})
    second = UserMappingStrategy({"a": "two"})
    runner = MappingStrategyRunner([first, second])
    assert runner.map("a") == "one"
    assert "strategies" in runner.describe()

    empty = MappingStrategyRunner()
    assert empty.map("a") is None
    assert "no strategies" in empty.describe().lower()


def test_mapping_runner_smart_numbering_and_preserve_original():
    ds = xr.Dataset({
        "tempA": (["time"], [1.0, 2.0]),
        "tempB": (["time"], [3.0, 4.0]),
    })
    runner = MappingRunner(
        custom_mappings={"tempA": "temperature", "tempB": "temperature"},
        use_default_mappings=False,
        use_reader_mappings=False,
        use_regex=False,
        preserve_original=True,
    )
    context = StageContext(ds, {})
    result = runner.process(context)

    assert "temperature_1" in result.dataset
    assert "temperature_2" in result.dataset
    assert "tempA" not in result.dataset
    assert result.dataset["temperature_1"].attrs.get("original_name") == "tempA"
    assert result.dataset["temperature_2"].attrs.get("original_name") == "tempB"


def test_mapping_runner_collision_with_existing_canonical():
    ds = xr.Dataset({
        "temperature": (["time"], [1.0, 2.0]),
        "tempA": (["time"], [3.0, 4.0]),
    })
    runner = MappingRunner(
        custom_mappings={"tempA": "temperature"},
        use_default_mappings=False,
        use_reader_mappings=False,
        use_regex=False,
        preserve_original=True,
    )
    result = runner.process(StageContext(ds, {}))

    assert "temperature" not in result.dataset
    assert "temperature_1" in result.dataset
    assert "temperature_2" in result.dataset
    assert result.dataset["temperature_1"].attrs.get("original_name") == "temperature"
    assert result.dataset["temperature_2"].attrs.get("original_name") == "tempA"


def test_mapping_runner_no_mappings_leaves_dataset():
    ds = xr.Dataset({"tempA": (["time"], [1.0, 2.0])})
    runner = MappingRunner(
        use_custom_mappings=False,
        use_default_mappings=False,
        use_reader_mappings=False,
        use_regex=False,
    )
    context = StageContext(ds, {})
    result = runner.process(context)

    assert list(result.dataset.data_vars) == ["tempA"]
    assert "variable_mappings" not in result.metadata
