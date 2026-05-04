import xarray as xr

from seasenselib.pipeline.base import StageContext
from seasenselib.pipeline.finalization.handlers.global_attributes import GlobalAttributes
from seasenselib.pipeline.finalization.handlers.processor_metadata import ProcessorMetadata
from seasenselib.pipeline.finalization.handlers.raw_metadata import RawMetadata
from seasenselib.pipeline.finalization.handlers.sorting import Sorting


def test_global_attributes_adds_basic_fields():
    ds = xr.Dataset({"temperature": (["time"], [1.0, 2.0])})
    context = StageContext(ds, {"source_file": "test.cnv"})
    handler = GlobalAttributes()

    result = handler.process(context)

    assert "history" in result.dataset.attrs
    assert "date_created" in result.dataset.attrs
    assert "date_modified" in result.dataset.attrs
    assert "processor" not in result.dataset.attrs
    assert "source" not in result.dataset.attrs


def test_global_attributes_preserves_history():
    ds = xr.Dataset({"temperature": (["time"], [1.0, 2.0])})
    ds.attrs["history"] = "old"
    context = StageContext(ds, {})
    handler = GlobalAttributes()

    result = handler.process(context)
    assert result.dataset.attrs["history"].endswith("old")


def test_global_attributes_disable_history():
    ds = xr.Dataset({"temperature": (["time"], [1.0, 2.0])})
    context = StageContext(ds, {})
    handler = GlobalAttributes(add_history=False)

    result = handler.process(context)
    assert "history" not in result.dataset.attrs
    assert "processor" not in result.dataset.attrs


def test_raw_metadata_moves_prefixes_and_sets_fields(tmp_path):
    file_path = tmp_path / "test.cnv"
    file_path.write_text("* test\n*END*\n", encoding="utf-8")

    ds = xr.Dataset({"temperature": (["time"], [1.0])})
    ds.attrs["cnv_sbe_model"] = "SBE 9"
    ds["temperature"].attrs["cnv_original_name"] = "t090C"
    ds.attrs["platform"] = "TestPlatform"

    context = StageContext(
        ds,
        {
            "source_file": str(file_path),
            "format_key": "sbe-cnv",
            "raw_header": "* test\n*END*",
        },
    )
    handler = RawMetadata()
    result = handler.process(context)

    assert "cnv_sbe_model" not in result.dataset.attrs
    assert result.dataset.attrs["platform"] == "TestPlatform"
    assert result.dataset.attrs["raw_filename"] == "test.cnv"
    assert "raw_sha256" in result.dataset.attrs
    assert "raw_metadata" in result.dataset.attrs
    assert "raw_metadata_schema" in result.dataset.attrs
    assert result.dataset["temperature"].attrs["cnv_original_name"] == "t090C"

    raw_meta = result.dataset.attrs["raw_metadata"]
    payload = __import__("json").loads(raw_meta)
    assert payload["blocks"]["other"]["global_attributes"]["cnv_sbe_model"] == "SBE 9"


def test_processor_metadata_sets_defaults():
    ds = xr.Dataset({"temperature": (["time"], [1.0])})
    context = StageContext(
        ds,
        {
            "reader_module": "seasenselib.readers.sbe_cnv_reader",
            "format_name": "SeaBird CNV Reader",
            "format_key": "sbe-cnv",
            "reader_class": "SbeCnvReader",
        },
    )
    handler = ProcessorMetadata()
    result = handler.process(context)

    assert result.dataset.attrs["processor_name"] == "SeaSenseLib"
    assert result.dataset.attrs["processor_level"] == "L1"
    assert result.dataset.attrs["processing_level"] == "L1"
    assert result.dataset.attrs["processor_module"] == "seasenselib.readers.sbe_cnv_reader"
    assert result.dataset.attrs["processor_module_key"] == "sbe-cnv"
    assert "processor_execution_time_utc" in result.dataset.attrs


def test_sorting_handler_orders_and_preserves_attrs():
    ds = xr.Dataset(
        {
            "b": (["time"], [1.0, 2.0]),
            "a": (["time"], [3.0, 4.0]),
        },
        coords={"time": [0, 1], "z": ("time", [10.0, 11.0])},
        attrs={"title": "test"},
    )
    context = StageContext(ds, {})
    sorter = Sorting()

    result = sorter.process(context)

    assert list(result.dataset.data_vars) == ["a", "b"]
    assert result.dataset.attrs["title"] == "test"
