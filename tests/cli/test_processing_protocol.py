import argparse
import json

import xarray as xr

from seasenselib.cli.commands.data_commands import _write_processing_protocol, ShowCommand


def test_processing_protocol_written(tmp_path):
    output_path = tmp_path / "out.nc.processing.protocol.json"

    args = argparse.Namespace(
        input="input.cnv",
        output="out.nc",
        output_format="netcdf",
        pipeline_profile="default",
        pipeline_file=None,
        pipeline_apply_stages=None,
        pipeline_skip_stages=None,
        pipeline_apply_handlers=None,
        pipeline_skip_handlers=None,
        raw_only=False,
        processing_protocol=str(output_path),
    )

    dataset = xr.Dataset({"temperature": ("time", [1.0, 2.0])})
    metadata = {
        "stages_applied": ["mapping", "finalization"],
        "unit_conversions": ["temperature: degC -> K"],
        "warnings": ["example warning"],
    }

    _write_processing_protocol(output_path, metadata, dataset, args, "convert")

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["command"] == "convert"
    assert payload["input_file"] == "input.cnv"
    assert payload["output_file"] == "out.nc"
    assert payload["pipeline_profile"] == "default"
    assert payload["stages_applied"] == ["mapping", "finalization"]
    assert payload["unit_conversions"] == {"temperature": ["degC -> K"]}
    assert "temperature" in payload["data_variables"]


def test_show_writes_processing_protocol(tmp_path):
    output_path = tmp_path / "input.cnv.processing.protocol.json"

    dataset = xr.Dataset({"temperature": ("time", [1.0, 2.0])})
    metadata = {
        "stages_applied": ["mapping"],
        "warnings": ["example warning"],
    }

    class DummyIO:
        def read_data(self, *args, **kwargs):
            if kwargs.get("return_metadata"):
                return dataset, metadata
            return dataset

    args = argparse.Namespace(
        input="input.cnv",
        input_format=None,
        header_input=None,
        schema="summary",
        mapping=None,
        no_sanitize=False,
        no_fix_coords=False,
        raw_only=False,
        pipeline_profile=None,
        pipeline_file=None,
        pipeline_apply_stages=None,
        pipeline_skip_stages=None,
        pipeline_apply_handlers=None,
        pipeline_skip_handlers=None,
        metadata=None,
        metadata_file=None,
        processing_protocol=str(output_path),
    )

    cmd = ShowCommand(DummyIO())
    result = cmd.execute(args)

    assert result.success
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "show"
    assert payload["input_file"] == "input.cnv"
    assert payload["stages_applied"] == ["mapping"]
