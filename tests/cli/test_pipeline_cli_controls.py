import argparse

import pytest

from seasenselib.cli.commands.data_commands import (
    _build_reader_kwargs,
    _build_stage_kwargs,
)
from seasenselib.cli.parser import ArgumentParser
from seasenselib.core.exceptions import ValidationError
from seasenselib.pipeline.config import PipelineConfig


def _base_args(**overrides):
    defaults = dict(
        raw_only=False,
        pipeline_profile=None,
        pipeline_file=None,
        pipeline_apply_stages=None,
        pipeline_skip_stages=None,
        pipeline_apply_handlers=None,
        pipeline_skip_handlers=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_stage_kwargs_profile():
    args = _base_args(pipeline_profile="default")
    stage_kwargs = _build_stage_kwargs(args)
    assert "pipeline_config" in stage_kwargs
    assert isinstance(stage_kwargs["pipeline_config"], PipelineConfig)
    assert stage_kwargs["pipeline_config"].global_config.get("profile") == "default"


def test_build_stage_kwargs_apply_stages():
    args = _base_args(pipeline_apply_stages="mapping,validation")
    stage_kwargs = _build_stage_kwargs(args)
    cfg = stage_kwargs["pipeline_config"]
    names = [stage.name for stage in cfg.pipeline]
    assert names == ["mapping", "validation"]


def test_build_stage_kwargs_skip_stages():
    args = _base_args(pipeline_skip_stages="validation")
    stage_kwargs = _build_stage_kwargs(args)
    cfg = stage_kwargs["pipeline_config"]
    names = [stage.name for stage in cfg.pipeline]
    assert names == [
        "mapping",
        "unit_handling",
        "transformation",
        "derivation",
        "metadata_extraction",
        "metadata_enrichment",
        "finalization",
    ]


def test_build_stage_kwargs_apply_handlers():
    args = _base_args(pipeline_apply_handlers="metadata_enrichment:cf")
    stage_kwargs = _build_stage_kwargs(args)
    cfg = stage_kwargs["pipeline_config"]
    stage = next(s for s in cfg.pipeline if s.name == "metadata_enrichment")
    assert stage.config.get("handlers") == ["cf"]


def test_build_stage_kwargs_raw_only_conflicts():
    args = _base_args(raw_only=True, pipeline_profile="default")
    with pytest.raises(ValueError):
        _build_stage_kwargs(args)


def test_build_reader_kwargs_parses_reader_args():
    args = argparse.Namespace(
        no_sanitize=True,
        no_fix_coords=True,
        reader_args=[
            "latitude=30.5",
            "round_digits=10",
            "strict=false",
            "label=MAPR",
        ],
    )

    reader_kwargs = _build_reader_kwargs(args)

    assert reader_kwargs == {
        "sanitize_input": False,
        "fix_missing_coords": False,
        "latitude": 30.5,
        "round_digits": 10,
        "strict": False,
        "label": "MAPR",
    }


def test_build_reader_kwargs_rejects_invalid_reader_arg():
    args = argparse.Namespace(
        no_sanitize=False,
        no_fix_coords=False,
        reader_args=["round_digits"],
    )

    with pytest.raises(ValidationError):
        _build_reader_kwargs(args)


def test_convert_parser_accepts_reader_arg():
    parser = ArgumentParser().create_command_parser("convert", lightweight=True)

    args = parser.parse_args([
        "-i", "file.LOG",
        "-f", "mapr-log",
        "-o", "out.nc",
        "-F", "netcdf",
        "--reader-arg", "latitude=30.0",
        "--reader-arg", "round_digits=10",
    ])

    assert args.reader_args == ["latitude=30.0", "round_digits=10"]
