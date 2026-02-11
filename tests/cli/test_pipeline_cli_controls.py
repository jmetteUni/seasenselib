import argparse

import pytest

from seasenselib.cli.commands.data_commands import _build_stage_kwargs
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
    assert "validation" not in names


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
