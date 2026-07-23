from __future__ import annotations

import json

import numpy as np
import xarray as xr

from seasenselib.pipeline.base import StageContext
from seasenselib.pipeline.config import PipelineConfig
from seasenselib.pipeline.factory import create_pipeline
from seasenselib.pipeline.finalization.handlers.global_attributes import GlobalAttributes
from seasenselib.pipeline.finalization.handlers.processor_metadata import ProcessorMetadata
from seasenselib.pipeline.interfaces import ITransformation, TransformationRecord
from seasenselib.pipeline.transformation import TransformationStage


class ScaleTemperature(ITransformation):
    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def name(self) -> str:
        return "scale_temperature"

    def can_transform(self, dataset: xr.Dataset, context=None) -> bool:
        return "temperature" in dataset

    def transform(self, dataset: xr.Dataset, context=None):
        ds = dataset.copy()
        ds["temperature"] = ds["temperature"] * self.factor
        ds["temperature"].attrs.update(dataset["temperature"].attrs)
        return ds, [
            TransformationRecord(
                transformation="scale_temperature",
                description="Scale temperature by a constant factor.",
                variables=["temperature"],
                parameters={"factor": self.factor},
            )
        ]


def test_transformation_stage_noops_without_reader_transformations():
    ds = xr.Dataset({"temperature": ("time", np.array([1.0, 2.0]))})
    context = StageContext(ds, {"format_key": "sbe-cnv"})

    result = TransformationStage().process(context)

    xr.testing.assert_identical(result.dataset, ds)
    assert "transformations" not in result.metadata
    assert "handlers_applied" not in result.metadata
    assert "processor_transformations" not in result.dataset.attrs


def test_transformation_stage_can_be_empty_explicitly():
    ds = xr.Dataset({"temperature": ("time", np.array([1.0, 2.0]))})
    context = StageContext(
        ds,
        {
            "format_key": "nortek-ascii",
            "reader_transformations": [ScaleTemperature(3.0)],
        },
    )

    stage = TransformationStage(transformations=[])
    result = stage.process(context)

    xr.testing.assert_identical(result.dataset, ds)
    assert "transformations" not in result.metadata


def test_transformation_stage_applies_reader_provided_transformations():
    ds = xr.Dataset(
        {"temperature": ("time", np.array([1.0, 2.0]), {"units": "degree_C"})}
    )
    context = StageContext(
        ds,
        {
            "format_key": "nortek-ascii",
            "reader_transformations": [ScaleTemperature(3.0)],
        },
    )

    result = TransformationStage().process(context)

    np.testing.assert_allclose(result.dataset["temperature"].values, [3.0, 6.0])
    assert result.metadata["handlers_applied"] == ["transformation:reader"]
    assert result.metadata["transformations"][0]["handler"] == "reader"
    assert result.metadata["transformations"][0]["transformation"] == (
        "scale_temperature"
    )
    assert result.metadata["transformations"][0]["parameters"] == {"factor": 3.0}

    variable_records = json.loads(
        result.dataset["temperature"].attrs["processing_transformations"]
    )
    assert variable_records[0]["transformation"] == "scale_temperature"
    assert variable_records[0]["parameters"] == {"factor": 3.0}
    assert result.dataset.attrs["processor_transformations_count"] == 1


def test_transformation_stage_honors_reader_group_filters():
    ds = xr.Dataset({"temperature": ("time", np.array([1.0, 2.0]))})
    stage = TransformationStage()
    stage.configure({"enabled_reader_groups": ["nortek"]})

    skipped = stage.process(
        StageContext(
            ds,
            {
                "format_key": "sbe-cnv",
                "reader_transformations": [ScaleTemperature()],
            },
        )
    )
    assert "transformations" not in skipped.metadata

    transformed = stage.process(
        StageContext(
            ds,
            {
                "format_key": "nortek-csv",
                "reader_transformations": [ScaleTemperature()],
            },
        )
    )
    np.testing.assert_allclose(transformed.dataset["temperature"].values, [2.0, 4.0])


def test_transformation_stage_can_run_explicit_handler():
    ds = xr.Dataset({"temperature": ("time", np.array([1.0, 2.0]))})
    stage = TransformationStage(transformations=[ScaleTemperature(4.0)])

    result = stage.process(StageContext(ds, {"format_key": "generic"}))

    np.testing.assert_allclose(result.dataset["temperature"].values, [4.0, 8.0])
    assert result.metadata["handlers_applied"] == [
        "transformation:scale_temperature"
    ]
    assert result.metadata["transformations"][0]["handler"] == "scale_temperature"


def test_transformation_metadata_is_available_to_finalization_handlers():
    ds = xr.Dataset({"temperature": ("time", np.array([1.0, 2.0]))})
    metadata = {
        "reader_class": "DummyReader",
        "format_name": "Dummy",
        "source_file": "input.txt",
        "stages_applied": ["mapping", "unit_handling", "transformation"],
        "transformations": [
            {
                "handler": "dummy",
                "transformation": "dummy_transform",
                "description": "Example transformation.",
                "variables": ["temperature"],
            }
        ],
    }
    context = ProcessorMetadata().process(StageContext(ds, metadata))
    context = GlobalAttributes().process(context)

    assert context.dataset.attrs["processor_transformations_count"] == 1
    transformations = json.loads(context.dataset.attrs["processor_transformations"])
    assert transformations[0]["transformation"] == "dummy_transform"
    assert "Transformed: 1 step(s)" in context.dataset.attrs["history"]


def test_configured_pipeline_runs_transformation_before_validation():
    config = PipelineConfig()
    config.add_stage("unit_handling")
    config.add_stage("transformation")
    config.add_stage("validation", config={"validators": ["unit"]})

    pipeline = create_pipeline(config=config)

    assert pipeline.get_stage_order() == [
        "unit_handling",
        "transformation",
        "validation",
    ]
