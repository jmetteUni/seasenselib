import pytest

from seasenselib.pipeline.config import PipelineConfig
from seasenselib.pipeline.utils import parse_handler_selectors, apply_handler_filters


def test_parse_handler_selectors_string():
    selectors = parse_handler_selectors("metadata_enrichment:cf,validation:unit")
    assert selectors == {
        "metadata_enrichment": ["cf"],
        "validation": ["unit"],
    }


def test_apply_handler_filters_apply_and_skip():
    cfg = PipelineConfig()
    cfg.add_stage("metadata_enrichment", config={"handlers": ["cf", "acdd", "acdd_auto"]})
    cfg.add_stage("validation", config={"validators": ["cf", "unit"]})

    apply_map = parse_handler_selectors("metadata_enrichment:cf")
    skip_map = parse_handler_selectors("validation:unit")

    filtered = apply_handler_filters(cfg, apply_map, skip_map)

    stage_meta = next(stage for stage in filtered.pipeline if stage.name == "metadata_enrichment")
    stage_val = next(stage for stage in filtered.pipeline if stage.name == "validation")

    assert stage_meta.config["handlers"] == ["cf"]
    assert stage_val.config["validators"] == ["cf"]
