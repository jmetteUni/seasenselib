from seasenselib.pipeline.config import PipelineConfig


def test_pipeline_profiles_have_descriptions():
    for name in ("minimal", "default", "full"):
        cfg = PipelineConfig.from_resource(name)
        description = cfg.global_config.get("description")
        assert description, f"Profile '{name}' missing description"


def test_default_profile_unit_handling_is_normalize_only():
    cfg = PipelineConfig.from_resource("default")
    stage = next((s for s in cfg.pipeline if s.name == "unit_handling"), None)
    assert stage is not None
    assert stage.config.get("handlers") == ["normalize"]


def test_full_profile_includes_all_stages():
    cfg = PipelineConfig.from_resource("full")
    stage_names = [stage.name for stage in cfg.pipeline]
    assert stage_names == [
        "mapping",
        "unit_handling",
        "derivation",
        "metadata_extraction",
        "metadata_enrichment",
        "validation",
        "finalization",
    ]
