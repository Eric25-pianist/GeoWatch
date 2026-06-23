"""Unit tests for foundation pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from geowatch.config.loader import load_config
from geowatch.pipelines.foundation import run_foundation_pipeline, write_map_readiness


def test_foundation_pipeline_writes_manifest(
    tmp_path: Path,
    sample_config_path: Path,
) -> None:
    """The foundation pipeline should write a readiness manifest."""
    config = load_config(sample_config_path)
    config.outputs.manifests = tmp_path / "manifests"

    manifest = run_foundation_pipeline(config, sample_config_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["phase"] == 1
    assert payload["status"] == "validated"


def test_map_readiness_writes_artifact(
    tmp_path: Path,
    sample_config_path: Path,
) -> None:
    """The map command support function should write readiness metadata."""
    config = load_config(sample_config_path)
    config.outputs.maps = tmp_path / "maps"

    artifact = write_map_readiness(config, sample_config_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["map_generation_phase"] == 5
