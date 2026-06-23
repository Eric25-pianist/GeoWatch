"""Unit tests for project initialization."""

from __future__ import annotations

from pathlib import Path

from geowatch.config.loader import load_config
from geowatch.core.initializer import initialize_project


def test_initialize_project_creates_required_assets(tmp_path: Path) -> None:
    """Initialization should create folders, config, schema, and sample AOI."""
    created = initialize_project(tmp_path, overwrite=True)

    assert created
    assert (tmp_path / "configs" / "default.yaml").exists()
    assert (tmp_path / "configs" / "schemas" / "pipeline.schema.json").exists()
    sample_aoi = tmp_path / "tests" / "fixtures" / "sample_data" / "sample_aoi.geojson"
    assert sample_aoi.exists()
    assert load_config(tmp_path / "configs" / "default.yaml").project_name
