"""Sensor policy, QA masks, and resume manifest tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geowatch.application.manifest import (
    RunManifest,
    load_or_create_manifest,
    save_manifest,
)
from geowatch.application.sensors import (
    LANDSAT_7,
    landsat_cloud_mask,
    required_assets,
    select_common_sensor,
    sentinel_cloud_mask,
)
from geowatch.core.errors import ConfigurationError


def test_historical_and_modern_sensor_selection() -> None:
    """Automatic selection should cover historical and modern comparisons."""
    assert select_common_sensor(2009, 2012).dataset == "landsat-7-c2-l2"
    assert select_common_sensor(2018, 2020).dataset == "sentinel-2-l2a"
    with pytest.raises(ConfigurationError, match="Sentinel-2"):
        select_common_sensor(2009, 2012, "sentinel-2")


def test_product_quality_masks() -> None:
    """Landsat QA bits and Sentinel SCL classes should mask invalid pixels."""
    landsat = np.array([[0, 1 << 3, 1 << 4]], dtype=np.uint16)
    sentinel = np.array([[4, 8, 11]], dtype=np.uint8)

    assert landsat_cloud_mask(landsat).tolist() == [[False, True, True]]
    assert sentinel_cloud_mask(sentinel).tolist() == [[False, True, True]]
    assert "qa_radsat" in required_assets(LANDSAT_7)


def test_manifest_only_resumes_existing_artifacts(tmp_path: Path) -> None:
    """A completed stage is reusable only while its artifacts exist."""
    project = tmp_path / "project.yaml"
    artifact = tmp_path / "catalog.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = RunManifest(project_file=project)
    manifest.start("acquisition:2020")
    manifest.complete("acquisition:2020", artifact)
    save_manifest(manifest, tmp_path / "run_manifest.json")

    loaded = load_or_create_manifest(tmp_path / "run_manifest.json", project)
    assert loaded.is_complete("acquisition:2020")
    artifact.unlink()
    assert not loaded.is_complete("acquisition:2020")
