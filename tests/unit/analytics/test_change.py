"""Unit tests for Phase 4 change detection."""

from __future__ import annotations

from geowatch.analytics import detect_change_suite
from geowatch.analytics.change import DEFAULT_CHANGE_METHODS
from geowatch.processing.models import RasterLayer


def test_detect_change_suite(scene_t1: RasterLayer, scene_t2: RasterLayer) -> None:
    """Every supported change detector should yield a result."""
    results = detect_change_suite(scene_t1, scene_t2)

    assert set(results) == set(DEFAULT_CHANGE_METHODS)
    for result in results.values():
        assert result.score.shape == scene_t1.data.shape[1:]
        assert result.threshold is not None
        assert result.threshold.mask.shape == scene_t1.data.shape[1:]
        assert result.metadata["threshold_method"] == "otsu"
