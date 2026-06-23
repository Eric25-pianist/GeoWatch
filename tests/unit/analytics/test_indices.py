"""Unit tests for Phase 4 spectral index calculations."""

from __future__ import annotations

import numpy as np

from geowatch.analytics import (
    ANALYTICS_INDEX_NAMES,
    compute_scene_indices,
    compute_spectral_indices,
)
from geowatch.processing.models import RasterLayer


def test_compute_scene_indices_and_spectral_bundle(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> None:
    """All requested spectral indices should be generated consistently."""
    scene_indices = compute_scene_indices(scene_t1)
    spectral_bundle = compute_spectral_indices(scene_t1, scene_t2)

    assert set(scene_indices) == set(ANALYTICS_INDEX_NAMES)
    assert set(spectral_bundle) == set(ANALYTICS_INDEX_NAMES)
    for index_name in ANALYTICS_INDEX_NAMES:
        assert scene_indices[index_name].shape == scene_t1.data.shape[1:]
        result = spectral_bundle[index_name]
        assert result.t1.shape == scene_t1.data.shape[1:]
        assert result.t2.shape == scene_t1.data.shape[1:]
        assert result.difference.shape == scene_t1.data.shape[1:]
        assert result.statistics.t1.valid_pixels == 16
        assert result.statistics.difference.total_pixels == 16
        assert np.isfinite(result.difference).any()
