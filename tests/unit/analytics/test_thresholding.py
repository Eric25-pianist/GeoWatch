"""Unit tests for Phase 4 thresholding helpers."""

from __future__ import annotations

import numpy as np

from geowatch.analytics import apply_threshold


def test_threshold_variants() -> None:
    """Thresholding modes should produce masks and metadata."""
    score = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [0.70, 0.80, 0.90],
        ],
        dtype=np.float32,
    )

    otsu = apply_threshold(score, method="otsu")
    percentile = apply_threshold(score, method="percentile", percentile=50.0)
    manual = apply_threshold(score, method="manual", manual_threshold=0.55)
    adaptive = apply_threshold(score, method="adaptive", window_size=3, offset=0.0)

    assert otsu.mask.shape == score.shape
    assert percentile.changed_pixels == 4
    assert manual.changed_pixels == 4
    assert adaptive.mask.shape == score.shape
    assert isinstance(adaptive.threshold, np.ndarray)
    assert adaptive.threshold.shape == score.shape
