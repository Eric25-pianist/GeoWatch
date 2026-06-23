"""End-to-end integration tests for Phase 4 analytics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from geowatch.analytics import run_analytics_pipeline
from geowatch.processing.models import RasterGrid, RasterLayer


def _scene(name: str, *, delta: float = 0.0) -> RasterLayer:
    """Build a larger synthetic scene for integration coverage."""
    height = width = 12
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    blue = 0.04 + (0.001 * xx) + (0.001 * yy)
    green = 0.06 + (0.0015 * xx) + (0.001 * yy)
    red = 0.05 + (0.001 * xx) + (0.0015 * yy)
    nir = 0.28 + (0.002 * xx) + (0.002 * yy) + delta
    swir1 = 0.15 + (0.0012 * xx) + (0.001 * yy) + (delta / 2.0)
    swir2 = 0.11 + (0.001 * xx) + (0.0008 * yy)
    data = np.stack([blue, green, red, nir, swir1, swir2], axis=0).astype(np.float32)
    grid = RasterGrid(
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, float(height)),
        width=width,
        height=height,
        band_names=("blue", "green", "red", "nir", "swir1", "swir2"),
        nodata=-9999.0,
    )
    return RasterLayer(name=name, data=data, grid=grid)


def test_end_to_end_analytics_pipeline(tmp_path: Path) -> None:
    """The analytics pipeline should complete on a larger synthetic scene."""
    scene_t1 = _scene("integration_t1")
    scene_t2 = _scene("integration_t2", delta=0.08)
    training_labels: NDArray[np.int64] = np.tile(
        np.array(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1],
                [0, 2, 2, 1, 2, 2, 3, 3, 0, 2, 1, 1],
                [3, 2, 2, 1, 2, 2, 3, 3, 3, 2, 2, 1],
                [3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 1],
            ],
            dtype=np.int64,
        ),
        (3, 1),
    ).astype(np.int64, copy=False)
    reference_labels: NDArray[np.int64] = training_labels.copy()

    report = run_analytics_pipeline(
        scene_t1,
        scene_t2,
        output_root=tmp_path,
        classification_method="random_forest",
        training_labels_t1=training_labels,
        training_labels_t2=training_labels,
        reference_labels_t1=reference_labels,
        reference_labels_t2=reference_labels,
    )

    assert report.phase == 4
    assert report.change_results["mad"].threshold is not None
    assert report.artifacts["change_statistics"].exists()
    assert report.artifacts["classification_statistics"].exists()
    assert report.artifacts["analytics_report"].exists()
