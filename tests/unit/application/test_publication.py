"""Tests for professional publication validation helpers."""

from __future__ import annotations

import numpy as np

from geowatch.application.publication import _grids_are_spatially_aligned
from geowatch.processing.models import RasterGrid, RasterLayer


def _scene(*, x_origin: float = 100.0) -> RasterLayer:
    """Build a compact scene with a NaN nodata value."""
    grid = RasterGrid(
        crs="EPSG:32642",
        transform=(10.0, 0.0, x_origin, 0.0, -10.0, 200.0),
        width=2,
        height=2,
        band_names=("red",),
        nodata=float("nan"),
    )
    return RasterLayer(
        name="scene",
        data=np.ones((1, 2, 2), dtype=np.float32),
        grid=grid,
    )


def test_spatial_alignment_accepts_matching_nan_nodata_grids() -> None:
    """NaN nodata metadata must not create a false grid mismatch."""
    assert _grids_are_spatially_aligned(_scene(), _scene())


def test_spatial_alignment_rejects_shifted_grid() -> None:
    """A shifted affine origin must fail spatial alignment validation."""
    assert not _grids_are_spatially_aligned(_scene(), _scene(x_origin=110.0))
