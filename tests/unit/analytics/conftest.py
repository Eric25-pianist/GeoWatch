"""Shared fixtures for GeoWatch analytics unit tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from geowatch.processing.models import RasterGrid, RasterLayer

_BAND_NAMES: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
)


def _build_grid(width: int = 4, height: int = 4) -> RasterGrid:
    """Create a canonical raster grid for analytics tests."""
    return RasterGrid(
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, float(height)),
        width=width,
        height=height,
        band_names=_BAND_NAMES,
        nodata=-9999.0,
    )


def _build_scene(name: str, *, change: bool = False) -> RasterLayer:
    """Create a synthetic multispectral scene for test coverage."""
    yy, xx = np.mgrid[0:4, 0:4].astype(np.float32)
    blue = 0.05 + (0.01 * xx) + (0.002 * yy)
    green = 0.07 + (0.01 * xx) + (0.003 * yy)
    red = 0.04 + (0.008 * xx) + (0.002 * yy)
    nir = 0.30 + (0.02 * xx) + (0.01 * yy)
    swir1 = 0.16 + (0.01 * xx) + (0.006 * yy)
    swir2 = 0.12 + (0.009 * xx) + (0.005 * yy)
    if change:
        red = red.copy()
        nir = nir.copy()
        swir1 = swir1.copy()
        red[0, 0] += 0.04
        nir[1:, 1:] += 0.07
        swir1[2:, 2:] += 0.05
    data = np.stack([blue, green, red, nir, swir1, swir2], axis=0).astype(np.float32)
    return RasterLayer(name=name, data=data, grid=_build_grid())


@pytest.fixture
def scene_t1() -> RasterLayer:
    """Return the baseline synthetic scene."""
    return _build_scene("scene_t1")


@pytest.fixture
def scene_t2() -> RasterLayer:
    """Return a perturbed synthetic scene."""
    return _build_scene("scene_t2", change=True)


@pytest.fixture
def training_labels() -> NDArray[np.int64]:
    """Return integer training labels aligned to the synthetic scene."""
    return np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [3, 2, 2, 1],
            [3, 3, 3, 1],
        ],
        dtype=np.int64,
    )


@pytest.fixture
def reference_labels() -> NDArray[np.int64]:
    """Return a reference label map aligned to the synthetic scene."""
    return np.array(
        [
            [0, 1, 2, 4],
            [0, 2, 2, 1],
            [3, 2, 2, 1],
            [3, 4, 4, 1],
        ],
        dtype=np.int64,
    )
