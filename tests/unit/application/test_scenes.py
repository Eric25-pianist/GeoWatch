"""Scene compositing tests."""

from __future__ import annotations

import numpy as np

from geowatch.application.scenes import _combine_scene_data
from geowatch.processing.models import RasterGrid, RasterLayer


def _layer(data: np.ndarray) -> RasterLayer:
    """Create a compact raster layer for compositing tests."""
    _, rows, cols = data.shape
    grid = RasterGrid(
        crs="EPSG:32654",
        transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        width=cols,
        height=rows,
        band_names=("red", "nir"),
        nodata=np.nan,
    )
    return RasterLayer(name="scene", data=data.astype(np.float32), grid=grid)


def test_chunked_median_matches_numpy_reference() -> None:
    """Chunked median should match a direct nanmedian on small arrays."""
    scenes = [
        _layer(
            np.array(
                [
                    [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]],
                    [[10.0, 11.0, np.nan], [13.0, 14.0, 15.0]],
                ]
            )
        ),
        _layer(
            np.array(
                [
                    [[2.0, 2.0, 4.0], [np.nan, 7.0, 8.0]],
                    [[12.0, np.nan, 16.0], [18.0, 20.0, 22.0]],
                ]
            )
        ),
    ]
    expected = np.nanmedian(np.stack([scene.data for scene in scenes]), axis=0)

    result = _combine_scene_data(scenes, method="median", chunk_size=1)

    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_chunked_mean_and_first_valid() -> None:
    """Chunked mean and first-valid compositing should handle nodata correctly."""
    scenes = [
        _layer(np.array([[[1.0, np.nan], [np.nan, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]])),
        _layer(np.array([[[3.0, 2.0], [9.0, np.nan]], [[7.0, np.nan], [11.0, 12.0]]])),
    ]

    mean = _combine_scene_data(scenes, method="mean", chunk_size=1)
    first = _combine_scene_data(scenes, method="first", chunk_size=1)

    np.testing.assert_allclose(
        mean,
        np.array([[[2.0, 2.0], [9.0, 4.0]], [[6.0, 6.0], [11.0, 10.0]]]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first,
        np.array([[[1.0, 2.0], [9.0, 4.0]], [[5.0, 6.0], [11.0, 8.0]]]),
        equal_nan=True,
    )
