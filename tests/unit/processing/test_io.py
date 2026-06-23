"""Unit tests for raster I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from geowatch.processing.io import read_raster, write_raster
from geowatch.processing.models import RasterGrid, RasterLayer


def test_read_raster_pillow_fallback_reads_geotiff_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeoTIFF metadata should be parsed without Rasterio."""
    raster_path = tmp_path / "sample.tif"
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)
    image = Image.fromarray(array)
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[33550] = (10.0, 10.0, 0.0)
    tags[33922] = (0.0, 0.0, 0.0, 399960.0, 3600000.0, 0.0)
    tags[34735] = (
        1,
        1,
        0,
        1,
        1024,
        0,
        1,
        1,
        1025,
        0,
        1,
        1,
        1026,
        34737,
        7,
        0,
        3072,
        0,
        1,
        32643,
        3076,
        0,
        1,
        9001,
    )
    tags[34737] = "WGS 84 / UTM zone 43N|WGS 84|"
    image.save(raster_path, tiffinfo=tags)

    monkeypatch.setattr("geowatch.processing.io.has_rasterio", lambda: False)
    layer = read_raster(raster_path)

    assert layer.grid.crs == "EPSG:32643"
    assert layer.grid.transform == (10.0, 0.0, 399960.0, 0.0, -10.0, 3600000.0)
    assert layer.data.shape == (1, 4, 4)


def test_integer_cog_overviews_preserve_class_values(tmp_path: Path) -> None:
    """Categorical COG overviews must not invent averaged class values."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.enums import Resampling

    labels = ((np.indices((1024, 1024)).sum(axis=0) % 2) * 7).astype(np.uint8)
    labels[:128, :] = 255
    grid = RasterGrid(
        crs="EPSG:32642",
        transform=(10.0, 0.0, 250000.0, 0.0, -10.0, 2800000.0),
        width=1024,
        height=1024,
        band_names=("lulc",),
        nodata=255,
    )
    layer = RasterLayer(name="lulc", data=labels[np.newaxis, :, :], grid=grid)
    output = write_raster(layer, tmp_path / "lulc.tif", driver="COG")

    with rasterio.open(output) as dataset:
        reduced = dataset.read(
            1,
            out_shape=(64, 64),
            resampling=Resampling.nearest,
        )

    assert set(np.unique(reduced)).issubset({0, 7, 255})
