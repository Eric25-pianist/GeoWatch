"""Integration tests for large-AOI chunked processing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from geowatch.config.models import AOIConfig, DateRangeConfig, ProjectConfig
from geowatch.processing.engine import process_in_chunks, run_raster_processing
from geowatch.processing.models import RasterGrid, RasterLayer


def _large_layer(name: str, fill: float) -> RasterLayer:
    data = np.full((2, 64, 64), fill, dtype=np.float32)
    grid = RasterGrid(
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 64.0),
        width=64,
        height=64,
        band_names=("red", "green"),
        nodata=-9999.0,
    )
    return RasterLayer(name=name, data=data, grid=grid)


def test_large_aoi_chunking_and_pipeline(tmp_path: Path) -> None:
    """Large rasters should process successfully with chunked execution."""
    config = ProjectConfig(
        project_name="large-aoi",
        aoi=AOIConfig(kind="bbox", bbox=(0.0, 0.0, 10.0, 10.0)),
        dates=DateRangeConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        ),
    )
    config.outputs.root = tmp_path / "outputs"
    config.outputs.rasters = tmp_path / "outputs" / "rasters"
    config.outputs.statistics = tmp_path / "outputs" / "statistics"
    config.outputs.reports = tmp_path / "outputs" / "reports"

    layer_a = _large_layer("scene-a", 1.0)
    layer_b = _large_layer("scene-b", 2.0)

    doubled = process_in_chunks(
        layer_a.data,
        chunk_size=16,
        function=lambda chunk: chunk + 1,
        parallel_workers=4,
    )
    report = run_raster_processing((layer_a, layer_b), config, output_root=tmp_path)

    assert doubled.shape == layer_a.data.shape
    assert report.artifacts["statistics"].exists()
    assert report.statistics[-1].cloud_coverage == 0.0
