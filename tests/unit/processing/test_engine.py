"""Unit tests for the Phase 3 raster engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from geowatch.config.models import AOIConfig, DateRangeConfig, ProjectConfig
from geowatch.processing.engine import (
    align_layers,
    apply_cloud_mask,
    calculate_statistics,
    clip_layer,
    clip_layer_to_geometry,
    cloud_coverage_statistics,
    generate_processing_report,
    mosaic_layers,
    process_in_chunks,
    resample_layer,
    run_raster_processing,
    temporal_composite,
    validate_layers,
)
from geowatch.processing.io import reproject_layer
from geowatch.processing.models import RasterGrid, RasterLayer


def _grid(width: int = 4, height: int = 4) -> RasterGrid:
    return RasterGrid(
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, float(height)),
        width=width,
        height=height,
        band_names=("red", "green"),
        nodata=-9999.0,
    )


def _layer(
    name: str,
    values: np.ndarray,
    *,
    cloud_mask: np.ndarray | None = None,
) -> RasterLayer:
    return RasterLayer(
        name=name,
        data=values,
        grid=_grid(values.shape[-1], values.shape[-2]),
        cloud_mask=cloud_mask,
    )


def test_cloud_mask_and_statistics() -> None:
    """Cloud masking should replace masked pixels and compute coverage."""
    data = np.ones((2, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    layer = _layer("scene-a", data, cloud_mask=mask)

    masked = apply_cloud_mask(layer, mask_value=-9999.0)
    stats = calculate_statistics(masked)

    assert masked.data[:, 0, 0].tolist() == [-9999.0, -9999.0]
    assert cloud_coverage_statistics(mask) == 1 / 16
    assert stats.cloud_pixels == 1


def test_validate_layers_reports_alignment_notes() -> None:
    """Layer validation should report when alignment is needed."""
    layer_a = _layer("scene-a", np.ones((2, 4, 4), dtype=np.float32))
    layer_b = _layer("scene-b", np.ones((2, 4, 4), dtype=np.float32)).with_data(
        np.ones((2, 4, 4), dtype=np.float32),
        grid=RasterGrid(
            crs="EPSG:3857",
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
            width=4,
            height=4,
            band_names=("red", "green"),
            nodata=-9999.0,
        ),
    )

    messages = validate_layers((layer_a, layer_b))

    assert any("aligned" in message for message in messages)


def test_clip_align_mosaic_and_composite() -> None:
    """The core raster workflow should clip, align, mosaic, and composite."""
    first = _layer("scene-a", np.arange(32, dtype=np.float32).reshape(2, 4, 4))
    second = _layer("scene-b", np.full((2, 4, 4), 10.0, dtype=np.float32))

    clipped = clip_layer(first, (1.0, 0.0, 3.0, 3.0))
    aligned = align_layers((clipped, second), resampling="bilinear")
    mosaic = mosaic_layers(aligned)
    composite = temporal_composite(aligned)

    assert clipped.data.shape == (2, 3, 2)
    assert aligned[0].data.shape == aligned[1].data.shape
    assert mosaic.data.shape == aligned[0].data.shape
    assert composite.name.endswith("_composite")


def test_resample_and_chunk_processing() -> None:
    """Resampling and chunk processing should preserve shapes."""
    layer = _layer("scene-a", np.arange(32, dtype=np.float32).reshape(2, 4, 4))
    resampled = resample_layer(layer, target_shape=(8, 8))
    doubled = process_in_chunks(
        resampled.data,
        chunk_size=2,
        function=lambda chunk: chunk * 2,
        parallel_workers=2,
    )

    assert resampled.data.shape == (2, 8, 8)
    assert doubled.shape == (2, 8, 8)
    assert doubled[0, 0, 0] == resampled.data[0, 0, 0] * 2


def test_resample_layer_preserves_extent() -> None:
    """Resampling should keep the same spatial extent."""
    layer = _layer("scene-a", np.arange(32, dtype=np.float32).reshape(2, 4, 4))
    resampled = resample_layer(layer, target_shape=(8, 8))

    assert resampled.grid.transform[2] == layer.grid.transform[2]
    assert resampled.grid.transform[5] == layer.grid.transform[5]
    assert resampled.grid.transform[0] == layer.grid.transform[0] / 2
    assert resampled.grid.transform[4] == layer.grid.transform[4] / 2


def test_clip_layer_to_geometry_masks_polygon() -> None:
    """Polygon clipping should return a masked raster subset."""
    from shapely.geometry import box

    layer = _layer("scene-a", np.arange(32, dtype=np.float32).reshape(2, 4, 4))
    clipped = clip_layer_to_geometry(
        layer,
        box(1.0, 1.0, 3.0, 3.0),
        geometry_crs="EPSG:4326",
    )

    assert clipped.data.shape == (2, 2, 2)
    assert clipped.grid.width == 2
    assert clipped.grid.height == 2


def test_run_raster_processing_writes_outputs(tmp_path: Path) -> None:
    """The full raster pipeline should write artifacts and a report."""
    config = ProjectConfig(
        project_name="phase-three",
        aoi=AOIConfig(kind="bbox", bbox=(0.0, 0.0, 2.0, 2.0)),
        dates=DateRangeConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        ),
    )
    config.outputs.root = tmp_path / "outputs"
    config.outputs.rasters = tmp_path / "outputs" / "rasters"
    config.outputs.statistics = tmp_path / "outputs" / "statistics"
    config.outputs.reports = tmp_path / "outputs" / "reports"

    layer1 = _layer("scene-a", np.ones((2, 4, 4), dtype=np.float32))
    layer2 = _layer("scene-b", np.full((2, 4, 4), 3.0, dtype=np.float32))

    report = run_raster_processing((layer1, layer2), config, output_root=tmp_path)

    assert report.phase == 3
    assert report.artifacts["stack"].exists()
    assert report.artifacts["report"].exists()
    assert "Raster alignment completed." in report.messages


def test_reproject_layer_same_crs_is_noop() -> None:
    """Reprojection should be a no-op when the CRS already matches."""
    layer = _layer("scene-a", np.ones((2, 4, 4), dtype=np.float32))

    reproj = reproject_layer(layer, target_crs="EPSG:4326")

    assert reproj.grid.crs == "EPSG:4326"
    assert np.array_equal(reproj.data, layer.data)


def test_generate_processing_report_text() -> None:
    """The report text should mention the phase and layers."""
    layer = _layer("scene-a", np.ones((2, 4, 4), dtype=np.float32))
    stats = [calculate_statistics(layer)]

    text = generate_processing_report(
        (layer,),
        layer,
        [stats[0].__dict__],
        artifacts={"stack": Path("stack.npz")},
    )

    assert "Phase 3" in text
    assert "scene-a" in text
