"""Pure NumPy raster processing engine with optional Rasterio integration."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger
from scipy.ndimage import zoom
from shapely.geometry.base import BaseGeometry

from geowatch.config.models import ProjectConfig
from geowatch.processing.errors import ProcessingError
from geowatch.processing.io import (
    reproject_layer,
    write_cog_profile,
    write_raster,
    write_vrt,
)
from geowatch.processing.models import (
    ProcessingReport,
    RasterGrid,
    RasterLayer,
    RasterStatistics,
)
from geowatch.utils.geometry import (
    geometry_mask_for_grid,
    load_vector_geometry,
    reproject_geometry,
)


def validate_layers(layers: tuple[RasterLayer, ...]) -> list[str]:
    """Return human-readable quality checks for a raster stack."""
    messages: list[str] = []
    if not layers:
        raise ProcessingError("At least one raster layer is required.")
    band_counts = {layer.band_count for layer in layers}
    if len(band_counts) != 1:
        raise ProcessingError("All raster layers must have the same band count.")
    crs_values = {layer.grid.crs for layer in layers}
    if len(crs_values) != 1:
        messages.append("Layers span multiple CRS values and will be aligned.")
    messages.append(f"Validated {len(layers)} raster layers.")
    return messages


def cloud_coverage_statistics(mask: np.ndarray) -> float:
    """Return the fraction of masked pixels in a cloud mask."""
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def apply_cloud_mask(layer: RasterLayer, mask_value: float | int) -> RasterLayer:
    """Apply a boolean cloud mask to a raster layer."""
    if layer.cloud_mask is None:
        return layer
    data = layer.data.copy()
    if data.dtype.kind in {"i", "u"}:
        data = data.astype(np.float32)
    data[:, layer.cloud_mask] = mask_value
    return layer.with_data(data)


def clip_layer(
    layer: RasterLayer, bbox: tuple[float, float, float, float]
) -> RasterLayer:
    """Clip a north-up raster to an AOI bbox."""
    west, south, east, north = bbox
    row1, col1 = layer.grid.world_to_pixel(west, north)
    row2, col2 = layer.grid.world_to_pixel(east, south)
    row_start = max(0, min(row1, row2))
    row_stop = min(layer.grid.height, max(row1, row2))
    col_start = max(0, min(col1, col2))
    col_stop = min(layer.grid.width, max(col1, col2))
    if row_start >= row_stop or col_start >= col_stop:
        raise ProcessingError("AOI clip produced an empty raster.")
    clipped = layer.data[:, row_start:row_stop, col_start:col_stop]
    clipped_mask = (
        layer.cloud_mask[row_start:row_stop, col_start:col_stop]
        if layer.cloud_mask is not None
        else None
    )
    return layer.with_data(
        clipped,
        grid=layer.grid.subset(slice(row_start, row_stop), slice(col_start, col_stop)),
        cloud_mask=clipped_mask,
    )


def clip_layer_to_geometry(
    layer: RasterLayer,
    geometry: BaseGeometry,
    *,
    geometry_crs: str,
) -> RasterLayer:
    """Clip and mask a raster layer using a polygon geometry."""
    projected_geometry = reproject_geometry(geometry, geometry_crs, layer.grid.crs)
    west, south, east, north = projected_geometry.bounds
    row_start, col_start = _grid_index_for_world(layer.grid, west, north, start=True)
    row_stop, col_stop = _grid_index_for_world(layer.grid, east, south, start=False)
    row_start = max(0, min(row_start, row_stop))
    row_stop = min(layer.grid.height, max(row_start, row_stop))
    col_start = max(0, min(col_start, col_stop))
    col_stop = min(layer.grid.width, max(col_start, col_stop))
    if row_start >= row_stop or col_start >= col_stop:
        raise ProcessingError("AOI geometry does not overlap the raster footprint.")

    clipped = layer.data[:, row_start:row_stop, col_start:col_stop]
    clipped_grid = layer.grid.subset(
        slice(row_start, row_stop),
        slice(col_start, col_stop),
    )
    mask = geometry_mask_for_grid(projected_geometry, clipped_grid)
    masked = clipped.astype(np.float32, copy=True)
    masked[:, ~mask] = np.nan
    clipped_cloud_mask = (
        layer.cloud_mask[row_start:row_stop, col_start:col_stop]
        if layer.cloud_mask is not None
        else np.zeros(mask.shape, dtype=bool)
    )
    clipped_cloud_mask = np.asarray(clipped_cloud_mask | ~mask, dtype=bool)
    return layer.with_data(
        masked,
        grid=clipped_grid,
        cloud_mask=clipped_cloud_mask,
    )


def resample_layer(
    layer: RasterLayer,
    *,
    target_shape: tuple[int, int],
    method: str = "bilinear",
) -> RasterLayer:
    """Resample a raster layer to a new height and width."""
    row_scale = target_shape[0] / layer.grid.height
    col_scale = target_shape[1] / layer.grid.width
    order = {"nearest": 0, "bilinear": 1, "cubic": 3}[method]
    data = zoom(layer.data, (1, row_scale, col_scale), order=order)
    cloud_mask = None
    if layer.cloud_mask is not None:
        cloud_mask = (
            zoom(
                layer.cloud_mask.astype(np.uint8),
                (row_scale, col_scale),
                order=0,
            )
            > 0
        )
    a, b, c, d, e, f = layer.grid.transform
    new_transform = (
        a / col_scale,
        b,
        c,
        d,
        e / row_scale,
        f,
    )
    new_grid = RasterGrid(
        crs=layer.grid.crs,
        transform=new_transform,
        width=target_shape[1],
        height=target_shape[0],
        band_names=layer.grid.band_names,
        nodata=layer.grid.nodata,
    )
    return layer.with_data(data, grid=new_grid, cloud_mask=cloud_mask)


def align_layers(
    layers: tuple[RasterLayer, ...],
    *,
    target_grid: RasterGrid | None = None,
    resampling: str = "bilinear",
) -> tuple[RasterLayer, ...]:
    """Align multiple raster layers to a shared grid."""
    if not layers:
        raise ProcessingError("No raster layers were provided.")
    base_grid = target_grid or layers[0].grid
    aligned: list[RasterLayer] = []
    for layer in layers:
        if (
            layer.grid.width == base_grid.width
            and layer.grid.height == base_grid.height
        ):
            aligned.append(layer.with_data(layer.data, grid=base_grid))
        else:
            aligned.append(
                resample_layer(
                    layer,
                    target_shape=(base_grid.height, base_grid.width),
                    method=resampling,
                )
            )
    return tuple(aligned)


def mosaic_layers(
    layers: tuple[RasterLayer, ...],
    *,
    reducer: str = "first",
) -> RasterLayer:
    """Merge aligned raster layers into a single mosaic."""
    if not layers:
        raise ProcessingError("No raster layers available for mosaicking.")
    base = layers[0]
    data_stack = np.stack([layer.data for layer in layers], axis=0).astype(np.float32)
    if reducer == "mean":
        mosaic_data = np.nanmean(data_stack, axis=0)
    elif reducer == "max":
        mosaic_data = np.nanmax(data_stack, axis=0)
    else:
        mosaic_data = data_stack[0]
        for candidate in data_stack[1:]:
            mosaic_data = np.where(np.isnan(mosaic_data), candidate, mosaic_data)
    cloud_mask = None
    if any(layer.cloud_mask is not None for layer in layers):
        masks = [layer.cloud_mask for layer in layers if layer.cloud_mask is not None]
        cloud_mask = np.any(np.stack(masks, axis=0), axis=0)
    return base.with_data(mosaic_data, cloud_mask=cloud_mask)


def temporal_composite(
    layers: tuple[RasterLayer, ...],
    *,
    method: str = "median",
) -> RasterLayer:
    """Create a temporal composite from aligned raster layers."""
    if not layers:
        raise ProcessingError("No raster layers available for compositing.")
    base = layers[0]
    data_stack = np.stack([layer.data for layer in layers], axis=0).astype(np.float32)
    if method == "mean":
        composite = np.nanmean(data_stack, axis=0)
    elif method == "max":
        composite = np.nanmax(data_stack, axis=0)
    else:
        composite = np.nanmedian(data_stack, axis=0)
    return base.with_data(composite, name=f"{base.name}_composite")


def check_projection_consistency(layers: tuple[RasterLayer, ...]) -> list[str]:
    """Return projection consistency warnings."""
    crs_values = {layer.grid.crs for layer in layers}
    if len(crs_values) > 1:
        return ["Multiple CRS values detected; alignment is required."]
    return ["Projection check passed."]


def check_band_consistency(layers: tuple[RasterLayer, ...]) -> list[str]:
    """Return band consistency warnings."""
    band_counts = {layer.band_count for layer in layers}
    if len(band_counts) > 1:
        return ["Band counts differ between layers."]
    return ["Band count check passed."]


def calculate_statistics(layer: RasterLayer) -> RasterStatistics:
    """Calculate summary statistics for a layer."""
    data = np.asarray(layer.data, dtype=np.float32)
    if layer.grid.nodata is not None:
        data = np.where(data == layer.grid.nodata, np.nan, data)
    cloud_mask = (
        layer.cloud_mask
        if layer.cloud_mask is not None
        else np.zeros((layer.grid.height, layer.grid.width), dtype=bool)
    )
    valid_pixels = int(np.isfinite(data).sum())
    nodata_pixels = int(np.isnan(data).sum())
    cloud_pixels = int(cloud_mask.sum())
    finite = data[np.isfinite(data)]
    minimum = float(finite.min()) if finite.size else float("nan")
    maximum = float(finite.max()) if finite.size else float("nan")
    mean = float(finite.mean()) if finite.size else float("nan")
    std = float(finite.std()) if finite.size else float("nan")
    coverage = cloud_coverage_statistics(cloud_mask)
    return RasterStatistics(
        layer_name=layer.name,
        valid_pixels=valid_pixels,
        cloud_pixels=cloud_pixels,
        nodata_pixels=nodata_pixels,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=std,
        cloud_coverage=coverage,
    )


def process_in_chunks(
    array: np.ndarray,
    *,
    chunk_size: int,
    function: Callable[[np.ndarray], np.ndarray],
    parallel_workers: int = 1,
) -> np.ndarray:
    """Process a 2D or 3D array in chunks using optional parallel execution."""
    if array.ndim not in {2, 3}:
        raise ProcessingError("Chunk processing expects a 2D or 3D array.")
    output = np.empty_like(array)
    windows = list(_chunk_windows(array.shape[-2:], chunk_size))

    def work(window: tuple[slice, slice]) -> tuple[tuple[slice, slice], np.ndarray]:
        rows, cols = window
        chunk = array[..., rows, cols]
        return window, function(chunk)

    if parallel_workers > 1 and len(windows) > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for window, result in executor.map(work, windows):
                rows, cols = window
                output[..., rows, cols] = result
    else:
        for window in windows:
            rows, cols = window
            output[..., rows, cols] = function(array[..., rows, cols])
    return output


def process_with_dask(
    array: np.ndarray,
    *,
    chunk_size: int,
    function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Process an array with Dask when available, otherwise use chunked NumPy."""
    if importlib.util.find_spec("dask.array") is None:
        return process_in_chunks(
            array,
            chunk_size=chunk_size,
            function=function,
            parallel_workers=1,
        )
    import dask.array as dask_array_module

    da = cast(Any, dask_array_module)

    chunks = (
        (array.shape[0], chunk_size, chunk_size)
        if array.ndim == 3
        else (chunk_size, chunk_size)
    )
    dask_array = da.from_array(array, chunks=chunks)
    result = dask_array.map_blocks(function, dtype=array.dtype)
    return cast(np.ndarray, result.compute())


def write_processing_outputs(
    layers: tuple[RasterLayer, ...],
    composite: RasterLayer,
    config: ProjectConfig,
    *,
    output_root: Path,
) -> dict[str, Path]:
    """Write stack, composite, VRT, statistics, and report artifacts."""
    base = output_root
    rasters_dir = _resolve_output_dir(base, config.outputs.rasters)
    statistics_dir = _resolve_output_dir(base, config.outputs.statistics)
    reports_dir = _resolve_output_dir(base, config.outputs.reports)
    rasters_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stack_path = rasters_dir / "clean_stack.npz"
    mosaic_path = rasters_dir / f"{layers[0].name}_mosaic.npz"
    composite_path = rasters_dir / f"{composite.name}.npz"
    cog_profile_path = rasters_dir / f"{composite.name}_cog_profile.json"
    stats_path = statistics_dir / "processing_statistics.json"
    report_path = reports_dir / "processing_report.md"
    vrt_path = rasters_dir / "clean_stack.vrt"

    np.savez_compressed(
        stack_path,
        stack=np.stack([layer.data for layer in layers], axis=0),
        names=np.array([layer.name for layer in layers], dtype=object),
    )
    mosaic = mosaic_layers(layers)
    np.savez_compressed(mosaic_path, data=mosaic.data, name="mosaic")
    composite_artifact = write_raster(
        composite,
        composite_path.with_suffix(".tif"),
        driver=config.raster_processing.output_driver,
        compress=config.raster_processing.compress,
    )
    mosaic_artifact = write_raster(
        mosaic,
        mosaic_path.with_suffix(".tif"),
        driver=config.raster_processing.output_driver,
        compress=config.raster_processing.compress,
    )
    cog_profile_path.write_text(
        json.dumps(
            write_cog_profile(
                composite.grid,
                compress=config.raster_processing.compress,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    write_vrt(layers, vrt_path)
    statistics = [asdict(calculate_statistics(layer)) for layer in layers]
    statistics.append(asdict(calculate_statistics(composite)))
    statistics.append(asdict(calculate_statistics(mosaic)))
    stats_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    report_path.write_text(
        generate_processing_report(
            layers,
            composite,
            statistics,
            artifacts={
                "stack": stack_path,
                "mosaic": mosaic_artifact,
                "composite": composite_artifact,
                "cog_profile": cog_profile_path,
                "vrt": vrt_path,
                "statistics": stats_path,
                "report": report_path,
            },
        ),
        encoding="utf-8",
    )
    logger.info("Wrote raster processing outputs to {}", rasters_dir)
    return {
        "stack": stack_path,
        "mosaic": mosaic_artifact,
        "composite": composite_artifact,
        "cog_profile": cog_profile_path,
        "vrt": vrt_path,
        "statistics": stats_path,
        "report": report_path,
    }


def generate_processing_report(
    layers: tuple[RasterLayer, ...],
    composite: RasterLayer,
    statistics: list[dict[str, object]],
    *,
    artifacts: dict[str, Path],
) -> str:
    """Render a markdown report for Phase 3 outputs."""
    lines = [
        "# GeoWatch Phase 3 Report",
        "",
        "- Phase: 3 - Raster Processing Engine",
        f"- Raster layers: {len(layers)}",
        f"- Composite: {composite.name}",
        "",
        "## Artifacts",
        "",
    ]
    for label, path in artifacts.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## Statistics", ""])
    for item in statistics:
        lines.append(
            f"- {item['layer_name']}: mean={item['mean']:.4f} "
            f"cloud={item['cloud_coverage']:.2%}"
        )
    return "\n".join(lines) + "\n"


def _chunk_windows(
    shape: tuple[int, int],
    chunk_size: int,
) -> Iterable[tuple[slice, slice]]:
    """Yield row/column slices for chunked processing."""
    rows, cols = shape
    for row_start in range(0, rows, chunk_size):
        for col_start in range(0, cols, chunk_size):
            yield (
                slice(row_start, min(rows, row_start + chunk_size)),
                slice(col_start, min(cols, col_start + chunk_size)),
            )


def run_raster_processing(
    layers: tuple[RasterLayer, ...],
    config: ProjectConfig,
    *,
    output_root: Path | None = None,
) -> ProcessingReport:
    """Run the Phase 3 raster processing workflow."""
    messages = validate_layers(layers)
    messages.extend(check_projection_consistency(layers))
    messages.extend(check_band_consistency(layers))

    clipped = (
        tuple(clip_layer(scene, config.aoi.bbox) for scene in layers)
        if config.aoi.bbox is not None and config.aoi.path is None
        else _clip_layers_with_aoi_geometry(layers, config)
    )
    masked = tuple(
        apply_cloud_mask(scene, config.raster_processing.mask_value)
        for scene in clipped
    )
    reprojected = tuple(
        reproject_layer(
            scene,
            target_crs=config.raster_processing.target_crs,
            resampling=config.raster_processing.resampling,
        )
        for scene in masked
    )
    aligned = align_layers(
        reprojected,
        resampling=config.raster_processing.resampling,
    )
    mosaic = mosaic_layers(aligned)
    composite = temporal_composite(aligned)
    artifacts = write_processing_outputs(
        aligned,
        composite,
        config,
        output_root=output_root or config.outputs.root,
    )
    messages.append("Cloud masking applied.")
    messages.append("Reprojection completed.")
    messages.append("Raster alignment completed.")
    messages.append("Temporal composite generated.")
    if config.raster_processing.use_dask:
        messages.append("Dask-compatible chunking enabled.")
    statistics = tuple(
        calculate_statistics(layer) for layer in (*aligned, mosaic, composite)
    )
    logger.info("Completed Phase 3 raster processing for {}", config.project_name)
    return ProcessingReport(
        phase=3,
        messages=tuple(messages),
        statistics=statistics,
        artifacts=artifacts,
    )


def _clip_layers_with_aoi_geometry(
    layers: tuple[RasterLayer, ...],
    config: ProjectConfig,
) -> tuple[RasterLayer, ...]:
    """Clip raster layers using the configured AOI geometry when available."""
    if config.aoi.path is None:
        return layers
    aoi_path = (
        config.aoi.path
        if config.aoi.path.is_absolute()
        else Path.cwd() / config.aoi.path
    )
    loaded = load_vector_geometry(aoi_path)
    return tuple(
        clip_layer_to_geometry(
            scene,
            loaded.geometry,
            geometry_crs=loaded.crs,
        )
        for scene in layers
    )


def _grid_index_for_world(
    grid: RasterGrid,
    x: float,
    y: float,
    *,
    start: bool,
) -> tuple[int, int]:
    """Convert world coordinates to row and column indices for a north-up grid."""
    a, _, c, _, e, f = grid.transform
    if a == 0.0 or e == 0.0:
        raise ProcessingError("Raster grid transform must be north-up.")
    col = (x - c) / a
    row = (y - f) / e
    if start:
        return math.floor(row), math.floor(col)
    return math.ceil(row), math.ceil(col)


def _resolve_output_dir(base: Path, configured: Path) -> Path:
    """Resolve an output directory relative to ``base`` when needed."""
    return configured if configured.is_absolute() else base / configured
