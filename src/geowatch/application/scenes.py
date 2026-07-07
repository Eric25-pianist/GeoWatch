"""Build aligned, scaled, cloud-masked scene composites from acquisition catalogs."""

from __future__ import annotations

import importlib.util
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from geowatch.application.sensors import (
    SensorProfile,
    landsat_cloud_mask,
    sentinel_cloud_mask,
)
from geowatch.core.errors import GeoWatchError
from geowatch.processing.io import write_raster
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.utils.geometry import (
    geometry_mask_for_grid,
    load_vector_geometry,
    reproject_geometry,
)

CANONICAL_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
DEFAULT_MAX_PIXELS = 12_000_000


def build_year_composite(
    catalog_path: Path,
    boundary_path: Path,
    profile: SensorProfile,
    *,
    year: int,
    output_path: Path,
    method: str = "median",
    target_crs: str = "auto",
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_valid_coverage: float = 0.70,
    hard_min_valid_coverage: float = 0.20,
) -> RasterLayer:
    """Create one harmonized yearly reflectance composite from downloaded assets."""
    if importlib.util.find_spec("rasterio") is None:
        raise GeoWatchError(
            "Real imagery processing requires Rasterio. Run setup-micromamba.ps1 "
            "or install GeoWatch with the 'geo' extra under Python 3.12."
        )
    catalog = _catalog_payload(catalog_path)
    downloads = _catalog_downloads(catalog_path, payload=catalog)
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for item in downloads:
        grouped[str(item["scene_id"])][str(item["asset_name"])] = Path(
            str(item["path"])
        )
    complete = [
        assets for assets in grouped.values() if _has_required_assets(assets, profile)
    ]
    if not complete:
        raise GeoWatchError(
            f"No complete analytical scene was downloaded for {year}. "
            "Required: six reflectance bands and one QA asset."
        )

    boundary = load_vector_geometry(boundary_path)
    destination_crs = (
        _local_utm(boundary.geometry.centroid.x, boundary.geometry.centroid.y)
        if target_crs == "auto"
        else target_crs
    )
    projected = reproject_geometry(boundary.geometry, boundary.crs, destination_crs)
    resolution, grid = _target_grid(
        projected.bounds,
        profile.resolution_m,
        max_pixels=max_pixels,
        crs=destination_crs,
    )
    logger.info(
        "Building {} composite at {:.1f} m in {} using {} scene(s)",
        year,
        resolution,
        destination_crs,
        len(complete),
    )
    scenes = [_warp_scene(assets, profile, grid, projected) for assets in complete]
    stack = np.stack([scene.data for scene in scenes], axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        if method == "mean":
            data = np.nanmean(stack, axis=0)
        elif method == "first":
            data = stack[0]
            for candidate in stack[1:]:
                data = np.where(np.isfinite(data), data, candidate)
        else:
            data = np.nanmedian(stack, axis=0)
    invalid = ~np.any(np.isfinite(data), axis=0)
    inside = geometry_mask_for_grid(projected, grid)
    inside_pixels = int(inside.sum())
    valid_inside = int((inside & np.isfinite(data[0])).sum())
    valid_coverage = valid_inside / inside_pixels if inside_pixels else 0.0
    if valid_coverage < hard_min_valid_coverage:
        raise GeoWatchError(
            f"{year} composite has only {valid_coverage:.1%} valid AOI coverage; "
            f"hard minimum is {hard_min_valid_coverage:.1%}. Try a wider season, "
            "a higher cloud ceiling, more scenes, or a newer sensor."
        )
    quality_warnings: list[str] = []
    if valid_coverage < min_valid_coverage:
        warning = (
            f"{year} composite has {valid_coverage:.1%} valid AOI coverage, below "
            f"the recommended {min_valid_coverage:.1%} target. GeoWatch will "
            "continue and mark the run as lower quality."
        )
        quality_warnings.append(warning)
        logger.warning(warning)
    finite = data[:, inside & np.isfinite(data[0])]
    outside_physical_range = (
        int(((finite < 0.0) | (finite > 1.0)).sum()) if finite.size else 0
    )
    raw_catalog_scenes = catalog.get("scenes", [])
    catalog_scenes = raw_catalog_scenes if isinstance(raw_catalog_scenes, list) else []
    result = RasterLayer(
        name=f"{profile.dataset}-{year}",
        data=data.astype(np.float32),
        grid=grid,
        cloud_mask=invalid,
        metadata={
            "dataset": profile.dataset,
            "year": year,
            "scene_count": len(scenes),
            "resolution_m": resolution,
            "catalog": str(catalog_path),
            "valid_aoi_fraction": valid_coverage,
            "recommended_valid_aoi_fraction": min_valid_coverage,
            "hard_minimum_valid_aoi_fraction": hard_min_valid_coverage,
            "quality_warnings": tuple(quality_warnings),
            "saturated_pixels_masked": sum(
                _integer_metadata(scene, "saturated_pixels") for scene in scenes
            ),
            "reflectance_values_outside_0_1": outside_physical_range,
            "source_scene_ids": tuple(grouped),
            "source_dates": tuple(
                str(scene.get("acquired_at"))
                for scene in catalog_scenes
                if isinstance(scene, dict) and scene.get("scene_id") in grouped
            ),
            "provider": str(catalog.get("provider", "unknown")),
        },
    )
    written = write_raster(result, output_path, driver="COG")
    result.metadata["output_path"] = str(written)
    return result


def load_processed_composite(path: Path) -> RasterLayer:
    """Read a previously generated composite for a resumed run."""
    from geowatch.processing.io import read_raster

    if path.exists():
        return read_raster(path)
    fallback = path.with_suffix(".npz")
    if fallback.exists():
        raise GeoWatchError(
            "Fallback NPZ composites cannot be used for production publication. "
            "Install Rasterio and reprocess the project."
        )
    raise GeoWatchError(f"Processed composite does not exist: {path}")


def _warp_scene(
    assets: dict[str, Path],
    profile: SensorProfile,
    grid: RasterGrid,
    boundary: Any,
) -> RasterLayer:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    from rasterio.warp import reproject
    from shapely.geometry import mapping

    destination_transform = Affine(*grid.transform)
    arrays: list[np.ndarray] = []
    for canonical in CANONICAL_BANDS:
        asset_name, path = _find_asset(assets, profile.band_aliases[canonical])
        destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        with rasterio.open(path) as source:
            source_data = source.read(1).astype(np.float32)
            source_nodata = source.nodata
            if source_nodata is not None:
                source_data[source_data == source_nodata] = np.nan
            source_data = (source_data * profile.scale) + profile.offset
            reproject(
                source=source_data,
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=destination_transform,
                dst_crs=grid.crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        logger.debug("Warped {} from {}", canonical, asset_name)
        arrays.append(destination)

    qa_name, qa_path = _find_asset(assets, profile.qa_aliases)
    qa_destination = _warp_quality_asset(qa_path, grid, destination_transform)
    cloud_mask = (
        sentinel_cloud_mask(qa_destination, asset_name=qa_name)
        if profile.dataset == "sentinel-2-l2a"
        else landsat_cloud_mask(qa_destination)
    )
    inside = geometry_mask(
        [mapping(boundary)],
        out_shape=(grid.height, grid.width),
        transform=destination_transform,
        invert=True,
    )
    saturation_mask = np.zeros_like(cloud_mask, dtype=bool)
    if profile.saturation_aliases:
        _, saturation_path = _find_asset(assets, profile.saturation_aliases)
        saturation = _warp_quality_asset(saturation_path, grid, destination_transform)
        saturation_mask = saturation != 0
    invalid = np.asarray(cloud_mask | saturation_mask | ~inside, dtype=bool)
    data = np.stack(arrays, axis=0).astype(np.float32)
    data[:, invalid] = np.nan
    return RasterLayer(
        name="warped-scene",
        data=data,
        grid=grid,
        cloud_mask=invalid,
        metadata={"saturated_pixels": int((saturation_mask & inside).sum())},
    )


def _warp_quality_asset(
    path: Path,
    grid: RasterGrid,
    destination_transform: Any,
) -> np.ndarray:
    """Warp an integer QA asset to the analytical grid with nearest sampling."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.zeros((grid.height, grid.width), dtype=np.uint16)
    with rasterio.open(path) as source:
        reproject(
            source=source.read(1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=destination_transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
        )
    return destination


def _target_grid(
    bounds: tuple[float, float, float, float],
    resolution: float,
    *,
    max_pixels: int,
    crs: str,
) -> tuple[float, RasterGrid]:
    west, south, east, north = bounds
    width = max(1, math.ceil((east - west) / resolution))
    height = max(1, math.ceil((north - south) / resolution))
    if width * height > max_pixels:
        factor = math.sqrt((width * height) / max_pixels)
        resolution *= factor
        width = max(1, math.ceil((east - west) / resolution))
        height = max(1, math.ceil((north - south) / resolution))
        logger.warning(
            "AOI exceeds the laptop pixel budget; effective resolution is {:.1f} m.",
            resolution,
        )
    return resolution, RasterGrid(
        crs=crs,
        transform=(resolution, 0.0, west, 0.0, -resolution, north),
        width=width,
        height=height,
        band_names=CANONICAL_BANDS,
        nodata=np.nan,
    )


def _catalog_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeoWatchError(f"Could not read acquisition catalog: {path}") from exc
    if not isinstance(payload, dict):
        raise GeoWatchError(f"Acquisition catalog root is invalid: {path}")
    return payload


def _catalog_downloads(
    path: Path,
    *,
    payload: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    catalog = payload or _catalog_payload(path)
    downloads = catalog.get("downloads")
    if not isinstance(downloads, list):
        raise GeoWatchError(f"Catalog contains no download list: {path}")
    return [item for item in downloads if isinstance(item, dict)]


def _has_required_assets(assets: dict[str, Path], profile: SensorProfile) -> bool:
    names = {name.casefold() for name in assets}
    spectral = all(
        any(alias.casefold() in names for alias in profile.band_aliases[band])
        for band in CANONICAL_BANDS
    )
    qa = any(alias.casefold() in names for alias in profile.qa_aliases)
    saturation = not profile.saturation_aliases or any(
        alias.casefold() in names for alias in profile.saturation_aliases
    )
    return spectral and qa and saturation


def _find_asset(assets: dict[str, Path], aliases: tuple[str, ...]) -> tuple[str, Path]:
    folded = {name.casefold(): (name, path) for name, path in assets.items()}
    for alias in aliases:
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    raise GeoWatchError(f"Missing required asset; accepted names: {aliases}")


def _local_utm(longitude: float, latitude: float) -> str:
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def _integer_metadata(layer: RasterLayer, key: str) -> int:
    value = layer.metadata.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0
