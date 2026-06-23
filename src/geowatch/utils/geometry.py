"""Geometry helpers for GeoWatch vector validation and raster masking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely import contains_xy, make_valid
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from geowatch.core.errors import GeoWatchError
from geowatch.processing.models import RasterGrid


@dataclass(frozen=True)
class LoadedGeometry:
    """Validated vector geometry with CRS metadata."""

    geometry: BaseGeometry
    crs: str
    source_path: Path
    feature_count: int


def load_vector_geometry(path: Path) -> LoadedGeometry:
    """Read a vector dataset and dissolve it into a single geometry."""
    try:
        frame = gpd.read_file(path)
    except Exception as exc:  # pragma: no cover - depends on local vector drivers
        logger.exception("Failed to read vector geometry {}", path)
        raise GeoWatchError(f"Could not read vector geometry: {path}") from exc

    if frame.empty:
        raise GeoWatchError(f"Vector geometry is empty: {path}")
    if frame.geometry.isna().any():
        raise GeoWatchError(f"Vector geometry contains missing features: {path}")

    crs = str(frame.crs) if frame.crs is not None else "EPSG:4326"
    geometry = frame.geometry.union_all()
    geometry = ensure_valid_geometry(geometry, source_path=path)
    logger.info(
        "Loaded vector geometry {} with {} features in {}",
        path,
        len(frame),
        crs,
    )
    return LoadedGeometry(
        geometry=geometry,
        crs=crs,
        source_path=path,
        feature_count=len(frame),
    )


def ensure_valid_geometry(
    geometry: BaseGeometry,
    *,
    source_path: Path | None = None,
) -> BaseGeometry:
    """Return a valid geometry, repairing simple topology issues when needed."""
    if geometry.is_valid:
        return geometry
    repaired = make_valid(geometry)
    if repaired.is_valid:
        logger.warning(
            "Repaired invalid geometry{}",
            f" from {source_path}" if source_path is not None else "",
        )
        return repaired
    fallback = geometry.buffer(0.0)
    if fallback.is_valid and not fallback.is_empty:
        logger.warning(
            "Buffered invalid geometry{} to recover a valid footprint",
            f" from {source_path}" if source_path is not None else "",
        )
        return fallback
    raise GeoWatchError(
        f"Geometry could not be repaired{f' for {source_path}' if source_path else ''}."
    )


def reproject_geometry(
    geometry: BaseGeometry,
    source_crs: str,
    target_crs: str,
) -> BaseGeometry:
    """Reproject a geometry between two coordinate reference systems."""
    if CRS.from_user_input(source_crs) == CRS.from_user_input(target_crs):
        return geometry
    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )
    projected = transform_geometry(transformer.transform, geometry)
    return ensure_valid_geometry(projected)


def geometry_mask_for_grid(
    geometry: BaseGeometry,
    grid: RasterGrid,
    *,
    buffer_pixels: float = 0.5,
) -> NDArray[np.bool_]:
    """Rasterize a geometry into a boolean mask aligned with a raster grid."""
    a, _, c, _, e, f = grid.transform
    if a == 0.0 or e == 0.0:
        raise GeoWatchError("Raster grid must be north-up for geometry masking.")

    pixel_size = max(abs(a), abs(e)) * buffer_pixels
    buffered_geometry = geometry.buffer(pixel_size)
    x_centers = c + ((np.arange(grid.width, dtype=np.float64) + 0.5) * a)
    y_centers = f + ((np.arange(grid.height, dtype=np.float64) + 0.5) * e)
    xs, ys = np.meshgrid(x_centers, y_centers)
    mask = contains_xy(buffered_geometry, xs, ys)
    logger.debug(
        "Built geometry mask for grid {}x{} with {} covered pixels",
        grid.width,
        grid.height,
        int(np.sum(mask)),
    )
    return np.asarray(mask, dtype=bool)
