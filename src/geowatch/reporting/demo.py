"""Synthetic demo data for Phase 5 publication outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pyproj import Transformer

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.classification import CLASS_PROTOTYPE_SIGNATURES
from geowatch.analytics.models import ANALYTICS_CLASS_NAMES
from geowatch.config.models import AOIConfig, ProjectConfig
from geowatch.processing.models import RasterGrid, RasterLayer

_BAND_NAMES: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
)


@dataclass(frozen=True)
class DemoPublicationInputs:
    """Synthetic inputs used to build example Phase 5 outputs."""

    scene_t1: RasterLayer
    scene_t2: RasterLayer
    training_labels_t1: NDArray[np.int64]
    training_labels_t2: NDArray[np.int64]
    reference_labels_t1: NDArray[np.int64]
    reference_labels_t2: NDArray[np.int64]
    sources: tuple[SceneMetadata, ...]


def build_demo_publication_inputs(
    config: ProjectConfig,
    *,
    width: int = 160,
    height: int = 120,
) -> DemoPublicationInputs:
    """Build a synthetic two-scene demo set for cartography publication."""
    bbox = _require_bbox(config.aoi)
    grid = _build_projected_grid(bbox, width=width, height=height)
    class_map_t1 = _build_class_map(height=height, width=width)
    class_map_t2 = _build_changed_class_map(class_map_t1)
    scene_t1 = _build_scene("demo_t1", class_map_t1, grid, seed=7)
    scene_t2 = _build_scene("demo_t2", class_map_t2, grid, seed=19)
    sources = _build_sources(config.project_name, bbox)
    logger.info(
        "Built demo publication inputs for {} with {}x{} scene dimensions.",
        config.project_name,
        width,
        height,
    )
    return DemoPublicationInputs(
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        training_labels_t1=class_map_t1,
        training_labels_t2=class_map_t2,
        reference_labels_t1=class_map_t1.copy(),
        reference_labels_t2=class_map_t2.copy(),
        sources=sources,
    )


def _require_bbox(aoi: AOIConfig) -> tuple[float, float, float, float]:
    """Extract a bounding box for demo scene synthesis."""
    if aoi.bbox is None:
        raise ValueError("Phase 5 demo publication requires a bbox AOI.")
    return aoi.bbox


def _build_projected_grid(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> RasterGrid:
    """Create a projected raster grid aligned to the AOI bbox."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    west, south, east, north = bbox
    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)
    pixel_width = (xmax - xmin) / float(width)
    pixel_height = (ymax - ymin) / float(height)
    transform = (pixel_width, 0.0, xmin, 0.0, -pixel_height, ymax)
    return RasterGrid(
        crs="EPSG:3857",
        transform=transform,
        width=width,
        height=height,
        band_names=_BAND_NAMES,
        nodata=-9999.0,
    )


def _build_class_map(*, height: int, width: int) -> NDArray[np.int64]:
    """Create a spatially varied class map covering all LULC classes."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    class_map = np.full((height, width), 2, dtype=np.int64)

    snow = y < 0.14
    agriculture = (x > 0.62) & (y < 0.42)
    urban = (x > 0.34) & (x < 0.66) & (y > 0.32) & (y < 0.68)
    water_center = ((x - 0.24) ** 2) / 0.018 + ((y - 0.72) ** 2) / 0.035 < 1.0
    wetlands = ((x - 0.24) ** 2) / 0.028 + ((y - 0.72) ** 2) / 0.05 < 1.0
    bare_soil = (x > 0.70) & (y > 0.58)
    forest = (x < 0.28) & (y < 0.42)

    class_map[snow] = 7
    class_map[agriculture] = 3
    class_map[urban] = 1
    class_map[water_center] = 0
    class_map[wetlands] = 6
    class_map[bare_soil] = 4
    class_map[forest] = 5
    class_map[
        (~snow)
        & (~agriculture)
        & (~urban)
        & (~water_center)
        & (~wetlands)
        & (~bare_soil)
        & (~forest)
    ] = 2
    return class_map


def _build_changed_class_map(
    class_map: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Create a slightly changed class map for the after scene."""
    changed = class_map.copy()
    height, width = changed.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    changed[(x > 0.35) & (x < 0.68) & (y > 0.30) & (y < 0.56)] = 1
    changed[(x > 0.66) & (y < 0.18)] = 4
    changed[((x - 0.24) ** 2) / 0.022 + ((y - 0.72) ** 2) / 0.04 < 1.0] = 6
    changed[(x < 0.24) & (y < 0.18)] = 0
    changed[(x < 0.22) & (y > 0.24) & (y < 0.40)] = 2
    changed[(x > 0.76) & (y > 0.48)] = 4
    changed[(x < 0.24) & (y < 0.10)] = 7
    return changed


def _build_scene(
    name: str,
    class_map: NDArray[np.int64],
    grid: RasterGrid,
    *,
    seed: int,
) -> RasterLayer:
    """Synthesize a multispectral scene from class signatures."""
    rng = np.random.default_rng(seed)
    height, width = class_map.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    gradient = 0.025 * x + 0.018 * y
    data = np.zeros((len(_BAND_NAMES), height, width), dtype=np.float32)

    for class_index, class_name in enumerate(ANALYTICS_CLASS_NAMES):
        signature = CLASS_PROTOTYPE_SIGNATURES[class_name]
        mask = class_map == class_index
        for band_index, band_name in enumerate(_BAND_NAMES):
            base = float(signature[band_name])
            band = data[band_index]
            band[mask] = base + gradient[mask] + (0.03 * class_index / len(_BAND_NAMES))

    noise = rng.normal(0.0, 0.0035, size=data.shape).astype(np.float32)
    data = np.clip(data + noise, 0.0, 1.0)
    return RasterLayer(name=name, data=data, grid=grid)


def _build_sources(
    project_name: str,
    bbox: tuple[float, float, float, float],
) -> tuple[SceneMetadata, ...]:
    """Create synthetic satellite source metadata for the report."""
    base_date = date(2024, 2, 14)
    return (
        SceneMetadata(
            scene_id=f"{project_name}-sentinel-2-t1",
            provider="copernicus",
            dataset="sentinel-2-l2a",
            acquired_at=datetime.combine(base_date, datetime.min.time()),
            bbox=bbox,
            cloud_cover=8.5,
            assets=(),
            metadata={"demo": True, "sensor": "Sentinel-2"},
            source_url="https://example.com/geowatch/demo/sentinel-2",
        ),
        SceneMetadata(
            scene_id=f"{project_name}-landsat-8-t2",
            provider="usgs",
            dataset="landsat-8-c2-l2",
            acquired_at=datetime.combine(
                date(2024, 3, 2),
                datetime.min.time(),
            ),
            bbox=bbox,
            cloud_cover=11.2,
            assets=(),
            metadata={"demo": True, "sensor": "Landsat 8"},
            source_url="https://example.com/geowatch/demo/landsat-8",
        ),
    )
