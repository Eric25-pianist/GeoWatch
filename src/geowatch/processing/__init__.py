"""Raster processing engine for GeoWatch Phase 3."""

from __future__ import annotations

from pathlib import Path

from geowatch.config.models import ProjectConfig
from geowatch.processing.models import (
    ProcessingReport,
    RasterGrid,
    RasterLayer,
    RasterStatistics,
)


def run_raster_processing(
    layers: tuple[RasterLayer, ...],
    config: ProjectConfig,
    *,
    output_root: Path | None = None,
) -> ProcessingReport:
    """Load and run the processing engine without creating import cycles."""
    from geowatch.processing.engine import run_raster_processing as run_engine

    return run_engine(layers, config, output_root=output_root)


__all__ = [
    "ProcessingReport",
    "RasterGrid",
    "RasterLayer",
    "RasterStatistics",
    "run_raster_processing",
]
