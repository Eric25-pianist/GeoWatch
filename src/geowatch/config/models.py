"""Pydantic V2 configuration schemas for GeoWatch Phase 1."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from geowatch.acquisition.models import AcquisitionConfig
from geowatch.cartography.themes import MapThemeName


class AOIConfig(BaseModel):
    """Area-of-interest configuration using either a bbox or vector file."""

    kind: Literal["bbox", "geojson", "shapefile", "geopackage"] = "bbox"
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box as west, south, east, north in EPSG:4326.",
    )
    path: Path | None = Field(default=None, description="Path to AOI vector data.")
    crs: str = Field(
        default="EPSG:4326",
        description="AOI coordinate reference system.",
    )

    @model_validator(mode="after")
    def validate_aoi_source(self) -> AOIConfig:
        """Ensure the selected AOI source has the required values."""
        if self.kind == "bbox":
            if self.bbox is None:
                logger.error("AOI kind 'bbox' requires bbox coordinates.")
                raise ValueError("AOI kind 'bbox' requires bbox coordinates.")
            west, south, east, north = self.bbox
            if west >= east or south >= north:
                logger.error("Invalid bbox extent: {}", self.bbox)
                raise ValueError("AOI bbox must be ordered west, south, east, north.")
            if not (-180 <= west <= 180 and -180 <= east <= 180):
                raise ValueError("AOI longitude values must be between -180 and 180.")
            if not (-90 <= south <= 90 and -90 <= north <= 90):
                raise ValueError("AOI latitude values must be between -90 and 90.")
        elif self.path is None:
            logger.error("AOI kind '{}' requires a file path.", self.kind)
            raise ValueError(f"AOI kind '{self.kind}' requires a file path.")
        return self


class DateRangeConfig(BaseModel):
    """Temporal range for a foundation validation run."""

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_order(self) -> DateRangeConfig:
        """Require the start date to be on or before the end date."""
        if self.start_date > self.end_date:
            logger.error(
                "Start date {} is after end date {}",
                self.start_date,
                self.end_date,
            )
            raise ValueError("start_date must be on or before end_date.")
        return self


class OutputConfig(BaseModel):
    """Output directory configuration for Phase 1."""

    root: Path = Path("outputs")
    rasters: Path = Path("outputs/rasters")
    vectors: Path = Path("outputs/vectors")
    maps: Path = Path("outputs/maps")
    reports: Path = Path("outputs/reports")
    statistics: Path = Path("outputs/statistics")
    manifests: Path = Path("outputs/manifests")
    exports: Path = Path("outputs/exports")
    map_theme: MapThemeName = "academic"

    @field_validator(
        "root",
        "rasters",
        "vectors",
        "maps",
        "reports",
        "statistics",
        "manifests",
        "exports",
        mode="before",
    )
    @classmethod
    def expand_output_paths(cls, value: object, info: ValidationInfo) -> object:
        """Normalize output path strings without requiring directories to exist."""
        if isinstance(value, str):
            logger.debug("Normalizing output path field {}", info.field_name)
            return Path(value).expanduser()
        return value

    def directories(self) -> tuple[Path, ...]:
        """Return all configured output directories."""
        return (
            self.root,
            self.rasters,
            self.vectors,
            self.maps,
            self.reports,
            self.statistics,
            self.manifests,
            self.exports,
        )


class ProcessingConfig(BaseModel):
    """Processing settings that can be validated before later phases exist."""

    target_crs: str = "EPSG:3857"
    tile_size: int = Field(default=1024, ge=128, le=16384)
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    max_workers: int = Field(default=1, ge=1, le=64)
    dry_run: bool = True


class RasterProcessingConfig(BaseModel):
    """Raster processing configuration for Phase 3 workflows."""

    target_crs: str = "EPSG:3857"
    chunk_size: int = Field(default=512, ge=64, le=16384)
    tile_size: int = Field(default=2048, ge=64, le=16384)
    resampling: Literal["nearest", "bilinear", "cubic"] = "bilinear"
    output_driver: Literal["COG", "GTiff"] = "COG"
    compress: Literal["deflate", "lzw", "zstd"] = "deflate"
    use_dask: bool = False
    max_workers: int = Field(default=2, ge=1, le=64)
    cloud_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    mask_value: float = Field(default=-9999.0)
    nodata_value: float = Field(default=-9999.0)


class LoggingConfig(BaseModel):
    """Log destination settings."""

    directory: Path = Path("logs")
    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    rotation: str = "10 MB"
    retention: str = "14 days"


class ProjectConfig(BaseModel):
    """Top-level GeoWatch project configuration."""

    project_name: str = Field(default="geowatch-project", min_length=1)
    aoi: AOIConfig
    dates: DateRangeConfig
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    raster_processing: RasterProcessingConfig = Field(
        default_factory=RasterProcessingConfig
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "json_schema_extra": {
            "examples": [
                {
                    "project_name": "lahore-foundation-example",
                    "aoi": {
                        "kind": "bbox",
                        "bbox": [74.15, 31.35, 74.55, 31.7],
                        "crs": "EPSG:4326",
                    },
                    "dates": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                }
            ]
        },
    }

    def output_directories(self) -> tuple[Path, ...]:
        """Return output directories used by the project."""
        return self.outputs.directories()
