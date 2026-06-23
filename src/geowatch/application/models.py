"""Versioned configuration models for terminal-operated GeoWatch projects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from geowatch.analytics.models import ANALYTICS_INDEX_NAMES
from geowatch.cartography.themes import MapThemeName

TemporalMode = Literal["endpoints", "annual", "interval"]
SensorPreference = Literal["auto", "landsat", "sentinel-2"]
ProviderPreference = Literal["auto", "planetary-computer", "usgs", "copernicus"]
ClassificationMethod = Literal[
    "kmeans", "isodata", "random_forest", "xgboost", "svm", "none"
]


class LocationSpec(BaseModel):
    """Human and spatial identity of a requested area of interest."""

    name: str = Field(min_length=1)
    country: str = Field(min_length=1)
    region: str | None = None
    administrative_level: str | None = None
    boundary_path: Path | None = None
    boundary_source: str | None = None
    boundary_source_url: str | None = None
    boundary_license: str | None = None


class TemporalSpec(BaseModel):
    """Comparison years and common seasonal window."""

    start_year: int = Field(ge=1984, le=2100)
    end_year: int = Field(ge=1984, le=2100)
    start_month: int = Field(default=6, ge=1, le=12)
    end_month: int = Field(default=9, ge=1, le=12)
    mode: TemporalMode = "endpoints"
    interval_years: int = Field(default=1, ge=1, le=50)

    @model_validator(mode="after")
    def validate_period(self) -> TemporalSpec:
        """Require ordered years and a non-wrapping seasonal window."""
        if self.start_year >= self.end_year:
            raise ValueError("end_year must be later than start_year")
        if self.start_month > self.end_month:
            raise ValueError("start_month must not be later than end_month")
        return self

    def years(self) -> tuple[int, ...]:
        """Return the years selected by the temporal strategy."""
        if self.mode == "endpoints":
            return self.start_year, self.end_year
        step = 1 if self.mode == "annual" else self.interval_years
        years = list(range(self.start_year, self.end_year + 1, step))
        if years[-1] != self.end_year:
            years.append(self.end_year)
        return tuple(years)


class ImagerySpec(BaseModel):
    """Imagery search, download, and compositing preferences."""

    sensor: SensorPreference = "auto"
    provider: ProviderPreference = "auto"
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    max_scenes_per_year: int = Field(default=3, ge=1, le=20)
    composite_method: Literal["median", "mean", "first"] = "median"


class AnalysisSpec(BaseModel):
    """Index, change detection, and classification settings."""

    indices: tuple[str, ...] = ANALYTICS_INDEX_NAMES
    change_methods: tuple[str, ...] = (
        "index_differencing",
        "cva",
        "pca",
        "mad",
        "irmad",
        "image_ratioing",
        "magnitude",
    )
    classification: ClassificationMethod = "kmeans"
    training_data: Path | None = None

    @model_validator(mode="after")
    def validate_analysis(self) -> AnalysisSpec:
        """Validate index names and supervised training requirements."""
        unknown = sorted(set(self.indices) - set(ANALYTICS_INDEX_NAMES))
        if unknown:
            raise ValueError(f"Unsupported indices: {', '.join(unknown)}")
        if (
            self.classification in {"random_forest", "xgboost", "svm"}
            and self.training_data is None
        ):
            raise ValueError(f"{self.classification} requires labeled training_data")
        return self


class OutputSpec(BaseModel):
    """Publication and execution output settings."""

    root: Path = Path("outputs")
    formats: tuple[Literal["png", "jpeg", "pdf", "svg"], ...] = (
        "png",
        "jpeg",
        "pdf",
    )
    dpi: tuple[Literal[300, 600], ...] = (300, 600)
    target_crs: str = "auto"
    max_workers: int = Field(default=2, ge=1, le=64)
    map_theme: MapThemeName = "academic"


class RunSpecification(BaseModel):
    """Complete, serializable contract for one GeoWatch project."""

    schema_version: Literal["1.0"] = "1.0"
    location: LocationSpec
    temporal: TemporalSpec
    imagery: ImagerySpec = Field(default_factory=ImagerySpec)
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)
    outputs: OutputSpec = Field(default_factory=OutputSpec)

    def log_summary(self) -> None:
        """Log a concise run summary without exposing credentials."""
        logger.info(
            "GeoWatch run: {}, {} | years={} | sensor={} | provider={}",
            self.location.name,
            self.location.country,
            self.temporal.years(),
            self.imagery.sensor,
            self.imagery.provider,
        )
