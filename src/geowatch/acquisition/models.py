"""Typed acquisition models for scenes, assets, downloads, and configuration."""

from __future__ import annotations

from datetime import date
from datetime import datetime as dt
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

DatasetName = Literal[
    "sentinel-2-l2a",
    "landsat-5-c2-l2",
    "landsat-7-c2-l2",
    "landsat-8-c2-l2",
    "landsat-9-c2-l2",
]
ProviderName = Literal[
    "copernicus",
    "planetary-computer",
    "nasa-earthdata",
    "usgs",
]
ChecksumAlgorithm = Literal["sha256", "md5"]


class AcquisitionConfig(BaseModel):
    """Data acquisition settings shared by all Phase 2 providers."""

    enabled: bool = True
    download: bool = False
    provider: ProviderName | Literal["auto"] = "auto"
    datasets: tuple[DatasetName, ...] = (
        "sentinel-2-l2a",
        "landsat-5-c2-l2",
        "landsat-7-c2-l2",
        "landsat-8-c2-l2",
        "landsat-9-c2-l2",
    )
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    max_results: int = Field(default=10, ge=1, le=500)
    max_downloads: int = Field(default=1, ge=1, le=100)
    download_directory: Path = Path("data/raw")
    metadata_catalog: Path = Path("outputs/manifests/acquisition_catalog.json")
    acquisition_report: Path = Path("outputs/reports/acquisition_report.md")
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    max_download_bytes: int = Field(default=2_147_483_648, ge=1)
    preferred_asset_roles: tuple[str, ...] = ("data", "visual", "thumbnail")
    selected_scene_ids: tuple[str, ...] = ()

    @field_validator(
        "download_directory",
        "metadata_catalog",
        "acquisition_report",
        mode="before",
    )
    @classmethod
    def normalize_paths(cls, value: object) -> object:
        """Normalize acquisition path fields while preserving relative paths."""
        if isinstance(value, str):
            logger.debug("Normalizing acquisition path {}", value)
            return Path(value).expanduser()
        return value


class SearchQuery(BaseModel):
    """Provider-neutral scene search request."""

    bbox: tuple[float, float, float, float]
    start_date: date
    end_date: date
    datasets: tuple[DatasetName, ...]
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    limit: int = Field(default=10, ge=1, le=500)

    @model_validator(mode="after")
    def validate_query(self) -> SearchQuery:
        """Validate spatial and temporal query constraints."""
        west, south, east, north = self.bbox
        if west >= east or south >= north:
            raise ValueError("Search bbox must be ordered west, south, east, north.")
        if self.start_date > self.end_date:
            raise ValueError("Search start_date must be on or before end_date.")
        return self

    @property
    def datetime_range(self) -> str:
        """Return a STAC-compatible datetime interval."""
        return f"{self.start_date.isoformat()}/{self.end_date.isoformat()}"


class AssetMetadata(BaseModel):
    """Downloadable scene asset metadata."""

    name: str
    href: str
    media_type: str | None = None
    roles: tuple[str, ...] = ()
    checksum: str | None = None
    checksum_algorithm: ChecksumAlgorithm | None = None
    size: int | None = Field(default=None, ge=0)


class SceneMetadata(BaseModel):
    """Normalized scene metadata returned by any acquisition provider."""

    scene_id: str
    provider: ProviderName
    dataset: DatasetName
    acquired_at: dt | None = None
    bbox: tuple[float, float, float, float] | None = None
    cloud_cover: float | None = Field(default=None, ge=0.0, le=100.0)
    assets: tuple[AssetMetadata, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)
    source_url: str | None = None

    def preferred_assets(self, roles: tuple[str, ...]) -> tuple[AssetMetadata, ...]:
        """Return assets matching preferred roles, or all assets if none match."""
        matches: list[AssetMetadata] = []
        seen: set[tuple[str, str]] = set()
        for role in roles:
            for asset in self.assets:
                key = (asset.name, str(asset.href))
                if key not in seen and (role in asset.roles or asset.name == role):
                    matches.append(asset)
                    seen.add(key)
        if matches:
            return tuple(matches)
        return self.assets


class DownloadRequest(BaseModel):
    """Single asset download request."""

    scene: SceneMetadata
    asset: AssetMetadata
    destination: Path
    max_bytes: int = Field(default=2_147_483_648, ge=1)


class DownloadResult(BaseModel):
    """Result of a verified download."""

    scene_id: str
    asset_name: str
    path: Path
    bytes_written: int
    checksum: str | None = None
    verified: bool


class AcquisitionResult(BaseModel):
    """Phase 2 acquisition orchestration result."""

    provider: ProviderName
    scenes: tuple[SceneMetadata, ...]
    downloads: tuple[DownloadResult, ...]
    catalog_path: Path
    report_path: Path
