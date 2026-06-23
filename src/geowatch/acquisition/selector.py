"""Automatic source selection and provider ranking."""

from __future__ import annotations

from datetime import date
from datetime import datetime as dt

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import HTTPClient
from geowatch.acquisition.models import (
    AcquisitionConfig,
    DatasetName,
    ProviderName,
    SceneMetadata,
)
from geowatch.acquisition.providers.base import ImageryProvider
from geowatch.acquisition.providers.copernicus import CopernicusProvider
from geowatch.acquisition.providers.nasa import NasaEarthdataProvider
from geowatch.acquisition.providers.planetary import PlanetaryComputerProvider
from geowatch.acquisition.providers.usgs import USGSProvider
from geowatch.acquisition.retry import RetryPolicy


def choose_datasets(
    start_date: date,
    configured: tuple[DatasetName, ...],
) -> tuple[DatasetName, ...]:
    """Choose appropriate datasets for the requested timeline."""
    if start_date.year >= 2015 and "sentinel-2-l2a" in configured:
        return ("sentinel-2-l2a",)
    landsat = tuple(dataset for dataset in configured if dataset.startswith("landsat"))
    return landsat or configured


def rank_providers(datasets: tuple[DatasetName, ...]) -> tuple[ProviderName, ...]:
    """Rank providers for selected datasets."""
    if datasets == ("sentinel-2-l2a",):
        return ("planetary-computer", "copernicus")
    return ("planetary-computer", "usgs", "nasa-earthdata")


def rank_scenes(
    scenes: tuple[SceneMetadata, ...],
    *,
    datasets: tuple[DatasetName, ...],
    aoi_bbox: tuple[float, float, float, float],
    temporal_midpoint: dt,
) -> tuple[SceneMetadata, ...]:
    """Filter exact missions and rank complete, clear, season-centred scenes."""
    matching = tuple(scene for scene in scenes if scene.dataset in datasets)
    return tuple(
        sorted(
            matching,
            key=lambda scene: (
                -scene_aoi_coverage(scene, aoi_bbox),
                scene.cloud_cover is None,
                scene.cloud_cover if scene.cloud_cover is not None else 101.0,
                _date_distance(scene.acquired_at, temporal_midpoint),
                scene.scene_id,
            ),
        )
    )


def scene_aoi_coverage(
    scene: SceneMetadata,
    aoi_bbox: tuple[float, float, float, float],
) -> float:
    """Estimate the fraction of an AOI bbox covered by a scene bbox."""
    scene_bbox = scene.bbox
    if scene_bbox is None:
        return 0.0
    sw, ss, se, sn = scene_bbox
    aw, ass, ae, an = aoi_bbox
    intersection = max(0.0, min(se, ae) - max(sw, aw)) * max(
        0.0, min(sn, an) - max(ss, ass)
    )
    area = max(0.0, ae - aw) * max(0.0, an - ass)
    return intersection / area if area else 0.0


def _date_distance(acquired_at: dt | None, midpoint: dt) -> float:
    if acquired_at is None:
        return float("inf")
    candidate = acquired_at.replace(tzinfo=None)
    return abs((candidate - midpoint.replace(tzinfo=None)).total_seconds())


def build_provider(
    provider: ProviderName,
    config: AcquisitionConfig,
    *,
    http_client: HTTPClient | None = None,
    auth_manager: AuthManager | None = None,
) -> ImageryProvider:
    """Instantiate a provider connector from configuration."""
    retry_policy = RetryPolicy(
        attempts=config.retry_attempts,
        backoff_seconds=config.retry_backoff_seconds,
    )
    if provider == "copernicus":
        return CopernicusProvider(
            http_client=http_client,
            timeout=config.request_timeout_seconds,
            retry_policy=retry_policy,
            auth_manager=auth_manager,
        )
    if provider == "planetary-computer":
        return PlanetaryComputerProvider(
            http_client=http_client,
            timeout=config.request_timeout_seconds,
            retry_policy=retry_policy,
            auth_manager=auth_manager,
        )
    if provider == "nasa-earthdata":
        return NasaEarthdataProvider(
            http_client=http_client,
            timeout=config.request_timeout_seconds,
            retry_policy=retry_policy,
            auth_manager=auth_manager,
        )
    return USGSProvider(
        http_client=http_client,
        timeout=config.request_timeout_seconds,
        retry_policy=retry_policy,
        auth_manager=auth_manager,
    )
