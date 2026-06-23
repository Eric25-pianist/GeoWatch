"""Phase 2 acquisition orchestration."""

from __future__ import annotations

import json
from datetime import datetime as dt
from pathlib import Path

import geopandas as gpd
from loguru import logger

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.catalog import (
    write_acquisition_report,
    write_metadata_catalog,
)
from geowatch.acquisition.download import DownloadManager, build_download_requests
from geowatch.acquisition.http import AcquisitionError, HTTPClient
from geowatch.acquisition.models import (
    AcquisitionResult,
    DownloadResult,
    SceneMetadata,
    SearchQuery,
)
from geowatch.acquisition.selector import (
    build_provider,
    choose_datasets,
    rank_providers,
    rank_scenes,
)
from geowatch.config.models import ProjectConfig


def run_acquisition(
    config: ProjectConfig,
    *,
    base_dir: Path | None = None,
    http_client: HTTPClient | None = None,
    auth_manager: AuthManager | None = None,
) -> AcquisitionResult:
    """Search imagery, optionally download assets, and write Phase 2 outputs."""
    if not config.acquisition.enabled:
        raise AcquisitionError("Acquisition is disabled in the configuration.")

    datasets = choose_datasets(config.dates.start_date, config.acquisition.datasets)
    provider_names = (
        (config.acquisition.provider,)
        if config.acquisition.provider != "auto"
        else rank_providers(datasets)
    )
    bbox = _resolve_query_bbox(config, base_dir=base_dir)
    query = SearchQuery(
        bbox=bbox,
        start_date=config.dates.start_date,
        end_date=config.dates.end_date,
        datasets=datasets,
        max_cloud_cover=config.acquisition.max_cloud_cover,
        limit=config.acquisition.max_results,
    )

    last_error: Exception | None = None
    for provider_name in provider_names:
        try:
            provider = build_provider(
                provider_name,
                config.acquisition,
                http_client=http_client,
                auth_manager=auth_manager,
            )
            scenes = rank_scenes(
                provider.search(query),
                datasets=datasets,
                aoi_bbox=bbox,
                temporal_midpoint=dt.combine(
                    config.dates.start_date
                    + ((config.dates.end_date - config.dates.start_date) / 2),
                    dt.min.time(),
                ),
            )
            if config.acquisition.selected_scene_ids:
                selected = set(config.acquisition.selected_scene_ids)
                scenes = tuple(scene for scene in scenes if scene.scene_id in selected)
            _require_scenes(scenes, provider.name, datasets)
            downloads = _download_if_requested(config, scenes, http_client=http_client)
            catalog_path = write_metadata_catalog(
                scenes,
                downloads,
                config.acquisition.metadata_catalog,
                provider=provider.name,
            )
            report_path = write_acquisition_report(
                scenes,
                downloads,
                config.acquisition.acquisition_report,
                provider=provider.name,
            )
            logger.bind(channel="pipeline").info(
                "Acquisition completed with {} scenes and {} downloads.",
                len(scenes),
                len(downloads),
            )
            return AcquisitionResult(
                provider=provider.name,
                scenes=scenes,
                downloads=downloads,
                catalog_path=catalog_path,
                report_path=report_path,
            )
        except AcquisitionError as exc:
            last_error = exc
            logger.warning("Provider {} failed: {}", provider_name, exc)
            continue
    raise AcquisitionError("All acquisition providers failed.") from last_error


def _download_if_requested(
    config: ProjectConfig,
    scenes: tuple[SceneMetadata, ...],
    *,
    http_client: HTTPClient | None,
) -> tuple[DownloadResult, ...]:
    """Download scene assets when configured."""
    if not config.acquisition.download:
        return ()
    requests = build_download_requests(
        scenes,
        download_directory=config.acquisition.download_directory,
        preferred_roles=config.acquisition.preferred_asset_roles,
        max_downloads=config.acquisition.max_downloads,
        max_bytes=config.acquisition.max_download_bytes,
    )
    manager = DownloadManager(
        http_client=http_client,
        timeout=config.acquisition.request_timeout_seconds,
    )
    return tuple(manager.download(request) for request in requests)


def _resolve_query_bbox(
    config: ProjectConfig,
    *,
    base_dir: Path | None,
) -> tuple[float, float, float, float]:
    """Resolve the acquisition search bbox from a bbox or vector AOI."""
    if config.aoi.bbox is not None:
        return config.aoi.bbox
    if config.aoi.path is None:
        raise AcquisitionError("AOI path is required when bbox coordinates are absent.")

    path = config.aoi.path
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    if not path.exists():
        raise AcquisitionError(f"AOI file does not exist: {path}")

    try:
        if path.suffix.lower() in {".geojson", ".json"}:
            return _geojson_bbox(path)
        frame = gpd.read_file(path)
        bounds = frame.total_bounds
    except Exception as exc:  # pragma: no cover - depends on optional vector readers
        logger.exception("Failed to resolve AOI bbox from {}", path)
        raise AcquisitionError(f"Could not read AOI vector file: {path}") from exc

    west, south, east, north = (float(value) for value in bounds)
    if west >= east or south >= north:
        raise AcquisitionError(f"Invalid AOI extent derived from {path}")
    return west, south, east, north


def _geojson_bbox(path: Path) -> tuple[float, float, float, float]:
    """Derive a bbox from a GeoJSON AOI file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    coordinates = _collect_geojson_coordinates(data)
    if not coordinates:
        raise AcquisitionError(f"GeoJSON AOI does not contain coordinates: {path}")
    xs = [float(point[0]) for point in coordinates]
    ys = [float(point[1]) for point in coordinates]
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)
    if west >= east or south >= north:
        raise AcquisitionError(f"Invalid GeoJSON AOI extent: {path}")
    return west, south, east, north


def _collect_geojson_coordinates(value: object) -> list[tuple[float, float]]:
    """Collect coordinate pairs from a GeoJSON-like object."""
    coordinates: list[tuple[float, float]] = []
    if isinstance(value, dict):
        if value.get("type") == "FeatureCollection":
            for feature in value.get("features", []):
                coordinates.extend(_collect_geojson_coordinates(feature))
        elif value.get("type") == "Feature":
            coordinates.extend(_collect_geojson_coordinates(value.get("geometry")))
        elif value.get("type") in {"Polygon", "MultiPolygon", "LineString", "Point"}:
            coordinates.extend(_collect_geojson_coordinates(value.get("coordinates")))
    elif isinstance(value, list):
        if value and isinstance(value[0], (int, float)) and len(value) >= 2:
            coordinates.append((float(value[0]), float(value[1])))
        else:
            for item in value:
                coordinates.extend(_collect_geojson_coordinates(item))
    return coordinates


def _require_scenes(
    scenes: tuple[SceneMetadata, ...],
    provider_name: str,
    datasets: object,
) -> None:
    """Require exact mission-matching scenes from a provider."""
    if not scenes:
        raise AcquisitionError(
            f"Provider {provider_name} returned no matching {datasets} scenes."
        )
