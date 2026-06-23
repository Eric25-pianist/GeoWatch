"""USGS M2M connector for official Landsat search metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from loguru import logger

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import AcquisitionError, HTTPClient, UrllibHTTPClient
from geowatch.acquisition.models import (
    DatasetName,
    ProviderName,
    SceneMetadata,
    SearchQuery,
)
from geowatch.acquisition.providers.base import ImageryProvider
from geowatch.acquisition.retry import RetryPolicy, retry_call

USGS_M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"


class USGSProvider(ImageryProvider):
    """Search Landsat Collection 2 metadata through the USGS M2M API."""

    name: ProviderName = "usgs"
    DATASET_NAMES: ClassVar[dict[DatasetName, str]] = {
        "landsat-8-c2-l2": "landsat_ot_c2_l2",
        "landsat-9-c2-l2": "landsat_ot_c2_l2",
    }

    def __init__(
        self,
        *,
        http_client: HTTPClient | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        auth_manager: AuthManager | None = None,
    ) -> None:
        self.http_client = http_client or UrllibHTTPClient()
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.headers = (auth_manager or AuthManager()).authorization_headers(self.name)
        self._cache: dict[str, SceneMetadata] = {}

    def search(self, query: SearchQuery) -> tuple[SceneMetadata, ...]:
        """Search USGS M2M scene metadata for supported Landsat datasets."""
        scenes: list[SceneMetadata] = []
        for dataset in query.datasets:
            dataset_name = self.DATASET_NAMES.get(dataset)
            if dataset_name is None:
                continue
            scenes.extend(self._search_dataset(query, dataset, dataset_name))
        limited = tuple(scenes[: query.limit])
        self._cache.update({scene.scene_id: scene for scene in limited})
        logger.info("USGS search returned {} scenes", len(limited))
        return limited

    def metadata(self, scene_id: str) -> SceneMetadata:
        """Return cached metadata from a previous search."""
        if scene_id not in self._cache:
            raise AcquisitionError(f"Scene {scene_id} has not been searched by USGS.")
        return self._cache[scene_id]

    def _search_dataset(
        self,
        query: SearchQuery,
        dataset: DatasetName,
        dataset_name: str,
    ) -> tuple[SceneMetadata, ...]:
        west, south, east, north = query.bbox
        body = {
            "datasetName": dataset_name,
            "maxResults": query.limit,
            "sceneFilter": {
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {"latitude": south, "longitude": west},
                    "upperRight": {"latitude": north, "longitude": east},
                },
                "acquisitionFilter": {
                    "start": query.start_date.isoformat(),
                    "end": query.end_date.isoformat(),
                },
                "cloudCoverFilter": {
                    "min": 0,
                    "max": query.max_cloud_cover,
                    "includeUnknown": True,
                },
            },
        }

        def action() -> tuple[SceneMetadata, ...]:
            response = self.http_client.request(
                "POST",
                f"{USGS_M2M_URL}/scene-search",
                headers=self.headers,
                json_body=body,
                timeout=self.timeout,
            )
            if not response.ok:
                msg = f"USGS search failed: HTTP {response.status_code}"
                raise AcquisitionError(msg)
            payload = response.json()
            return tuple(_normalize_usgs_results(payload, dataset))

        return retry_call(action, self.retry_policy)


def _normalize_usgs_results(
    payload: dict[str, object],
    dataset: DatasetName,
) -> list[SceneMetadata]:
    data = payload.get("data", {})
    results = data.get("results", []) if isinstance(data, dict) else []
    if not isinstance(results, list):
        raise AcquisitionError("USGS response data.results must be a list.")
    scenes: list[SceneMetadata] = []
    for result in results:
        if isinstance(result, dict):
            scenes.append(_normalize_usgs_scene(result, dataset))
    return scenes


def _normalize_usgs_scene(
    result: dict[str, Any],
    dataset: DatasetName,
) -> SceneMetadata:
    scene_id = str(
        result.get("entityId") or result.get("displayId") or result.get("id")
    )
    acquisition_date = _parse_usgs_datetime(result.get("acquisitionDate"))
    spatial_bounds = result.get("spatialBounds")
    return SceneMetadata(
        scene_id=scene_id,
        provider="usgs",
        dataset=dataset,
        acquired_at=acquisition_date,
        bbox=_usgs_bbox(spatial_bounds),
        cloud_cover=_float_or_none(result.get("cloudCover")),
        assets=(),
        metadata=dict(result),
        source_url=None,
    )


def _parse_usgs_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _usgs_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    west = _float_or_none(value.get("minX") or value.get("west"))
    south = _float_or_none(value.get("minY") or value.get("south"))
    east = _float_or_none(value.get("maxX") or value.get("east"))
    north = _float_or_none(value.get("maxY") or value.get("north"))
    if None in {west, south, east, north}:
        return None
    return (west, south, east, north)  # type: ignore[return-value]
