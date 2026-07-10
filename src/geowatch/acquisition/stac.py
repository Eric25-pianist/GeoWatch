"""STAC search client and item normalization helpers."""

from __future__ import annotations

from datetime import datetime as dt
from typing import Any, cast

from loguru import logger

from geowatch.acquisition.http import (
    AcquisitionError,
    HTTPClient,
    NonRetryableAcquisitionError,
    UrllibHTTPClient,
)
from geowatch.acquisition.models import (
    AssetMetadata,
    ChecksumAlgorithm,
    DatasetName,
    ProviderName,
    SceneMetadata,
    SearchQuery,
)
from geowatch.acquisition.retry import RetryPolicy, retry_call


class STACClient:
    """Minimal STAC API client for provider search endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        provider: ProviderName,
        http_client: HTTPClient | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.http_client = http_client or UrllibHTTPClient()
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.headers = headers or {}

    def search(
        self,
        query: SearchQuery,
        *,
        collections: tuple[str, ...],
        dataset_map: dict[str, DatasetName],
    ) -> tuple[SceneMetadata, ...]:
        """Search a STAC API and return normalized scenes."""
        body = {
            "bbox": list(query.bbox),
            "datetime": query.datetime_range,
            "collections": list(collections),
            "limit": query.limit,
            "query": {"eo:cloud_cover": {"lte": query.max_cloud_cover}},
        }

        def action() -> tuple[SceneMetadata, ...]:
            response = self.http_client.request(
                "POST",
                f"{self.base_url}/search",
                headers=self.headers,
                json_body=body,
                timeout=self.timeout,
            )
            if not response.ok:
                msg = (
                    f"STAC search failed for {self.provider}: "
                    f"HTTP {response.status_code}"
                )
                if response.status_code not in self.retry_policy.retry_statuses:
                    raise NonRetryableAcquisitionError(msg)
                raise AcquisitionError(msg)
            payload = response.json()
            features = payload.get("features", [])
            if not isinstance(features, list):
                raise AcquisitionError("STAC response 'features' must be a list.")
            scenes = [
                normalize_stac_item(
                    item,
                    provider=self.provider,
                    dataset_map=dataset_map,
                )
                for item in features
                if isinstance(item, dict)
            ]
            logger.info(
                "STAC search returned {} scenes from {}",
                len(scenes),
                self.provider,
            )
            return tuple(scenes)

        return retry_call(action, self.retry_policy)


def normalize_stac_item(
    item: dict[str, object],
    *,
    provider: ProviderName,
    dataset_map: dict[str, DatasetName],
) -> SceneMetadata:
    """Convert a STAC item dictionary into ``SceneMetadata``."""
    properties = _dict_value(item, "properties")
    assets = _dict_value(item, "assets")
    collection = str(item.get("collection", ""))
    dataset = _infer_dataset(collection, properties, dataset_map)
    acquired_at = _parse_datetime(properties.get("datetime"))
    cloud_cover = _cloud_cover(properties)
    bbox = _bbox(item.get("bbox"))
    return SceneMetadata(
        scene_id=str(item.get("id", "")),
        provider=provider,
        dataset=dataset,
        acquired_at=acquired_at,
        bbox=bbox,
        cloud_cover=cloud_cover,
        assets=tuple(_normalize_assets(assets)),
        metadata=properties,
        source_url=_source_url(item),
    )


def _normalize_assets(assets: dict[str, object]) -> list[AssetMetadata]:
    normalized: list[AssetMetadata] = []
    for name, raw_asset in assets.items():
        if not isinstance(raw_asset, dict):
            continue
        href = raw_asset.get("href")
        if not isinstance(href, str) or not href:
            continue
        roles_value = raw_asset.get("roles", ())
        roles = (
            tuple(str(role) for role in roles_value)
            if isinstance(roles_value, list)
            else ()
        )
        checksum, algorithm = _checksum(raw_asset)
        size_value = raw_asset.get("file:size") or raw_asset.get("size")
        size = int(size_value) if isinstance(size_value, int | float) else None
        normalized.append(
            AssetMetadata(
                name=name,
                href=href,
                media_type=_optional_string(raw_asset.get("type")),
                roles=roles,
                checksum=checksum,
                checksum_algorithm=cast(ChecksumAlgorithm | None, algorithm),
                size=size,
            )
        )
    return normalized


def _dict_value(item: dict[str, object], key: str) -> dict[str, Any]:
    value = item.get(key, {})
    return value if isinstance(value, dict) else {}


def _parse_datetime(value: object) -> dt | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Could not parse STAC datetime {}", value)
        return None


def _cloud_cover(properties: dict[str, Any]) -> float | None:
    value = properties.get("eo:cloud_cover")
    if value is None:
        value = properties.get("landsat:cloud_cover_land")
    if isinstance(value, int | float):
        return float(value)
    return None


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if all(isinstance(number, int | float) for number in value):
        return tuple(float(number) for number in value)  # type: ignore[return-value]
    return None


def _checksum(raw_asset: dict[str, object]) -> tuple[str | None, str | None]:
    checksum_value = raw_asset.get("checksum:multihash") or raw_asset.get("checksum")
    if not isinstance(checksum_value, str):
        return None, None
    if checksum_value.startswith("sha256:"):
        return checksum_value.removeprefix("sha256:"), "sha256"
    if checksum_value.startswith("md5:"):
        return checksum_value.removeprefix("md5:"), "md5"
    return checksum_value, None


def _source_url(item: dict[str, object]) -> str | None:
    links = item.get("links", [])
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and link.get("rel") == "self":
                href = link.get("href")
                if isinstance(href, str):
                    return href
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _infer_dataset(
    collection: str,
    properties: dict[str, Any] | None = None,
    dataset_map: dict[str, DatasetName] | None = None,
) -> DatasetName:
    """Infer a dataset, including the mission inside generic Landsat collections."""
    lowered = collection.lower()
    if "sentinel" in lowered or "s2" in lowered:
        return "sentinel-2-l2a"
    metadata = properties or {}
    platform = str(metadata.get("platform", "")).lower()
    spacecraft = str(metadata.get("landsat:spacecraft_id", "")).lower()
    mission = f"{lowered} {platform} {spacecraft}"
    if "landsat-5" in mission or "landsat_5" in mission:
        return "landsat-5-c2-l2"
    if "landsat-7" in mission or "landsat_7" in mission:
        return "landsat-7-c2-l2"
    if "landsat-9" in mission or "landsat_9" in mission:
        return "landsat-9-c2-l2"
    if "landsat-8" in mission or "landsat_8" in mission:
        return "landsat-8-c2-l2"
    if dataset_map and collection in dataset_map:
        return dataset_map[collection]
    return "landsat-8-c2-l2"
