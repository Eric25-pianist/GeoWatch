"""Reusable STAC provider implementation."""

from __future__ import annotations

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import AcquisitionError, HTTPClient
from geowatch.acquisition.models import (
    DatasetName,
    ProviderName,
    SceneMetadata,
    SearchQuery,
)
from geowatch.acquisition.providers.base import ImageryProvider
from geowatch.acquisition.retry import RetryPolicy
from geowatch.acquisition.stac import STACClient


class STACProvider(ImageryProvider):
    """Provider backed by a STAC API search endpoint."""

    def __init__(
        self,
        *,
        name: ProviderName,
        base_url: str,
        collection_map: dict[DatasetName, str],
        http_client: HTTPClient | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        auth_manager: AuthManager | None = None,
    ) -> None:
        self.name = name
        self.collection_map = collection_map
        self.dataset_map = {
            collection: dataset for dataset, collection in collection_map.items()
        }
        headers = (auth_manager or AuthManager()).authorization_headers(name)
        self.client = STACClient(
            base_url,
            provider=name,
            http_client=http_client,
            timeout=timeout,
            retry_policy=retry_policy,
            headers=headers,
        )
        self._cache: dict[str, SceneMetadata] = {}

    def search(self, query: SearchQuery) -> tuple[SceneMetadata, ...]:
        """Search provider collections for query datasets."""
        collections = tuple(
            dict.fromkeys(
                self.collection_map[dataset]
                for dataset in query.datasets
                if dataset in self.collection_map
            )
        )
        scenes = self.client.search(
            query,
            collections=collections,
            dataset_map=self.dataset_map,
        )
        self._cache.update({scene.scene_id: scene for scene in scenes})
        return scenes

    def metadata(self, scene_id: str) -> SceneMetadata:
        """Return metadata from the latest search result cache."""
        if scene_id not in self._cache:
            msg = f"Scene {scene_id} has not been searched by {self.name}."
            raise AcquisitionError(msg)
        return self._cache[scene_id]
