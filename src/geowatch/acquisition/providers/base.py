"""Provider interfaces for GeoWatch acquisition connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from geowatch.acquisition.models import ProviderName, SceneMetadata, SearchQuery


class ImageryProvider(ABC):
    """Abstract imagery provider connector."""

    name: ProviderName

    @abstractmethod
    def search(self, query: SearchQuery) -> tuple[SceneMetadata, ...]:
        """Search imagery scenes for ``query``."""

    @abstractmethod
    def metadata(self, scene_id: str) -> SceneMetadata:
        """Fetch or return normalized metadata for a scene."""
