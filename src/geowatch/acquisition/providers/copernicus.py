"""Copernicus Data Space Ecosystem connector."""

from __future__ import annotations

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import HTTPClient
from geowatch.acquisition.providers.stac_provider import STACProvider
from geowatch.acquisition.retry import RetryPolicy

COPERNICUS_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"


class CopernicusProvider(STACProvider):
    """Search Sentinel data through the official Copernicus STAC API."""

    def __init__(
        self,
        *,
        http_client: HTTPClient | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        auth_manager: AuthManager | None = None,
    ) -> None:
        super().__init__(
            name="copernicus",
            base_url=COPERNICUS_STAC_URL,
            collection_map={"sentinel-2-l2a": "sentinel-2-l2a"},
            http_client=http_client,
            timeout=timeout,
            retry_policy=retry_policy,
            auth_manager=auth_manager,
        )
