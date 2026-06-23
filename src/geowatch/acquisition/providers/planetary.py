"""Microsoft Planetary Computer STAC connector."""

from __future__ import annotations

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import HTTPClient
from geowatch.acquisition.providers.stac_provider import STACProvider
from geowatch.acquisition.retry import RetryPolicy

PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class PlanetaryComputerProvider(STACProvider):
    """Search public STAC collections hosted by Microsoft Planetary Computer."""

    def __init__(
        self,
        *,
        http_client: HTTPClient | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        auth_manager: AuthManager | None = None,
    ) -> None:
        super().__init__(
            name="planetary-computer",
            base_url=PLANETARY_COMPUTER_STAC_URL,
            collection_map={
                "sentinel-2-l2a": "sentinel-2-l2a",
                "landsat-5-c2-l2": "landsat-c2-l2",
                "landsat-7-c2-l2": "landsat-c2-l2",
                "landsat-8-c2-l2": "landsat-c2-l2",
                "landsat-9-c2-l2": "landsat-c2-l2",
            },
            http_client=http_client,
            timeout=timeout,
            retry_policy=retry_policy,
            auth_manager=auth_manager,
        )
