"""NASA Earthdata CMR-STAC connector."""

from __future__ import annotations

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import HTTPClient
from geowatch.acquisition.providers.stac_provider import STACProvider
from geowatch.acquisition.retry import RetryPolicy

NASA_CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"


class NasaEarthdataProvider(STACProvider):
    """Search NASA Earthdata CMR-STAC collections for Landsat scenes."""

    def __init__(
        self,
        *,
        http_client: HTTPClient | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        auth_manager: AuthManager | None = None,
    ) -> None:
        super().__init__(
            name="nasa-earthdata",
            base_url=NASA_CMR_STAC_URL,
            collection_map={
                "landsat-8-c2-l2": "LANDSAT_8_C2_L2",
                "landsat-9-c2-l2": "LANDSAT_9_C2_L2",
            },
            http_client=http_client,
            timeout=timeout,
            retry_policy=retry_policy,
            auth_manager=auth_manager,
        )
