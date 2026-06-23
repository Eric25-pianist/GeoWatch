"""Environment-based authentication support for acquisition providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

from loguru import logger

from geowatch.acquisition.models import ProviderName


@dataclass(frozen=True)
class ProviderCredentials:
    """Credentials available for a provider."""

    username: str | None = None
    password: str | None = None
    token: str | None = None

    @property
    def has_secret(self) -> bool:
        """Return whether any secret-bearing value is present."""
        return bool(self.token or (self.username and self.password))


class AuthManager:
    """Read provider credentials from environment variables."""

    ENV_PREFIXES: ClassVar[dict[ProviderName, str]] = {
        "copernicus": "GEOWATCH_COPERNICUS",
        "planetary-computer": "GEOWATCH_PLANETARY_COMPUTER",
        "nasa-earthdata": "GEOWATCH_NASA_EARTHDATA",
        "usgs": "GEOWATCH_USGS",
    }

    def credentials_for(self, provider: ProviderName) -> ProviderCredentials:
        """Return credentials for ``provider`` without logging secret values."""
        prefix = self.ENV_PREFIXES[provider]
        credentials = ProviderCredentials(
            username=os.getenv(f"{prefix}_USERNAME"),
            password=os.getenv(f"{prefix}_PASSWORD"),
            token=os.getenv(f"{prefix}_TOKEN"),
        )
        logger.debug("Credential presence for {}: {}", provider, credentials.has_secret)
        return credentials

    def authorization_headers(self, provider: ProviderName) -> dict[str, str]:
        """Return authorization headers when a bearer token is configured."""
        credentials = self.credentials_for(provider)
        if credentials.token:
            return {"Authorization": f"Bearer {credentials.token}"}
        return {}
