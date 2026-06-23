"""Sentinel provider facade."""

from __future__ import annotations

from geowatch.acquisition.providers.copernicus import CopernicusProvider


class SentinelProvider(CopernicusProvider):
    """Sentinel-2 L2A connector using Copernicus Data Space by default."""
