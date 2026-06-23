"""Landsat provider facade."""

from __future__ import annotations

from geowatch.acquisition.providers.usgs import USGSProvider


class LandsatProvider(USGSProvider):
    """Landsat Collection 2 Level 2 connector using USGS by default."""
