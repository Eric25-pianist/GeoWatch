"""Provider connectors for GeoWatch acquisition."""

from __future__ import annotations

from geowatch.acquisition.providers.copernicus import CopernicusProvider
from geowatch.acquisition.providers.landsat import LandsatProvider
from geowatch.acquisition.providers.nasa import NasaEarthdataProvider
from geowatch.acquisition.providers.planetary import PlanetaryComputerProvider
from geowatch.acquisition.providers.sentinel import SentinelProvider
from geowatch.acquisition.providers.usgs import USGSProvider

__all__ = [
    "CopernicusProvider",
    "LandsatProvider",
    "NasaEarthdataProvider",
    "PlanetaryComputerProvider",
    "SentinelProvider",
    "USGSProvider",
]
