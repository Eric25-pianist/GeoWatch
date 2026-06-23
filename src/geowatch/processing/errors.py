"""Raster processing error types."""

from __future__ import annotations

from geowatch.core.errors import GeoWatchError


class ProcessingError(GeoWatchError):
    """Raised when raster processing fails."""


class AlignmentError(ProcessingError):
    """Raised when raster alignment cannot be completed."""


class RasterDependencyError(ProcessingError):
    """Raised when an optional raster dependency is unavailable."""
