"""Error types for remote sensing analytics workflows."""

from __future__ import annotations

from geowatch.core.errors import GeoWatchError


class AnalyticsError(GeoWatchError):
    """Raised when an analytics workflow cannot continue."""


class IndexValidationError(AnalyticsError):
    """Raised when required spectral inputs are missing or malformed."""


class ChangeDetectionError(AnalyticsError):
    """Raised when a change detection method fails."""


class ThresholdingError(AnalyticsError):
    """Raised when threshold estimation fails."""


class ClassificationError(AnalyticsError):
    """Raised when land-cover classification fails."""
