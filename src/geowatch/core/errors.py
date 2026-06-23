"""Typed exception hierarchy for GeoWatch foundation behavior."""

from __future__ import annotations


class GeoWatchError(Exception):
    """Base class for all project-specific errors."""


class ConfigurationError(GeoWatchError):
    """Raised when configuration loading or validation fails."""


class InitializationError(GeoWatchError):
    """Raised when project initialization cannot complete."""


class ValidationFailure(GeoWatchError):
    """Raised when one or more validation checks fail."""


class LoggingSetupError(GeoWatchError):
    """Raised when log destinations cannot be configured."""
