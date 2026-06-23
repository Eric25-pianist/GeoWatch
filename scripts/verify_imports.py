"""Verify that the GeoWatch package imports successfully."""

from __future__ import annotations

from geowatch.cli.app import app
from geowatch.config.models import ProjectConfig
from geowatch.logging.manager import LoggerManager
from geowatch.validation.checks import run_validation


def main() -> None:
    """Import public foundation symbols."""
    _ = (app, ProjectConfig, LoggerManager, run_validation)
    print("GeoWatch imports verified")


if __name__ == "__main__":
    main()
