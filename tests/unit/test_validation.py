"""Unit tests for validation checks."""

from __future__ import annotations

from pathlib import Path

from geowatch.config.models import AOIConfig
from geowatch.validation.checks import (
    run_validation,
    validate_aoi,
    validate_dependency,
    validate_python_version,
)


def test_python_version_validation_passes() -> None:
    """The active interpreter should satisfy project requirements."""
    messages = validate_python_version()

    assert messages[0].severity in {"info", "warning"}


def test_optional_missing_dependency_is_warning() -> None:
    """Optional dependency checks should not fail Phase 1 validation."""
    messages = validate_dependency("module_that_should_not_exist_123", required=False)

    assert messages[0].severity == "warning"


def test_required_missing_dependency_is_error() -> None:
    """Required dependency checks should be errors."""
    messages = validate_dependency("module_that_should_not_exist_123", required=True)

    assert messages[0].severity == "error"


def test_validate_geojson_aoi(sample_config_path: Path) -> None:
    """A committed sample GeoJSON AOI should validate."""
    aoi = AOIConfig(
        kind="geojson",
        path=Path("../tests/fixtures/sample_data/sample_aoi.geojson"),
        crs="EPSG:4326",
    )

    messages = validate_aoi(aoi, base_dir=sample_config_path.parent)

    assert all(message.severity != "error" for message in messages)


def test_run_validation_default_config(sample_config_path: Path) -> None:
    """Default validation should succeed without strict geospatial dependencies."""
    report = run_validation(sample_config_path)

    assert report.ok
    assert "GeoWatch validation report" in report.format_text()
