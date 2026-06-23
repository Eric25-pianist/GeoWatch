"""Administrative boundary validation and persistence tests."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon

from geowatch.application.boundaries import (
    BoundaryCandidate,
    render_boundary_preview,
    save_boundary_candidate,
    validate_candidate,
)


def test_boundary_validation_preview_and_provenance(tmp_path: Path) -> None:
    """Confirmed boundaries should be validated, previewed, and checksummed."""
    geometry = Polygon(((74.0, 31.0), (75.0, 31.0), (75.0, 32.0), (74.0, 32.0)))
    candidate = BoundaryCandidate(
        name="Test City",
        display_name="Test City, Test Country",
        country_code="tc",
        administrative_level="6",
        source="Test source",
        source_url="https://example.test/boundary",
        license="Test license",
        geometry=geometry,
        centroid=(74.5, 31.5),
        area_sq_km=10_000.0,
    )

    findings = validate_candidate(candidate, expected_country_code="TC")
    preview = render_boundary_preview(candidate, tmp_path / "preview.png")
    paths = save_boundary_candidate(
        candidate,
        source_path=tmp_path / "source.geojson",
        validated_path=tmp_path / "validated.geojson",
        metadata_path=tmp_path / "provenance.json",
    )

    assert "Geometry is valid and polygonal." in findings
    assert preview.exists()
    assert all(path.exists() for path in paths)


def test_boundary_preview_handles_non_ascii_names(tmp_path: Path) -> None:
    """Preview rendering should not depend on non-ASCII font glyph coverage."""
    geometry = Polygon(((139.6, 35.5), (139.9, 35.5), (139.9, 35.8), (139.6, 35.8)))
    candidate = BoundaryCandidate(
        name="東京都",
        display_name="東京都, 日本",
        country_code="jp",
        administrative_level="4",
        source="OpenStreetMap Nominatim",
        source_url="https://example.test/tokyo",
        license="ODbL",
        geometry=geometry,
        centroid=(139.75, 35.65),
        area_sq_km=100.0,
    )

    preview = render_boundary_preview(candidate, tmp_path / "tokyo_preview.png")

    assert preview.exists()
