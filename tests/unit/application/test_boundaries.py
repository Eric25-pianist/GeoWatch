"""Administrative boundary validation and persistence tests."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon

from geowatch.application.boundaries import (
    BoundaryCandidate,
    _boundary_queries,
    boundary_warning_messages,
    render_boundary_preview,
    save_boundary_candidate,
    search_boundaries,
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


def test_dispersed_boundary_produces_confirmation_warnings() -> None:
    """Multipart island-style boundaries should not be accepted silently."""
    geometry = MultiPolygon(
        [
            Polygon(((139.6, 35.5), (139.9, 35.5), (139.9, 35.8), (139.6, 35.8))),
            Polygon(((153.9, 20.4), (154.1, 20.4), (154.1, 20.6), (153.9, 20.6))),
        ]
    )
    candidate = BoundaryCandidate(
        name="Tokyo",
        display_name="Tokyo Metropolis, Japan",
        country_code="jp",
        administrative_level="4",
        source="OpenStreetMap Nominatim",
        source_url="https://example.test/tokyo",
        license="ODbL",
        geometry=geometry,
        centroid=(146.85, 28.0),
        area_sq_km=2_200.0,
    )

    warnings = boundary_warning_messages(candidate, requested_kind="city")

    assert candidate.is_spatially_dispersed
    assert any("multipart" in warning for warning in warnings)
    assert any("state/prefecture-style" in warning for warning in warnings)


def test_urban_boundary_queries_include_tokyo_special_wards() -> None:
    """Urban-core searches should try Tokyo ward-style names before broad fallback."""
    queries = _boundary_queries("Tokyo", "Japan", None, "urban")

    assert "Tokyo special wards, Japan" in queries
    assert "Tokyo 23 special wards, Japan" in queries


def test_boundary_search_retries_district_query_for_point_result(
    monkeypatch: object,
) -> None:
    """Places whose city query is point-only should retry district-style names."""
    calls: list[str] = []

    def fake_get_json(url: str) -> object:
        calls.append(url)
        if "Lahore+District" not in url:
            return [
                {
                    "name": "Lahore",
                    "display_name": "Lahore, Punjab, Pakistan",
                    "geojson": {"type": "Point", "coordinates": [74.35, 31.55]},
                }
            ]
        return [
            {
                "name": "Lahore District",
                "display_name": "Lahore District, Punjab, Pakistan",
                "osm_type": "relation",
                "osm_id": 123,
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [74.0, 31.0],
                            [75.0, 31.0],
                            [75.0, 32.0],
                            [74.0, 32.0],
                            [74.0, 31.0],
                        ]
                    ],
                },
                "address": {"country_code": "pk"},
                "extratags": {"admin_level": "6"},
            }
        ]

    monkeypatch.setattr("geowatch.application.boundaries._get_json", fake_get_json)

    candidates = search_boundaries("Lahore", "Pakistan", "Punjab")

    assert len(candidates) == 1
    assert candidates[0].name == "Lahore District"
    assert any("Lahore+District" in call for call in calls)


def test_city_boundary_search_prefers_compact_municipal_candidate(
    monkeypatch: object,
) -> None:
    """City-intent searches should rank compact admin candidates above broad regions."""

    def fake_get_json(url: str) -> object:
        del url
        return [
            {
                "name": "Tokyo Building",
                "display_name": "Tokyo Building, Japan",
                "osm_type": "way",
                "osm_id": 99,
                "class": "building",
                "type": "yes",
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [139.70, 35.68],
                            [139.71, 35.68],
                            [139.71, 35.69],
                            [139.70, 35.69],
                            [139.70, 35.68],
                        ]
                    ],
                },
                "address": {"country_code": "jp"},
                "extratags": {},
            },
            {
                "name": "Tokyo",
                "display_name": "Tokyo Metropolis, Japan",
                "osm_type": "relation",
                "osm_id": 1,
                "geojson": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [
                            [
                                [139.6, 35.5],
                                [139.9, 35.5],
                                [139.9, 35.8],
                                [139.6, 35.8],
                                [139.6, 35.5],
                            ]
                        ],
                        [
                            [
                                [153.9, 20.4],
                                [154.1, 20.4],
                                [154.1, 20.6],
                                [153.9, 20.6],
                                [153.9, 20.4],
                            ]
                        ],
                    ],
                },
                "address": {"country_code": "jp"},
                "extratags": {"admin_level": "4", "boundary": "administrative"},
            },
            {
                "name": "Tokyo City",
                "display_name": "Tokyo City, Japan",
                "osm_type": "relation",
                "osm_id": 2,
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [139.6, 35.5],
                            [139.9, 35.5],
                            [139.9, 35.8],
                            [139.6, 35.8],
                            [139.6, 35.5],
                        ]
                    ],
                },
                "address": {"country_code": "jp"},
                "extratags": {"admin_level": "8", "boundary": "administrative"},
            },
        ]

    monkeypatch.setattr("geowatch.application.boundaries._get_json", fake_get_json)

    candidates = search_boundaries("Tokyo", "Japan", boundary_kind="city")

    assert candidates[0].name == "Tokyo City"
    assert all(candidate.name != "Tokyo Building" for candidate in candidates)
