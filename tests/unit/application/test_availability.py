"""All-years imagery availability policy tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from geowatch.acquisition.http import AcquisitionError
from geowatch.acquisition.models import SceneMetadata
from geowatch.application.availability import build_availability_plan
from geowatch.application.models import (
    ImagerySpec,
    LocationSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.sensors import LANDSAT_7, SENTINEL_2
from geowatch.core.errors import GeoWatchError


def test_availability_uses_one_common_fallback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """All years should share an approved expanded window and cloud ceiling."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[-1,51],[0,51],[0,52],[-1,52],[-1,51]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(name="London", country="England", boundary_path=boundary),
        temporal=TemporalSpec(
            start_year=2010,
            end_year=2012,
            start_month=6,
            end_month=8,
            mode="annual",
        ),
        imagery=ImagerySpec(
            sensor="landsat", max_cloud_cover=20.0, max_scenes_per_year=3
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            enough = query.start_date.month == 5 and query.max_cloud_cover >= 40
            count = 3 if enough else 1
            return tuple(
                SceneMetadata(
                    scene_id=f"LE07-{year}-{index}",
                    provider="planetary-computer",
                    dataset="landsat-7-c2-l2",
                    acquired_at=datetime(year, 7, index + 1, tzinfo=UTC),
                    bbox=(-2.0, 50.0, 1.0, 53.0),
                    cloud_cover=10.0 + index,
                    assets=(),
                )
                for index in range(count)
            )

    def fake_build_provider(*_args: Any, **_kwargs: Any) -> FakeProvider:
        return FakeProvider()

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider", fake_build_provider
    )
    plan = build_availability_plan(spec, boundary, LANDSAT_7)

    assert (plan.effective_start_month, plan.effective_end_month) == (5, 9)
    assert plan.effective_cloud_cover == 40.0
    assert plan.used_fallback
    assert all(item.scene_count == 3 for item in plan.years.values())


def test_availability_selects_same_day_tiles_for_large_aoi(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Multiple Sentinel tiles should be accepted when their union covers the AOI."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Wide City",
            country="Test Country",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2018,
            end_year=2019,
            start_month=3,
            end_month=5,
        ),
        imagery=ImagerySpec(
            sensor="sentinel-2",
            max_cloud_cover=20.0,
            max_scenes_per_year=1,
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"S2-{year}-west",
                    provider="planetary-computer",
                    dataset="sentinel-2-l2a",
                    acquired_at=datetime(year, 4, 15, tzinfo=UTC),
                    bbox=(0.0, 0.0, 1.1, 1.0),
                    cloud_cover=2.0,
                    assets=(),
                ),
                SceneMetadata(
                    scene_id=f"S2-{year}-east",
                    provider="planetary-computer",
                    dataset="sentinel-2-l2a",
                    acquired_at=datetime(year, 4, 15, tzinfo=UTC),
                    bbox=(0.9, 0.0, 2.0, 1.0),
                    cloud_cover=3.0,
                    assets=(),
                ),
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )
    plan = build_availability_plan(spec, boundary, SENTINEL_2)

    assert all(item.scene_count == 2 for item in plan.years.values())
    assert all(item.aoi_coverage == 1.0 for item in plan.years.values())
    assert "expanded the scene allowance" in plan.fallback_messages[0]


def test_availability_accepts_seasonal_multidate_mosaic(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Large AOIs may be covered by a seasonal mosaic rather than one same-day set."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Mountain District",
            country="Test Country",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2018,
            end_year=2019,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(
            sensor="landsat",
            max_cloud_cover=20.0,
            max_scenes_per_year=1,
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"LC08-{year}-west",
                    provider="planetary-computer",
                    dataset="landsat-8-c2-l2",
                    acquired_at=datetime(year, 7, 1, tzinfo=UTC),
                    bbox=(0.0, 0.0, 1.15, 1.0),
                    cloud_cover=8.0,
                    assets=(),
                ),
                SceneMetadata(
                    scene_id=f"LC08-{year}-east",
                    provider="planetary-computer",
                    dataset="landsat-8-c2-l2",
                    acquired_at=datetime(year, 7, 20, tzinfo=UTC),
                    bbox=(0.85, 0.0, 2.0, 1.0),
                    cloud_cover=9.0,
                    assets=(),
                ),
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )
    from geowatch.application.sensors import LANDSAT_8

    plan = build_availability_plan(spec, boundary, LANDSAT_8)

    assert all(item.scene_count == 2 for item in plan.years.values())
    assert all(item.aoi_coverage == 1.0 for item in plan.years.values())
    assert "expanded the scene allowance" in "; ".join(plan.fallback_messages)


def test_full_year_fallback_prefers_season_center_over_low_winter_cloud(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Expanded full-year searches should not prefer snowy low-cloud winter scenes."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Mountain District",
            country="Test Country",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2019,
            end_year=2020,
            start_month=12,
            end_month=12,
        ),
        imagery=ImagerySpec(
            sensor="landsat",
            max_cloud_cover=20.0,
            max_scenes_per_year=3,
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            if (
                query.start_date.month,
                query.end_date.month,
                int(query.max_cloud_cover),
            ) != (1, 12, 40):
                return ()
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"LC08-{year}-winter-west",
                    provider="planetary-computer",
                    dataset="landsat-8-c2-l2",
                    acquired_at=datetime(year, 12, 29, tzinfo=UTC),
                    bbox=(0.0, 0.0, 1.1, 1.0),
                    cloud_cover=1.0,
                    assets=(),
                ),
                SceneMetadata(
                    scene_id=f"LC08-{year}-winter-east",
                    provider="planetary-computer",
                    dataset="landsat-8-c2-l2",
                    acquired_at=datetime(year, 12, 29, tzinfo=UTC),
                    bbox=(0.9, 0.0, 2.0, 1.0),
                    cloud_cover=2.0,
                    assets=(),
                ),
                SceneMetadata(
                    scene_id=f"LC08-{year}-summer-west",
                    provider="planetary-computer",
                    dataset="landsat-8-c2-l2",
                    acquired_at=datetime(year, 7, 15, tzinfo=UTC),
                    bbox=(0.0, 0.0, 1.1, 1.0),
                    cloud_cover=30.0,
                    assets=(),
                ),
                SceneMetadata(
                    scene_id=f"LC08-{year}-summer-east",
                    provider="planetary-computer",
                    dataset="landsat-8-c2-l2",
                    acquired_at=datetime(year, 7, 15, tzinfo=UTC),
                    bbox=(0.9, 0.0, 2.0, 1.0),
                    cloud_cover=35.0,
                    assets=(),
                ),
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )
    from geowatch.application.sensors import LANDSAT_8

    plan = build_availability_plan(spec, boundary, LANDSAT_8)

    assert all(
        item.acquired_dates == ("2019-07-15",) * 2
        if year == 2019
        else item.acquired_dates == ("2020-07-15",) * 2
        for year, item in plan.years.items()
    )
    assert "seasonal fallback expanded to the full year" in "; ".join(
        plan.fallback_messages
    )


def test_landsat7_gap_fill_selects_multiple_dates(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Post-2003 Landsat 7 should plan multi-date mosaics for SLC-off gaps."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="SLC Gap Area",
            country="Test",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2011,
            end_year=2012,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(
            sensor="landsat",
            max_cloud_cover=20.0,
            max_scenes_per_year=3,
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            scenes: list[SceneMetadata] = []
            for month, day in ((6, 10), (7, 12), (8, 14)):
                scenes.extend(
                    (
                        SceneMetadata(
                            scene_id=f"LE07-{year}-{month:02d}{day:02d}-west",
                            provider="planetary-computer",
                            dataset="landsat-7-c2-l2",
                            acquired_at=datetime(year, month, day, tzinfo=UTC),
                            bbox=(0.0, 0.0, 1.15, 1.0),
                            cloud_cover=5.0,
                            assets=(),
                        ),
                        SceneMetadata(
                            scene_id=f"LE07-{year}-{month:02d}{day:02d}-east",
                            provider="planetary-computer",
                            dataset="landsat-7-c2-l2",
                            acquired_at=datetime(year, month, day, tzinfo=UTC),
                            bbox=(0.85, 0.0, 2.0, 1.0),
                            cloud_cover=6.0,
                            assets=(),
                        ),
                    )
                )
            return tuple(scenes)

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )

    plan = build_availability_plan(spec, boundary, LANDSAT_7)

    assert all(item.scene_count == 6 for item in plan.years.values())
    assert all(
        item.estimated_valid_coverage is not None
        and item.estimated_valid_coverage >= 0.65
        for item in plan.years.values()
    )
    assert "SLC-off gap-fill" in "; ".join(plan.fallback_messages)


def test_availability_rejects_single_landsat7_scene_after_slc_off(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """One Landsat 7 scene after 2003 should fail before downloading."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[-1,51],[0,51],[0,52],[-1,52],[-1,51]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Sparse Area",
            country="Test",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2010,
            end_year=2011,
            start_month=12,
            end_month=12,
        ),
        imagery=ImagerySpec(
            sensor="landsat",
            max_cloud_cover=20.0,
            max_scenes_per_year=3,
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"LE07-{year}",
                    provider="planetary-computer",
                    dataset="landsat-7-c2-l2",
                    acquired_at=datetime(year, 12, 15, tzinfo=UTC),
                    bbox=(-2.0, 50.0, 1.0, 53.0),
                    cloud_cover=15.0,
                    assets=(),
                ),
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )

    with pytest.raises(GeoWatchError, match="No common imagery policy"):
        build_availability_plan(spec, boundary, LANDSAT_7)


def test_availability_accepts_low_partial_coverage_with_warning(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Dispersed AOIs should continue with explicit low-coverage disclosure."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Dispersed Prefecture",
            country="Test Country",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2020,
            end_year=2025,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(
            sensor="sentinel-2",
            max_cloud_cover=20.0,
            max_scenes_per_year=3,
        ),
    )

    class FakeProvider:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"S2-{year}-partial",
                    provider="planetary-computer",
                    dataset="sentinel-2-l2a",
                    acquired_at=datetime(year, 7, 15, tzinfo=UTC),
                    bbox=(0.0, 0.0, 5.1, 5.1),
                    cloud_cover=4.0,
                    assets=(),
                ),
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: FakeProvider(),
    )

    plan = build_availability_plan(spec, boundary, SENTINEL_2)

    assert all(item.scene_count == 1 for item in plan.years.values())
    assert all(0.25 <= item.aoi_coverage < 0.95 for item in plan.years.values())
    assert "planned footprint coverage is below" in "; ".join(plan.fallback_messages)


def test_availability_uses_successful_candidate_after_intermittent_search_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Later transient provider failures should not discard a valid plan."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(name="Tokyo", country="Japan", boundary_path=boundary),
        temporal=TemporalSpec(
            start_year=2020,
            end_year=2025,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(
            sensor="sentinel-2",
            max_cloud_cover=20.0,
            max_scenes_per_year=3,
        ),
    )

    class IntermittentProvider:
        calls = 0

        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            self.calls += 1
            if self.calls > 2:
                raise AcquisitionError(
                    "Could not resolve the imagery provider host for catalog search."
                )
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"S2-{year}",
                    provider="planetary-computer",
                    dataset="sentinel-2-l2a",
                    acquired_at=datetime(year, 7, 15, tzinfo=UTC),
                    bbox=(0.0, 0.0, 0.6, 1.0),
                    cloud_cover=5.0,
                    assets=(),
                ),
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: IntermittentProvider(),
    )

    plan = build_availability_plan(spec, boundary, SENTINEL_2)

    assert plan.years[2020].scene_count == 1
    assert plan.years[2025].scene_count == 1
    assert "intermittent failure" in "; ".join(plan.fallback_messages)


def test_availability_falls_back_from_copernicus_to_planetary_computer(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Provider-level Copernicus failures should not block public STAC fallback."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[-74,40],[-73,40],[-73,41],[-74,41],[-74,40]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="New York",
            country="United States",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(
            start_year=2015,
            end_year=2016,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(
            sensor="sentinel-2",
            provider="copernicus",
            max_cloud_cover=20.0,
        ),
    )

    class FailingCopernicus:
        def search(self, _query: Any) -> tuple[SceneMetadata, ...]:
            raise AcquisitionError("STAC search failed for copernicus: HTTP 400")

    class WorkingPlanetaryComputer:
        def search(self, query: Any) -> tuple[SceneMetadata, ...]:
            year = query.start_date.year
            return (
                SceneMetadata(
                    scene_id=f"S2-{year}",
                    provider="planetary-computer",
                    dataset="sentinel-2-l2a",
                    acquired_at=datetime(year, 7, 15, tzinfo=UTC),
                    bbox=(-74.2, 39.8, -72.8, 41.2),
                    cloud_cover=5.0,
                    assets=(),
                ),
            )

    providers: list[str] = []

    def fake_build_provider(provider_name: str, *_args: Any, **_kwargs: Any) -> Any:
        providers.append(provider_name)
        if provider_name == "copernicus":
            return FailingCopernicus()
        return WorkingPlanetaryComputer()

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        fake_build_provider,
    )

    plan = build_availability_plan(spec, boundary, SENTINEL_2)

    assert providers[:2] == ["copernicus", "planetary-computer"]
    assert plan.requested_provider == "copernicus"
    assert plan.effective_provider == "planetary-computer"
    assert "requested provider copernicus failed" in "; ".join(
        plan.fallback_messages
    )


def test_availability_reports_clear_error_when_catalog_is_unreachable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A fully unreachable provider should fail with actionable guidance."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(name="Tokyo", country="Japan", boundary_path=boundary),
        temporal=TemporalSpec(
            start_year=2020,
            end_year=2025,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(sensor="sentinel-2", max_cloud_cover=20.0),
    )

    class UnreachableProvider:
        def search(self, _query: Any) -> tuple[SceneMetadata, ...]:
            raise AcquisitionError(
                "Could not resolve the imagery provider host for catalog search."
            )

    monkeypatch.setattr(
        "geowatch.application.availability.build_provider",
        lambda *_args, **_kwargs: UnreachableProvider(),
    )

    with pytest.raises(GeoWatchError, match="Satellite catalog search was interrupted"):
        build_availability_plan(spec, boundary, SENTINEL_2)
