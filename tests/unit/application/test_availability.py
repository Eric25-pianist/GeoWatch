"""All-years imagery availability policy tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from geowatch.acquisition.models import SceneMetadata
from geowatch.application.availability import build_availability_plan
from geowatch.application.models import (
    ImagerySpec,
    LocationSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.sensors import LANDSAT_7, SENTINEL_2


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


def test_availability_degrades_to_single_landsat7_scene(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Landsat 7 should prefer three scenes but not fail when only one exists."""
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

    plan = build_availability_plan(spec, boundary, LANDSAT_7)

    assert all(item.scene_count == 1 for item in plan.years.values())
    assert "fewer than the preferred 3 scene" in "; ".join(plan.fallback_messages)
