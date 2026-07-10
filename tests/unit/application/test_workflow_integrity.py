"""Workflow integrity checks for cached availability and downloads."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geowatch.acquisition.models import DownloadResult
from geowatch.application.availability import AvailabilityPlan, YearAvailability
from geowatch.application.models import (
    ImagerySpec,
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.project import ProjectLayout
from geowatch.application.sensors import LANDSAT_8, required_assets
from geowatch.application.workflow import (
    _availability_matches_spec,
    _catalog_has_complete_downloads,
    _validate_acquisition_result,
    _year_config,
)
from geowatch.core.errors import GeoWatchError


def _downloads(
    tmp_path: Path,
    scene_id: str = "LC08-001",
) -> tuple[DownloadResult, ...]:
    """Create one complete Landsat 8 download group."""
    results: list[DownloadResult] = []
    for asset_name in required_assets(LANDSAT_8):
        path = tmp_path / f"{scene_id}_{asset_name}.tif"
        path.write_bytes(b"valid")
        results.append(
            DownloadResult(
                scene_id=scene_id,
                asset_name=asset_name,
                path=path,
                bytes_written=path.stat().st_size,
                verified=True,
            )
        )
    return tuple(results)


def _availability_plan() -> AvailabilityPlan:
    """Build a compact availability plan for cache checks."""
    return AvailabilityPlan(
        dataset="landsat-8-c2-l2",
        requested_start_month=6,
        requested_end_month=8,
        requested_cloud_cover=20.0,
        effective_start_month=6,
        effective_end_month=8,
        effective_cloud_cover=20.0,
        minimum_scenes_per_year=1,
        years={
            2018: YearAvailability(
                year=2018,
                scene_ids=("LC08-001",),
                scene_count=1,
                cloud_cover=(5.0,),
                acquired_dates=("2018-07-01",),
            ),
            2020: YearAvailability(
                year=2020,
                scene_ids=("LC08-002",),
                scene_count=1,
                cloud_cover=(5.0,),
                acquired_dates=("2020-07-01",),
            ),
        },
    )


def test_acquisition_result_requires_complete_band_set(tmp_path: Path) -> None:
    """Acquisition should fail before processing when a required QA asset is absent."""
    complete = list(_downloads(tmp_path))
    incomplete = tuple(item for item in complete if item.asset_name != "qa_radsat")

    with pytest.raises(GeoWatchError, match="complete analytical band set"):
        _validate_acquisition_result(
            SimpleNamespace(downloads=incomplete),
            LANDSAT_8,
            ("LC08-001",),
            2020,
        )


def test_acquisition_result_accepts_verified_complete_downloads(tmp_path: Path) -> None:
    """Complete verified downloads should pass the acquisition integrity gate."""
    _validate_acquisition_result(
        SimpleNamespace(downloads=_downloads(tmp_path)),
        LANDSAT_8,
        ("LC08-001",),
        2020,
    )


def test_acquisition_result_allows_missing_selected_scene_with_warning(
    tmp_path: Path,
) -> None:
    """A skipped selected scene should defer the decision to coverage validation."""
    _validate_acquisition_result(
        SimpleNamespace(downloads=_downloads(tmp_path, "LC08-001")),
        LANDSAT_8,
        ("LC08-001", "LC08-002"),
        2020,
    )


def test_cached_catalog_requires_download_files(tmp_path: Path) -> None:
    """Resume should not trust a catalog when a downloaded band file is gone."""
    downloads = _downloads(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        '{"downloads": '
        + "["
        + ",".join(item.model_dump_json() for item in downloads)
        + "]}",
        encoding="utf-8",
    )

    assert _catalog_has_complete_downloads(catalog, LANDSAT_8, ("LC08-001",))
    downloads[0].path.unlink()
    assert not _catalog_has_complete_downloads(catalog, LANDSAT_8, ("LC08-001",))


def test_cached_availability_must_match_current_spec() -> None:
    """Availability cache reuse should stop when project settings changed."""
    spec = RunSpecification(
        location=LocationSpec(name="Oxford", country="United Kingdom"),
        temporal=TemporalSpec(
            start_year=2018,
            end_year=2020,
            start_month=6,
            end_month=8,
        ),
        imagery=ImagerySpec(sensor="landsat", max_cloud_cover=20.0),
    )
    changed_cloud = spec.model_copy(
        update={"imagery": spec.imagery.model_copy(update={"max_cloud_cover": 40.0})}
    )

    assert _availability_matches_spec(_availability_plan(), spec, LANDSAT_8)
    assert not _availability_matches_spec(
        _availability_plan(), changed_cloud, LANDSAT_8
    )


def test_cached_availability_must_match_requested_provider() -> None:
    """Provider fallback metadata should be part of resume-cache compatibility."""
    spec = RunSpecification(
        location=LocationSpec(name="New York", country="United States"),
        temporal=TemporalSpec(start_year=2018, end_year=2020),
        imagery=ImagerySpec(sensor="landsat", provider="copernicus"),
    )

    assert not _availability_matches_spec(_availability_plan(), spec, LANDSAT_8)


def test_year_config_uses_effective_availability_provider(tmp_path: Path) -> None:
    """Downloads should use the provider that availability planning selected."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="New York",
            country="United States",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(start_year=2015, end_year=2016),
        imagery=ImagerySpec(sensor="landsat", provider="copernicus"),
        outputs=OutputSpec(root=tmp_path),
    )
    plan = AvailabilityPlan(
        requested_provider="copernicus",
        effective_provider="planetary-computer",
        dataset="landsat-8-c2-l2",
        requested_start_month=6,
        requested_end_month=9,
        requested_cloud_cover=20.0,
        effective_start_month=6,
        effective_end_month=9,
        effective_cloud_cover=20.0,
        minimum_scenes_per_year=1,
        years={
            2015: YearAvailability(
                year=2015,
                scene_ids=("LC08-001",),
                scene_count=1,
                cloud_cover=(5.0,),
                acquired_dates=("2015-07-01",),
            )
        },
    )

    config = _year_config(
        spec,
        ProjectLayout(tmp_path / "New_York"),
        boundary,
        "landsat-8-c2-l2",
        2015,
        availability=plan,
    )

    assert config.acquisition.provider == "planetary-computer"
