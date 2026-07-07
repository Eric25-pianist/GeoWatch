"""All-years imagery availability planning before material downloads."""

from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from shapely.geometry import GeometryCollection, box
from shapely.geometry.base import BaseGeometry

from geowatch.acquisition.models import AcquisitionConfig, SceneMetadata, SearchQuery
from geowatch.acquisition.selector import build_provider, rank_scenes
from geowatch.application.models import RunSpecification
from geowatch.application.sensors import SensorProfile
from geowatch.core.errors import GeoWatchError
from geowatch.utils.geometry import load_vector_geometry, reproject_geometry

IDEAL_AOI_COVERAGE = 0.95
MINIMUM_AOI_COVERAGE = 0.50
AUTOMATIC_TILE_SCENE_LIMIT = 20


class YearAvailability(BaseModel):
    """Ranked scene choices for one planned year."""

    year: int
    scene_ids: tuple[str, ...]
    scene_count: int
    cloud_cover: tuple[float | None, ...]
    acquired_dates: tuple[str, ...]
    aoi_coverage: float = 1.0


class AvailabilityPlan(BaseModel):
    """Common mission and temporal policy selected before downloads."""

    dataset: str
    requested_start_month: int
    requested_end_month: int
    requested_cloud_cover: float
    effective_start_month: int
    effective_end_month: int
    effective_cloud_cover: float
    minimum_scenes_per_year: int
    years: dict[int, YearAvailability]
    fallback_messages: tuple[str, ...] = ()

    @property
    def used_fallback(self) -> bool:
        """Return whether the plan differs from the requested settings."""
        return bool(self.fallback_messages)

    def summary(self) -> str:
        """Render a concise terminal summary."""
        lines = [
            "GeoWatch imagery availability plan",
            f"- Dataset: {self.dataset}",
            (
                f"- Effective window: months {self.effective_start_month}-"
                f"{self.effective_end_month}"
            ),
            f"- Effective scene cloud ceiling: {self.effective_cloud_cover:.0f}%",
        ]
        for year, item in sorted(self.years.items()):
            lines.append(
                f"- {year}: {item.scene_count} scene(s), "
                f"{item.aoi_coverage:.1%} planned AOI coverage"
            )
        lines.extend(f"- Fallback: {message}" for message in self.fallback_messages)
        return "\n".join(lines)


def build_availability_plan(
    spec: RunSpecification,
    boundary_path: Path,
    profile: SensorProfile,
) -> AvailabilityPlan:
    """Find one common defensible search policy for all requested years."""
    boundary = load_vector_geometry(boundary_path)
    geometry = reproject_geometry(boundary.geometry, boundary.crs, "EPSG:4326")
    west, south, east, north = geometry.bounds
    bbox = float(west), float(south), float(east), float(north)
    provider_name = (
        "planetary-computer"
        if spec.imagery.provider == "auto"
        else spec.imagery.provider
    )
    provider = build_provider(
        provider_name,
        AcquisitionConfig(
            provider=provider_name,
            datasets=(profile.dataset,),
            request_timeout_seconds=60.0,
        ),
    )
    minimum = 1
    recommended = _recommended_scene_count(profile.dataset)
    attempts = _candidate_policies(spec)
    candidates: list[AvailabilityPlan] = []
    for start_month, end_month, cloud_limit in attempts:
        yearly: dict[int, YearAvailability] = {}
        successful = True
        for year in spec.temporal.years():
            scenes = _search_year(
                provider,
                profile,
                geometry,
                bbox,
                year,
                start_month,
                end_month,
                cloud_limit,
                spec.imagery.max_scenes_per_year,
            )
            if len(scenes) < minimum:
                successful = False
                break
            yearly[year] = _year_availability(year, scenes, geometry)
        if successful:
            messages = _fallback_messages(
                spec,
                start_month,
                end_month,
                cloud_limit,
                yearly,
                recommended_scene_count=recommended,
            )
            candidates.append(
                AvailabilityPlan(
                    dataset=profile.dataset,
                    requested_start_month=spec.temporal.start_month,
                    requested_end_month=spec.temporal.end_month,
                    requested_cloud_cover=spec.imagery.max_cloud_cover,
                    effective_start_month=start_month,
                    effective_end_month=end_month,
                    effective_cloud_cover=cloud_limit,
                    minimum_scenes_per_year=minimum,
                    years=yearly,
                    fallback_messages=messages,
                )
            )
    if candidates:
        plan = max(
            candidates,
            key=lambda candidate: _plan_rank(candidate, spec, recommended),
        )
        logger.info("Selected imagery availability plan\n{}", plan.summary())
        return plan
    raise GeoWatchError(
        "No common imagery policy supplies sufficient mission-consistent scenes "
        "for every requested year, even after automatic cloud and season fallback. "
        "Try a newer year range, a smaller AOI, Sentinel-2 for 2015+, or a local "
        "boundary with a tighter extent."
    )


def _recommended_scene_count(dataset: str) -> int:
    """Return the preferred scene count before marking a fallback lower quality."""
    return 3 if dataset == "landsat-7-c2-l2" else 1


def _plan_rank(
    plan: AvailabilityPlan,
    spec: RunSpecification,
    recommended_scene_count: int,
) -> tuple[bool, float, float, int, float]:
    """Rank plans by robustness before convenience."""
    scene_sufficiency = sum(
        min(item.scene_count / recommended_scene_count, 1.0)
        for item in plan.years.values()
    ) / max(len(plan.years), 1)
    mean_coverage = sum(item.aoi_coverage for item in plan.years.values()) / max(
        len(plan.years), 1
    )
    all_recommended = all(
        item.scene_count >= recommended_scene_count for item in plan.years.values()
    )
    seasonal_delta = abs(plan.effective_start_month - spec.temporal.start_month) + abs(
        plan.effective_end_month - spec.temporal.end_month
    )
    return (
        all_recommended,
        scene_sufficiency,
        mean_coverage,
        -seasonal_delta,
        -plan.effective_cloud_cover,
    )


def _search_year(
    provider: object,
    profile: SensorProfile,
    geometry: BaseGeometry,
    bbox: tuple[float, float, float, float],
    year: int,
    start_month: int,
    end_month: int,
    cloud_limit: float,
    max_scenes: int,
) -> tuple[SceneMetadata, ...]:
    start = date(year, start_month, 1)
    end = date(year, end_month, calendar.monthrange(year, end_month)[1])
    query = SearchQuery(
        bbox=bbox,
        start_date=start,
        end_date=end,
        datasets=(profile.dataset,),
        max_cloud_cover=cloud_limit,
        limit=(
            500 if profile.dataset == "sentinel-2-l2a" else max(200, max_scenes * 25)
        ),
    )
    scenes = provider.search(query)  # type: ignore[attr-defined]
    midpoint = datetime.combine(start + ((end - start) / 2), datetime.min.time())
    ranked = rank_scenes(
        scenes,
        datasets=(profile.dataset,),
        aoi_bbox=bbox,
        temporal_midpoint=midpoint,
    )
    complete = tuple(
        scene
        for scene in ranked
        if _scene_geometry_coverage(scene, geometry) >= IDEAL_AOI_COVERAGE
    )
    if complete:
        return complete[:max_scenes]
    same_day = _select_same_day_mosaic(
        ranked,
        geometry,
        midpoint,
        max_scenes=max(max_scenes, AUTOMATIC_TILE_SCENE_LIMIT),
    )
    if same_day:
        return same_day
    seasonal, coverage = _greedy_cover(
        ranked,
        geometry,
        max_scenes=max(max_scenes, AUTOMATIC_TILE_SCENE_LIMIT),
    )
    if coverage >= MINIMUM_AOI_COVERAGE:
        return seasonal
    return ()


def _select_same_day_mosaic(
    scenes: tuple[SceneMetadata, ...],
    geometry: BaseGeometry,
    midpoint: datetime,
    *,
    max_scenes: int,
) -> tuple[SceneMetadata, ...]:
    """Select one same-day tile set whose union covers the AOI."""
    groups: dict[date, list[SceneMetadata]] = defaultdict(list)
    for scene in scenes:
        if scene.acquired_at is not None and scene.bbox is not None:
            groups[scene.acquired_at.date()].append(scene)

    candidates: list[tuple[float, float, float, tuple[SceneMetadata, ...]]] = []
    for acquired_date, group in groups.items():
        selected, coverage = _greedy_cover(
            tuple(group),
            geometry,
            max_scenes=max_scenes,
        )
        if coverage < IDEAL_AOI_COVERAGE:
            continue
        clouds = [
            scene.cloud_cover for scene in selected if scene.cloud_cover is not None
        ]
        mean_cloud = sum(clouds) / len(clouds) if clouds else 101.0
        acquired = datetime.combine(acquired_date, datetime.min.time())
        distance = abs((acquired - midpoint.replace(tzinfo=None)).total_seconds())
        candidates.append((-coverage, mean_cloud, distance, selected))
    if not candidates:
        return ()
    candidates.sort(key=lambda item: (item[0], item[1], item[2], len(item[3])))
    return candidates[0][3]


def _greedy_cover(
    scenes: tuple[SceneMetadata, ...],
    geometry: BaseGeometry,
    *,
    max_scenes: int,
) -> tuple[tuple[SceneMetadata, ...], float]:
    """Choose the smallest useful tile set by repeatedly adding maximum coverage."""
    selected: list[SceneMetadata] = []
    remaining = list(scenes)
    covered: BaseGeometry = GeometryCollection()
    while remaining and len(selected) < max_scenes:
        ranked: list[tuple[float, float, str, SceneMetadata, BaseGeometry]] = []
        for scene in remaining:
            footprint = _scene_footprint(scene, geometry)
            expanded = covered.union(footprint)
            gain = expanded.area - covered.area
            cloud = scene.cloud_cover if scene.cloud_cover is not None else 101.0
            ranked.append((-gain, cloud, scene.scene_id, scene, expanded))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        negative_gain, _, _, scene, expanded = ranked[0]
        if negative_gain >= 0.0:
            break
        selected.append(scene)
        remaining.remove(scene)
        covered = expanded
        coverage = covered.area / geometry.area if geometry.area else 0.0
        if coverage >= IDEAL_AOI_COVERAGE:
            return tuple(selected), coverage
    coverage = covered.area / geometry.area if geometry.area else 0.0
    return tuple(selected), coverage


def _scene_footprint(
    scene: SceneMetadata,
    geometry: BaseGeometry,
) -> BaseGeometry:
    """Return the scene bbox clipped to the approved AOI geometry."""
    if scene.bbox is None:
        return GeometryCollection()
    return box(*scene.bbox).intersection(geometry)


def _scene_geometry_coverage(
    scene: SceneMetadata,
    geometry: BaseGeometry,
) -> float:
    """Calculate scene coverage against the AOI polygon instead of its bbox."""
    footprint = _scene_footprint(scene, geometry)
    return footprint.area / geometry.area if geometry.area else 0.0


def _combined_coverage(
    scenes: tuple[SceneMetadata, ...],
    geometry: BaseGeometry,
) -> float:
    """Calculate the union coverage of selected scene footprints."""
    covered: BaseGeometry = GeometryCollection()
    for scene in scenes:
        covered = covered.union(_scene_footprint(scene, geometry))
    return covered.area / geometry.area if geometry.area else 0.0


def _candidate_policies(
    spec: RunSpecification,
) -> tuple[tuple[int, int, float], ...]:
    requested = spec.imagery.max_cloud_cover
    windows = (
        (spec.temporal.start_month, spec.temporal.end_month),
        (
            max(1, spec.temporal.start_month - 1),
            min(12, spec.temporal.end_month + 1),
        ),
        (
            max(1, spec.temporal.start_month - 2),
            min(12, spec.temporal.end_month + 2),
        ),
        (1, 12),
    )
    clouds = (requested, max(requested, 40.0), max(requested, 60.0), 80.0, 100.0)
    raw = tuple(
        (start_month, end_month, cloud)
        for start_month, end_month in windows
        for cloud in clouds
    )
    return tuple(dict.fromkeys(raw))


def _fallback_messages(
    spec: RunSpecification,
    start_month: int,
    end_month: int,
    cloud_limit: float,
    yearly: dict[int, YearAvailability],
    *,
    recommended_scene_count: int,
) -> tuple[str, ...]:
    messages: list[str] = []
    if cloud_limit != spec.imagery.max_cloud_cover:
        messages.append(
            f"scene cloud ceiling increased from {spec.imagery.max_cloud_cover:.0f}% "
            f"to {cloud_limit:.0f}%; pixel QA masking remains enabled"
        )
    if (start_month, end_month) != (
        spec.temporal.start_month,
        spec.temporal.end_month,
    ):
        messages.append(
            f"common seasonal window expanded from months "
            f"{spec.temporal.start_month}-{spec.temporal.end_month} to "
            f"{start_month}-{end_month}"
        )
    if (start_month, end_month) == (1, 12):
        messages.append(
            "seasonal fallback expanded to the full year; interpret seasonal "
            "change with extra caution"
        )
    if cloud_limit >= 80.0:
        messages.append(
            "high scene-cloud fallback was required; pixel-level QA masking and "
            "valid-coverage scoring are especially important"
        )
    effective_scene_count = max(item.scene_count for item in yearly.values())
    if effective_scene_count > spec.imagery.max_scenes_per_year:
        messages.append(
            "large-AOI/seasonal fallback expanded the scene allowance from "
            f"{spec.imagery.max_scenes_per_year} to {effective_scene_count} "
            "scenes per year"
        )
    low_coverage_years = [
        f"{year} ({item.aoi_coverage:.1%})"
        for year, item in sorted(yearly.items())
        if item.aoi_coverage < IDEAL_AOI_COVERAGE
    ]
    if low_coverage_years:
        messages.append(
            "planned footprint coverage is below the ideal 95% threshold for "
            + ", ".join(low_coverage_years)
        )
    if any(item.scene_count < recommended_scene_count for item in yearly.values()):
        messages.append(
            f"fewer than the preferred {recommended_scene_count} scene(s) were "
            "available for at least one year; cloud/SLC-off gaps may remain after "
            "compositing"
        )
    return tuple(messages)


def _year_availability(
    year: int,
    scenes: tuple[SceneMetadata, ...],
    geometry: BaseGeometry | None = None,
) -> YearAvailability:
    return YearAvailability(
        year=year,
        scene_ids=tuple(scene.scene_id for scene in scenes),
        scene_count=len(scenes),
        cloud_cover=tuple(scene.cloud_cover for scene in scenes),
        acquired_dates=tuple(
            scene.acquired_at.date().isoformat() if scene.acquired_at else "unknown"
            for scene in scenes
        ),
        aoi_coverage=(
            _combined_coverage(scenes, geometry) if geometry is not None else 1.0
        ),
    )
