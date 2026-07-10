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

from geowatch.acquisition.http import AcquisitionError
from geowatch.acquisition.models import (
    AcquisitionConfig,
    ProviderName,
    SceneMetadata,
    SearchQuery,
)
from geowatch.acquisition.selector import build_provider, rank_scenes
from geowatch.application.models import RunSpecification
from geowatch.application.sensors import SensorProfile
from geowatch.core.errors import GeoWatchError
from geowatch.utils.geometry import load_vector_geometry, reproject_geometry

IDEAL_AOI_COVERAGE = 0.95
MINIMUM_AOI_COVERAGE = 0.25
AUTOMATIC_TILE_SCENE_LIMIT = 60
LANDSAT7_SLC_OFF_YEAR = 2003
LANDSAT7_GAP_FILL_DATES = 3
LANDSAT7_GAP_FILL_SCENE_LIMIT = 12
LANDSAT7_SLC_VALID_FACTOR = 0.72
LANDSAT7_MIN_ESTIMATED_VALID_COVERAGE = 0.35
LANDSAT7_TARGET_ESTIMATED_VALID_COVERAGE = 0.65
MAX_SCENES_EVALUATED = 500
AVAILABILITY_PLANNER_VERSION = 3


class YearAvailability(BaseModel):
    """Ranked scene choices for one planned year."""

    year: int
    scene_ids: tuple[str, ...]
    scene_count: int
    cloud_cover: tuple[float | None, ...]
    acquired_dates: tuple[str, ...]
    aoi_coverage: float = 1.0
    estimated_valid_coverage: float | None = None


class AvailabilityPlan(BaseModel):
    """Common mission and temporal policy selected before downloads."""

    planner_version: int = AVAILABILITY_PLANNER_VERSION
    requested_provider: str = "auto"
    effective_provider: ProviderName = "planetary-computer"
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
            f"- Provider: {self.effective_provider}",
            f"- Dataset: {self.dataset}",
            (
                f"- Effective window: months {self.effective_start_month}-"
                f"{self.effective_end_month}"
            ),
            f"- Effective scene cloud ceiling: {self.effective_cloud_cover:.0f}%",
        ]
        for year, item in sorted(self.years.items()):
            quality_note = (
                f", estimated valid {item.estimated_valid_coverage:.1%}"
                if item.estimated_valid_coverage is not None
                else ""
            )
            lines.append(
                f"- {year}: {item.scene_count} scene(s), "
                f"{item.aoi_coverage:.1%} planned AOI coverage, "
                f"dates {_summarize_dates(item.acquired_dates)}"
                f"{quality_note}"
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
    minimum = 1
    recommended = _recommended_scene_count(profile.dataset)
    attempts = _candidate_policies(spec)
    all_search_failures: list[str] = []
    for provider_name in _provider_candidates(spec):
        provider = build_provider(
            provider_name,
            AcquisitionConfig(
                provider=provider_name,
                datasets=(profile.dataset,),
                request_timeout_seconds=90.0,
                retry_attempts=5,
                retry_backoff_seconds=2.0,
            ),
        )
        candidates, search_failures = _candidate_plans_for_provider(
            spec,
            profile,
            provider,
            provider_name,
            geometry,
            bbox,
            attempts,
            minimum_scene_count=minimum,
            recommended_scene_count=recommended,
        )
        all_search_failures.extend(
            f"{provider_name}: {failure}" for failure in search_failures
        )
        if not candidates:
            continue
        plan = max(
            candidates,
            key=lambda candidate: _plan_rank(candidate, spec, recommended),
        )
        provider_fallback = _provider_fallback_messages(spec, provider_name)
        if search_failures:
            plan = plan.model_copy(
                update={
                    "fallback_messages": (
                        *plan.fallback_messages,
                        *provider_fallback,
                        _intermittent_search_message(search_failures),
                    )
                }
            )
        elif provider_fallback:
            plan = plan.model_copy(
                update={
                    "fallback_messages": (*plan.fallback_messages, *provider_fallback)
                }
            )
        logger.info("Selected imagery availability plan\n{}", plan.summary())
        return plan
    if all_search_failures:
        raise GeoWatchError(
            "Satellite catalog search was interrupted by provider/network failures "
            "before GeoWatch could build a defensible common imagery plan. Check "
            "internet, DNS, VPN, firewall, provider availability, or choose "
            "`auto`/`planetary-computer`, then run `geowatch resume <project.yaml>`. "
            f"First failure: {all_search_failures[0]}"
        )
    raise GeoWatchError(
        "No common imagery policy supplies sufficient mission-consistent scenes "
        "for every requested year, even after automatic cloud and season fallback. "
        "Try a newer year range, a smaller AOI, Sentinel-2 for 2015+, or a local "
        "boundary with a tighter extent."
    )


def _candidate_plans_for_provider(
    spec: RunSpecification,
    profile: SensorProfile,
    provider: object,
    provider_name: ProviderName,
    geometry: BaseGeometry,
    bbox: tuple[float, float, float, float],
    attempts: tuple[tuple[int, int, float], ...],
    *,
    minimum_scene_count: int,
    recommended_scene_count: int,
) -> tuple[list[AvailabilityPlan], list[str]]:
    """Build candidate plans for one provider and return search failures."""
    candidates: list[AvailabilityPlan] = []
    search_failures: list[str] = []
    for start_month, end_month, cloud_limit in attempts:
        yearly: dict[int, YearAvailability] = {}
        successful = True
        for year in spec.temporal.years():
            try:
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
            except AcquisitionError as exc:
                successful = False
                failure = _search_failure_summary(
                    year,
                    start_month,
                    end_month,
                    cloud_limit,
                    exc,
                )
                search_failures.append(failure)
                logger.warning(
                    "Availability search failed for {} months {}-{} cloud <= {}%: {}",
                    year,
                    start_month,
                    end_month,
                    cloud_limit,
                    exc,
                )
                if _provider_level_failure(exc):
                    return candidates, search_failures
                break
            if len(scenes) < minimum_scene_count:
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
                recommended_scene_count=recommended_scene_count,
            )
            candidates.append(
                AvailabilityPlan(
                    requested_provider=spec.imagery.provider,
                    effective_provider=provider_name,
                    dataset=profile.dataset,
                    requested_start_month=spec.temporal.start_month,
                    requested_end_month=spec.temporal.end_month,
                    requested_cloud_cover=spec.imagery.max_cloud_cover,
                    effective_start_month=start_month,
                    effective_end_month=end_month,
                    effective_cloud_cover=cloud_limit,
                    minimum_scenes_per_year=minimum_scene_count,
                    years=yearly,
                    fallback_messages=messages,
                    planner_version=AVAILABILITY_PLANNER_VERSION,
                )
            )
            if _is_ideal_requested_policy(candidates[-1], spec):
                return candidates, search_failures
            if _is_robust_fallback_policy(candidates[-1], recommended_scene_count):
                return candidates, search_failures
    return candidates, search_failures


def _recommended_scene_count(dataset: str) -> int:
    """Return the preferred scene count before marking a fallback lower quality."""
    return 6 if dataset == "landsat-7-c2-l2" else 1


def _is_ideal_requested_policy(
    plan: AvailabilityPlan,
    spec: RunSpecification,
) -> bool:
    """Return True when more relaxed fallback policies are unnecessary."""
    if plan.fallback_messages:
        return False
    if (
        plan.effective_start_month != spec.temporal.start_month
        or plan.effective_end_month != spec.temporal.end_month
        or abs(plan.effective_cloud_cover - spec.imagery.max_cloud_cover) >= 1e-6
    ):
        return False
    return all(item.aoi_coverage >= IDEAL_AOI_COVERAGE for item in plan.years.values())


def _is_robust_fallback_policy(
    plan: AvailabilityPlan,
    recommended_scene_count: int,
) -> bool:
    """Return True when a fallback plan is strong enough to stop searching."""
    if not all(item.aoi_coverage >= IDEAL_AOI_COVERAGE for item in plan.years.values()):
        return False
    if not all(
        item.scene_count >= recommended_scene_count for item in plan.years.values()
    ):
        return False
    estimates = [
        item.estimated_valid_coverage
        for item in plan.years.values()
        if item.estimated_valid_coverage is not None
    ]
    return not estimates or all(
        estimate >= LANDSAT7_TARGET_ESTIMATED_VALID_COVERAGE for estimate in estimates
    )


def _provider_candidates(spec: RunSpecification) -> tuple[ProviderName, ...]:
    """Return providers to try for availability planning in priority order."""
    provider = spec.imagery.provider
    if provider == "auto":
        return ("planetary-computer",)
    if provider == "planetary-computer":
        return ("planetary-computer",)
    return provider, "planetary-computer"


def _provider_fallback_messages(
    spec: RunSpecification,
    effective_provider: ProviderName,
) -> tuple[str, ...]:
    """Describe provider fallback when the requested provider could not plan."""
    requested = spec.imagery.provider
    if requested in {"auto", effective_provider}:
        return ()
    return (
        f"requested provider {requested} failed during availability search; "
        f"using {effective_provider} for this run",
    )


def _provider_level_failure(exc: AcquisitionError) -> bool:
    """Return True when retrying more seasons on the same provider is futile."""
    message = str(exc).casefold()
    return (
        "http 400" in message
        or "http 401" in message
        or "http 403" in message
        or "stac search failed" in message
        or "could not resolve" in message
    )


def _summarize_dates(dates: tuple[str, ...]) -> str:
    """Return a compact date summary for terminal preflight output."""
    unique = tuple(dict.fromkeys(date for date in dates if date and date != "unknown"))
    if not unique:
        return "unknown"
    if len(unique) <= 3:
        return ", ".join(unique)
    return f"{unique[0]} to {unique[-1]} ({len(unique)} dates)"


def _search_failure_summary(
    year: int,
    start_month: int,
    end_month: int,
    cloud_limit: float,
    exc: AcquisitionError,
) -> str:
    """Summarize one failed availability query without losing its cause."""
    return (
        f"{year} months {start_month}-{end_month}, cloud <= {cloud_limit:.0f}%: "
        f"{exc}"
    )


def _intermittent_search_message(search_failures: list[str]) -> str:
    """Describe tolerated live catalog failures in the selected plan."""
    return (
        f"provider search had {len(search_failures)} intermittent failure(s) during "
        "availability planning; GeoWatch selected a policy from successful catalog "
        "responses and will validate downloaded coverage before analysis"
    )


def _plan_rank(
    plan: AvailabilityPlan,
    spec: RunSpecification,
    recommended_scene_count: int,
) -> tuple[bool, float, float, float, int, float]:
    """Rank plans by robustness before convenience."""
    scene_sufficiency = sum(
        min(item.scene_count / recommended_scene_count, 1.0)
        for item in plan.years.values()
    ) / max(len(plan.years), 1)
    mean_coverage = sum(item.aoi_coverage for item in plan.years.values()) / max(
        len(plan.years), 1
    )
    estimated_coverages = [
        item.estimated_valid_coverage
        for item in plan.years.values()
        if item.estimated_valid_coverage is not None
    ]
    mean_estimated_valid = (
        sum(estimated_coverages) / len(estimated_coverages)
        if estimated_coverages
        else mean_coverage
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
        mean_estimated_valid,
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
    )[:MAX_SCENES_EVALUATED]
    planning_geometry = _planning_geometry(geometry)
    scene_limit = _planning_scene_limit(profile, max_scenes)
    if _needs_landsat7_gap_fill(profile, year):
        gap_fill = _select_landsat7_gap_fill_mosaic(
            ranked,
            planning_geometry,
            midpoint,
            max_scenes=scene_limit,
        )
        if gap_fill:
            return gap_fill
        return ()
    complete = tuple(
        scene
        for scene in ranked
        if _scene_geometry_coverage(scene, planning_geometry) >= IDEAL_AOI_COVERAGE
    )
    if complete:
        return complete[:max_scenes]
    same_day = _select_same_day_mosaic(
        ranked,
        planning_geometry,
        midpoint,
        max_scenes=max(max_scenes, AUTOMATIC_TILE_SCENE_LIMIT),
    )
    if same_day:
        return same_day
    seasonal, coverage = _greedy_cover(
        ranked,
        planning_geometry,
        max_scenes=max(max_scenes, AUTOMATIC_TILE_SCENE_LIMIT),
    )
    if coverage >= MINIMUM_AOI_COVERAGE:
        return seasonal
    return ()


def _needs_landsat7_gap_fill(profile: SensorProfile, year: int) -> bool:
    """Return True when Landsat 7 SLC-off gap filling is required."""
    return profile.dataset == "landsat-7-c2-l2" and year >= LANDSAT7_SLC_OFF_YEAR


def _planning_scene_limit(profile: SensorProfile, user_limit: int) -> int:
    """Return the scene allowance needed for reliable planning."""
    if profile.dataset == "landsat-7-c2-l2":
        return max(user_limit, LANDSAT7_GAP_FILL_SCENE_LIMIT)
    return user_limit


def _select_landsat7_gap_fill_mosaic(
    scenes: tuple[SceneMetadata, ...],
    geometry: BaseGeometry,
    midpoint: datetime,
    *,
    max_scenes: int,
) -> tuple[SceneMetadata, ...]:
    """Select a multi-date Landsat 7 mosaic resilient to SLC-off striping."""
    groups: dict[date, list[SceneMetadata]] = defaultdict(list)
    for scene in scenes:
        if scene.acquired_at is not None and scene.bbox is not None:
            groups[scene.acquired_at.date()].append(scene)

    candidates: list[
        tuple[float, float, float, int, tuple[SceneMetadata, ...], float]
    ] = []
    for acquired_date, group in groups.items():
        selected, coverage = _greedy_cover(
            tuple(group),
            geometry,
            max_scenes=max_scenes,
        )
        if not selected or coverage < MINIMUM_AOI_COVERAGE:
            continue
        estimated_valid = _estimated_landsat7_valid_support(selected, coverage)
        acquired = datetime.combine(acquired_date, datetime.min.time())
        distance = abs((acquired - midpoint.replace(tzinfo=None)).total_seconds())
        mean_cloud = _mean_cloud(selected)
        candidates.append(
            (-estimated_valid, distance, mean_cloud, len(selected), selected, coverage)
        )
    if not candidates:
        return ()

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    selected_groups: list[tuple[SceneMetadata, ...]] = []
    selected_ids: set[str] = set()
    combined_estimate = 0.0
    scene_count = 0
    combined_coverage: BaseGeometry = GeometryCollection()
    for (
        negative_estimate,
        _distance,
        _cloud,
        group_size,
        selected_group,
        _coverage,
    ) in candidates:
        if scene_count + group_size > max_scenes:
            continue
        if any(scene.scene_id in selected_ids for scene in selected_group):
            continue
        selected_groups.append(selected_group)
        scene_count += group_size
        selected_ids.update(scene.scene_id for scene in selected_group)
        group_estimate = -negative_estimate
        combined_estimate = 1.0 - ((1.0 - combined_estimate) * (1.0 - group_estimate))
        for scene in selected_group:
            combined_coverage = combined_coverage.union(
                _scene_footprint(scene, geometry)
            )
        if (
            len(selected_groups) >= LANDSAT7_GAP_FILL_DATES
            and combined_estimate >= LANDSAT7_TARGET_ESTIMATED_VALID_COVERAGE
        ):
            break

    coverage = combined_coverage.area / geometry.area if geometry.area else 0.0
    if (
        len(selected_groups) < LANDSAT7_GAP_FILL_DATES
        or combined_estimate < LANDSAT7_MIN_ESTIMATED_VALID_COVERAGE
        or coverage < MINIMUM_AOI_COVERAGE
    ):
        logger.warning(
            "Rejected Landsat 7 plan with {} date group(s), estimated valid "
            "{:.1%}, footprint coverage {:.1%}.",
            len(selected_groups),
            combined_estimate,
            coverage,
        )
        return ()
    selected = tuple(scene for group in selected_groups for scene in group)
    logger.info(
        "Selected {} Landsat 7 scenes across {} dates with estimated valid "
        "coverage {:.1%}.",
        len(selected),
        len(selected_groups),
        combined_estimate,
    )
    return selected


def _estimated_landsat7_valid_support(
    scenes: tuple[SceneMetadata, ...],
    coverage: float,
) -> float:
    """Estimate usable AOI support for one Landsat 7 date mosaic."""
    clear_fraction = max(0.0, 1.0 - (_mean_cloud(scenes) / 100.0))
    return min(1.0, max(0.0, coverage * clear_fraction * LANDSAT7_SLC_VALID_FACTOR))


def _mean_cloud(scenes: tuple[SceneMetadata, ...]) -> float:
    """Return mean scene cloud percentage, pessimistic when unavailable."""
    clouds = [scene.cloud_cover for scene in scenes if scene.cloud_cover is not None]
    return sum(clouds) / len(clouds) if clouds else 100.0


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
        candidates.append((-coverage, distance, mean_cloud, selected))
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
    remaining = [
        (scene, footprint)
        for scene in scenes[:MAX_SCENES_EVALUATED]
        if not (footprint := _scene_footprint(scene, geometry)).is_empty
    ]
    covered: BaseGeometry = GeometryCollection()
    while remaining and len(selected) < max_scenes:
        ranked: list[tuple[float, float, str, SceneMetadata, BaseGeometry]] = []
        for scene, footprint in remaining:
            expanded = covered.union(footprint)
            gain = expanded.area - covered.area
            cloud = scene.cloud_cover if scene.cloud_cover is not None else 101.0
            ranked.append((-gain, cloud, scene.scene_id, scene, expanded))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        negative_gain, _, _, scene, expanded = ranked[0]
        if negative_gain >= -1e-12:
            break
        selected.append(scene)
        remaining = [item for item in remaining if item[0].scene_id != scene.scene_id]
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


def _planning_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Return a simplified geometry for fast scene-footprint planning."""
    west, south, east, north = geometry.bounds
    span = max(east - west, north - south, 0.0)
    tolerance = max(span / 10_000.0, 0.00005)
    simplified = geometry.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty or not simplified.is_valid or simplified.area <= 0.0:
        return geometry
    return simplified


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
    landsat7_estimates = [
        item.estimated_valid_coverage
        for item in yearly.values()
        if item.estimated_valid_coverage is not None
    ]
    if landsat7_estimates:
        messages.append(
            "Landsat 7 SLC-off gap-fill planning selected multi-date scenes; "
            "final pixel-level QA coverage will still be validated after download"
        )
        low_estimates = [
            f"{year} ({item.estimated_valid_coverage:.1%})"
            for year, item in sorted(yearly.items())
            if item.estimated_valid_coverage is not None
            and item.estimated_valid_coverage < LANDSAT7_TARGET_ESTIMATED_VALID_COVERAGE
        ]
        if low_estimates:
            messages.append(
                "estimated Landsat 7 valid coverage remains below the recommended "
                "65% target for " + ", ".join(low_estimates)
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
    estimated_valid_coverage = (
        _estimated_plan_valid_coverage(scenes, geometry)
        if scenes
        and geometry is not None
        and scenes[0].dataset == "landsat-7-c2-l2"
        and year >= LANDSAT7_SLC_OFF_YEAR
        else None
    )
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
        estimated_valid_coverage=estimated_valid_coverage,
    )


def _estimated_plan_valid_coverage(
    scenes: tuple[SceneMetadata, ...],
    geometry: BaseGeometry,
) -> float:
    """Estimate valid coverage for a selected multi-date Landsat 7 plan."""
    by_date: dict[str, list[SceneMetadata]] = defaultdict(list)
    for scene in scenes:
        key = scene.acquired_at.date().isoformat() if scene.acquired_at else "unknown"
        by_date[key].append(scene)
    combined = 0.0
    for group in by_date.values():
        group_tuple = tuple(group)
        coverage = _combined_coverage(group_tuple, geometry)
        group_estimate = _estimated_landsat7_valid_support(group_tuple, coverage)
        combined = 1.0 - ((1.0 - combined) * (1.0 - group_estimate))
    return combined
