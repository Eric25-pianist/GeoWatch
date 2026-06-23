"""Resumable real-data orchestration for professional GeoWatch projects."""

from __future__ import annotations

import calendar
import json
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
from loguru import logger

from geowatch.acquisition.models import AcquisitionConfig, DatasetName, SceneMetadata
from geowatch.acquisition.pipeline import run_acquisition
from geowatch.analytics.pipeline import run_analytics_pipeline
from geowatch.application.availability import (
    AvailabilityPlan,
    build_availability_plan,
)
from geowatch.application.manifest import (
    RunManifest,
    load_or_create_manifest,
    save_manifest,
)
from geowatch.application.models import RunSpecification
from geowatch.application.project import ProjectLayout, load_run_specification
from geowatch.application.publication import (
    write_annual_master_report,
    write_professional_outputs,
)
from geowatch.application.scenes import build_year_composite, load_processed_composite
from geowatch.application.sensors import (
    SENSOR_PROFILES,
    required_assets,
    select_common_sensor,
)
from geowatch.cartography.themes import MapThemeName
from geowatch.config.models import (
    AOIConfig,
    DateRangeConfig,
    LoggingConfig,
    OutputConfig,
    ProjectConfig,
    RasterProcessingConfig,
)
from geowatch.core.errors import GeoWatchError
from geowatch.logging.manager import LoggerManager
from geowatch.processing.models import RasterLayer
from geowatch.reporting.cartography import render_cartography_suite


def process_project(
    project_file: Path,
    *,
    resume: bool = True,
    map_theme: MapThemeName | None = None,
) -> Path:
    """Execute or resume a complete real-data terminal project."""
    spec = load_run_specification(project_file)
    if map_theme is not None:
        spec = spec.model_copy(
            update={"outputs": spec.outputs.model_copy(update={"map_theme": map_theme})}
        )
    layout = ProjectLayout(project_file.parent)
    layout.create(spec.temporal.years())
    LoggerManager(layout.root / "logs", "INFO").configure()
    spec.log_summary()
    boundary_path = spec.location.boundary_path
    if boundary_path is None or not boundary_path.exists():
        raise GeoWatchError("Project has no confirmed local administrative boundary.")
    manifest = load_or_create_manifest(layout.manifest, project_file)
    profile = select_common_sensor(
        spec.temporal.start_year,
        spec.temporal.end_year,
        spec.imagery.sensor,
    )
    availability = _prepare_availability(
        spec,
        layout,
        manifest,
        boundary_path,
        profile,
        resume=resume,
    )
    catalogs: dict[int, Path] = {}
    composites: dict[int, Path] = {}
    for year in spec.temporal.years():
        catalogs[year] = _acquire_year(
            spec,
            layout,
            manifest,
            boundary_path,
            profile.dataset,
            availability,
            year,
            resume=resume,
        )
        composites[year] = _process_year(
            spec,
            layout,
            manifest,
            boundary_path,
            catalogs[year],
            profile.dataset,
            year,
            resume=resume,
        )

    pairs = _comparison_pairs(spec.temporal.years(), spec.temporal.mode)
    comparison_reports: dict[str, Path] = {}
    for start_year, end_year in pairs:
        comparison_reports[f"{start_year}-{end_year}"] = _analyze_and_publish(
            spec,
            layout,
            manifest,
            boundary_path,
            catalogs,
            composites[start_year],
            composites[end_year],
            start_year,
            end_year,
            availability,
            resume=resume,
        )
    if spec.temporal.mode == "annual":
        annual_outputs = write_annual_master_report(
            spec, layout, availability, comparison_reports
        )
        manifest.complete(
            "publication:annual-master",
            *annual_outputs.values(),
            message=f"Published {len(comparison_reports)} annual comparisons",
        )
    save_manifest(manifest, layout.manifest)
    logger.info("Completed GeoWatch project {}", layout.root)
    return layout.root


def preflight_project(project_file: Path, *, force: bool = False) -> AvailabilityPlan:
    """Build and persist the all-years availability plan without downloading."""
    spec = load_run_specification(project_file)
    layout = ProjectLayout(project_file.parent)
    layout.create(spec.temporal.years())
    boundary_path = spec.location.boundary_path
    if boundary_path is None or not boundary_path.exists():
        raise GeoWatchError("Project has no confirmed local administrative boundary.")
    manifest = load_or_create_manifest(layout.manifest, project_file)
    profile = select_common_sensor(
        spec.temporal.start_year,
        spec.temporal.end_year,
        spec.imagery.sensor,
    )
    return _prepare_availability(
        spec,
        layout,
        manifest,
        boundary_path,
        profile,
        resume=not force,
    )


def project_status(project_file: Path) -> str:
    """Render a human-readable status table for one project."""
    layout = ProjectLayout(project_file.parent)
    manifest = load_or_create_manifest(layout.manifest, project_file)
    lines = [f"GeoWatch project: {layout.root}"]
    if not manifest.stages:
        lines.append("- No stages have run.")
    for name, record in manifest.stages.items():
        suffix = f" - {record.message}" if record.message else ""
        lines.append(f"- {name}: {record.status}{suffix}")
    return "\n".join(lines)


def _acquire_year(
    spec: RunSpecification,
    layout: ProjectLayout,
    manifest: RunManifest,
    boundary_path: Path,
    dataset: DatasetName,
    availability: AvailabilityPlan,
    year: int,
    *,
    resume: bool,
) -> Path:
    stage = f"acquisition:{year}"
    catalog = layout.root / "raw" / str(year) / "acquisition_catalog.json"
    if resume and manifest.is_complete(stage):
        return catalog
    manifest.start(stage)
    save_manifest(manifest, layout.manifest)
    try:
        config = _year_config(
            spec,
            layout,
            boundary_path,
            dataset,
            year,
            availability=availability,
        )
        config_path = layout.root / "configs" / f"{year}.yaml"
        from geowatch.config.loader import write_config

        write_config(config, config_path)
        result = run_acquisition(config, base_dir=layout.root)
        _require_downloads(result.downloads, year)
        manifest.complete(
            stage,
            result.catalog_path,
            result.report_path,
            message=f"{len(result.downloads)} verified assets from {result.provider}",
        )
        save_manifest(manifest, layout.manifest)
        return result.catalog_path
    except Exception as exc:
        manifest.fail(stage, str(exc))
        save_manifest(manifest, layout.manifest)
        raise


def _process_year(
    spec: RunSpecification,
    layout: ProjectLayout,
    manifest: RunManifest,
    boundary_path: Path,
    catalog: Path,
    dataset: DatasetName,
    year: int,
    *,
    resume: bool,
) -> Path:
    stage = f"processing:{year}"
    output = layout.root / "processed" / str(year) / "surface_reflectance.tif"
    if resume and manifest.is_complete(stage):
        return output
    manifest.start(stage)
    save_manifest(manifest, layout.manifest)
    try:
        layer = build_year_composite(
            catalog,
            boundary_path,
            SENSOR_PROFILES[dataset],
            year=year,
            output_path=output,
            method=spec.imagery.composite_method,
            target_crs=spec.outputs.target_crs,
            min_valid_coverage=(0.65 if dataset == "landsat-7-c2-l2" else 0.70),
        )
        actual_path = Path(str(layer.metadata["output_path"]))
        manifest.complete(
            stage,
            actual_path,
            message=(
                f"{layer.grid.width}x{layer.grid.height} pixels in {layer.grid.crs}"
            ),
        )
        save_manifest(manifest, layout.manifest)
        return actual_path
    except Exception as exc:
        manifest.fail(stage, str(exc))
        save_manifest(manifest, layout.manifest)
        raise


def _analyze_and_publish(
    spec: RunSpecification,
    layout: ProjectLayout,
    manifest: RunManifest,
    boundary_path: Path,
    catalogs: dict[int, Path],
    start_path: Path,
    end_path: Path,
    start_year: int,
    end_year: int,
    availability: AvailabilityPlan,
    *,
    resume: bool,
) -> Path:
    stage = f"publication:{start_year}-{end_year}"
    comparison_root = (
        layout.root
        if (start_year, end_year) == (spec.temporal.start_year, spec.temporal.end_year)
        else layout.root / "comparisons" / f"{start_year}_{end_year}"
    )
    report_path = comparison_root / "reports" / "report.html"
    if resume and manifest.is_complete(stage):
        return report_path
    manifest.start(stage)
    save_manifest(manifest, layout.manifest)
    try:
        scene_t1 = load_processed_composite(start_path)
        scene_t2 = load_processed_composite(end_path)
        mandatory_indices = {"ndvi", "ndbi", "ndwi"}
        indices = tuple(dict.fromkeys((*spec.analysis.indices, *mandatory_indices)))
        training = _load_training_labels(spec.analysis.training_data, scene_t1)
        analytics = run_analytics_pipeline(
            scene_t1,
            scene_t2,
            output_root=comparison_root,
            classification_method=spec.analysis.classification,
            training_labels_t1=training,
            training_labels_t2=training,
            index_names=indices,
            change_methods=spec.analysis.change_methods,
        )
        config = _year_config(
            spec,
            layout,
            boundary_path,
            select_common_sensor(start_year, end_year, spec.imagery.sensor).dataset,
            start_year,
        )
        config.project_name = f"{spec.location.name} {start_year}-{end_year}"
        maps = render_cartography_suite(
            config,
            scene_t1,
            scene_t2,
            analytics,
            output_dir=comparison_root / "maps",
        )
        sources = _read_sources(catalogs[start_year]) + _read_sources(
            catalogs[end_year]
        )
        outputs = write_professional_outputs(
            spec,
            ProjectLayout(comparison_root),
            boundary_path,
            scene_t1,
            scene_t2,
            analytics,
            maps,
            sources,
            availability,
        )
        report_path = outputs["html_report"]
        manifest.complete(
            stage,
            report_path,
            outputs["pdf_report"],
            outputs["validation_report"],
            outputs["dashboard"],
            *(
                (outputs["quality_markdown"],)
                if "quality_markdown" in outputs
                else ()
            ),
            message=f"Published {len(maps)} professional map themes",
        )
        save_manifest(manifest, layout.manifest)
        return report_path
    except Exception as exc:
        manifest.fail(stage, str(exc))
        save_manifest(manifest, layout.manifest)
        raise


def _year_config(
    spec: RunSpecification,
    layout: ProjectLayout,
    boundary_path: Path,
    dataset: DatasetName,
    year: int,
    availability: AvailabilityPlan | None = None,
) -> ProjectConfig:
    start_month = (
        availability.effective_start_month
        if availability is not None
        else spec.temporal.start_month
    )
    end_month = (
        availability.effective_end_month
        if availability is not None
        else spec.temporal.end_month
    )
    cloud_limit = (
        availability.effective_cloud_cover
        if availability is not None
        else spec.imagery.max_cloud_cover
    )
    final_day = calendar.monthrange(year, end_month)[1]
    raw = layout.root / "raw" / str(year)
    output = layout.root
    provider = spec.imagery.provider
    selected_scene_count = (
        availability.years[year].scene_count
        if availability is not None
        else spec.imagery.max_scenes_per_year
    )
    return ProjectConfig(
        project_name=f"{spec.location.name}-{year}",
        aoi=AOIConfig(kind="geojson", path=boundary_path.resolve(), crs="EPSG:4326"),
        dates=DateRangeConfig(
            start_date=date(year, start_month, 1),
            end_date=date(year, end_month, final_day),
        ),
        acquisition=AcquisitionConfig(
            download=True,
            provider=provider,
            datasets=(dataset,),
            max_cloud_cover=cloud_limit,
            max_results=500
            if availability is not None
            else max(50, spec.imagery.max_scenes_per_year * 10),
            max_downloads=(
                len(required_assets(SENSOR_PROFILES[dataset])) * selected_scene_count
            ),
            download_directory=raw / "assets",
            metadata_catalog=raw / "acquisition_catalog.json",
            acquisition_report=layout.root / "reports" / f"acquisition_{year}.md",
            max_download_bytes=8_589_934_592,
            selected_scene_ids=(
                availability.years[year].scene_ids if availability is not None else ()
            ),
        ),
        outputs=OutputConfig(
            root=output,
            rasters=output / "processed",
            vectors=output / "boundary",
            maps=output / "maps",
            reports=output / "reports",
            statistics=output / "statistics",
            manifests=output,
            exports=output / "exports",
            map_theme=spec.outputs.map_theme,
        ),
        raster_processing=RasterProcessingConfig(max_workers=spec.outputs.max_workers),
        logging=LoggingConfig(directory=layout.root / "logs", level="INFO"),
    )


def _read_sources(path: Path) -> tuple[SceneMetadata, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes = payload.get("scenes", [])
    return tuple(
        SceneMetadata.model_validate(item) for item in scenes if isinstance(item, dict)
    )


def _comparison_pairs(years: tuple[int, ...], mode: str) -> tuple[tuple[int, int], ...]:
    if mode == "endpoints":
        return ((years[0], years[-1]),)
    return tuple(pairwise(years))


def _load_training_labels(path: Path | None, scene: RasterLayer) -> np.ndarray | None:
    if path is None:
        return None
    from geowatch.processing.io import read_raster

    labels = read_raster(path).data[0]
    expected = (scene.grid.height, scene.grid.width)
    if labels.shape != expected:
        raise GeoWatchError("Training raster must match the processed analysis grid.")
    return np.asarray(labels, dtype=np.int64)


def _require_downloads(downloads: object, year: int) -> None:
    """Require at least one verified acquisition asset."""
    if not downloads:
        raise GeoWatchError(f"No imagery assets were downloaded for {year}.")


def _prepare_availability(
    spec: RunSpecification,
    layout: ProjectLayout,
    manifest: RunManifest,
    boundary_path: Path,
    profile: object,
    *,
    resume: bool,
) -> AvailabilityPlan:
    """Load or build the common imagery availability plan."""
    from geowatch.application.sensors import SensorProfile

    if not isinstance(profile, SensorProfile):
        raise GeoWatchError("Invalid sensor profile for availability planning.")
    stage = "availability"
    path = layout.root / "availability_plan.json"
    if resume and manifest.is_complete(stage):
        return AvailabilityPlan.model_validate_json(path.read_text(encoding="utf-8"))
    manifest.start(stage)
    save_manifest(manifest, layout.manifest)
    try:
        plan = build_availability_plan(spec, boundary_path, profile)
        path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        manifest.complete(stage, path, message=plan.summary())
        save_manifest(manifest, layout.manifest)
        return plan
    except Exception as exc:
        manifest.fail(stage, str(exc))
        save_manifest(manifest, layout.manifest)
        raise
