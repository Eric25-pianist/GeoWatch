"""Transparent run-quality scoring for GeoWatch outputs."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from pyproj import CRS, Geod

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import AccuracyAssessment, AnalyticsReport
from geowatch.application.availability import AvailabilityPlan
from geowatch.application.models import RunSpecification
from geowatch.processing.models import RasterLayer
from geowatch.utils.geometry import (
    geometry_mask_for_grid,
    load_vector_geometry,
    reproject_geometry,
)

if TYPE_CHECKING:
    from geowatch.reporting.models import MapArtifact


@dataclass(frozen=True)
class QualityComponent:
    """One weighted quality-score component with transparent reasoning."""

    key: str
    title: str
    weight: int
    score: float
    status: str
    summary: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "key": self.key,
            "title": self.title,
            "weight": self.weight,
            "score": round(self.score, 2),
            "status": self.status,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class QualityScoreReport:
    """Overall GeoWatch run-quality summary."""

    generated_at: datetime
    total_score: float
    max_score: int
    overall_status: str
    classification_confidence: str
    components: tuple[QualityComponent, ...]
    warnings: tuple[str, ...]
    accuracy_metrics: dict[str, dict[str, object]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def rounded_score(self) -> int:
        """Return the overall score rounded for display."""
        return round(self.total_score)

    def component(self, key: str) -> QualityComponent | None:
        """Return one component by key if present."""
        for item in self.components:
            if item.key == key:
                return item
        return None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_score": round(self.total_score, 2),
            "max_score": self.max_score,
            "rounded_score": self.rounded_score,
            "overall_status": self.overall_status,
            "classification_confidence": self.classification_confidence,
            "components": [item.as_dict() for item in self.components],
            "warnings": list(self.warnings),
            "accuracy_metrics": self.accuracy_metrics,
            "metadata": self.metadata,
        }

    def format_terminal(self) -> str:
        """Render a compact terminal summary."""
        lines = [
            f"GeoWatch Quality Score: {self.rounded_score}/{self.max_score}",
            f"Overall quality: {self.overall_status}",
        ]
        for item in self.components:
            lines.append(f"{item.title}: {item.status}")
        if self.component("classification") is None:
            lines.append(
                f"Classification confidence: {self.classification_confidence}"
            )
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)

    def as_markdown(self) -> str:
        """Render a human-readable Markdown report."""
        lines = [
            "# GeoWatch Quality Score",
            "",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- GeoWatch Quality Score: {self.rounded_score}/{self.max_score}",
            f"- Overall quality: {self.overall_status}",
            f"- Classification confidence: {self.classification_confidence}",
            "",
            "## Component Scores",
            "",
        ]
        for item in self.components:
            lines.extend(
                [
                    f"### {item.title}",
                    "",
                    f"- Score: {item.score:.1f}/{item.weight}",
                    f"- Status: {item.status}",
                    f"- Summary: {item.summary}",
                ]
            )
            for reason in item.reasons:
                lines.append(f"- Reason: {reason}")
            for warning in item.warnings:
                lines.append(f"- Warning: {warning}")
            lines.append("")
        lines.extend(["## Warnings", ""])
        if self.warnings:
            lines.extend(f"- {warning}" for warning in self.warnings)
        else:
            lines.append("- No run-quality warnings were generated.")
        if self.accuracy_metrics:
            lines.extend(["", "## Accuracy Metrics", ""])
            for name, metrics in self.accuracy_metrics.items():
                lines.append(
                    f"- {name}: overall_accuracy={metrics['overall_accuracy']}, "
                    f"kappa={metrics['kappa']}"
                )
        return "\n".join(lines).rstrip() + "\n"


def calculate_quality_score(
    *,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None = None,
    maps: Mapping[str, MapArtifact] | None = None,
    downloads: Mapping[str, Path] | None = None,
) -> QualityScoreReport:
    """Calculate the weighted GeoWatch run-quality score."""
    boundary = load_vector_geometry(boundary_path)
    boundary_area_km2 = _boundary_area_km2(boundary.geometry, boundary.crs)
    masks = {
        "t1": _boundary_mask(boundary_path, scene_t1),
        "t2": _boundary_mask(boundary_path, scene_t2),
    }
    valid_t1 = _valid_coverage(scene_t1, masks["t1"])
    valid_t2 = _valid_coverage(scene_t2, masks["t2"])
    invalid_t1 = 1.0 - valid_t1
    invalid_t2 = 1.0 - valid_t2
    avg_cloud = _average_cloud_cover(sources)
    components = (
        _boundary_component(spec, boundary_path, boundary_area_km2, boundary),
        _imagery_component(sources, availability),
        _cloud_component(valid_t1, valid_t2, invalid_t1, invalid_t2, avg_cloud),
        _sensor_component(sources),
        _season_component(spec, sources, availability),
        _processing_component(spec, analytics, maps or {}, downloads or {}),
        _classification_component(spec, analytics),
    )
    warnings = _unique(
        warning for component in components for warning in component.warnings
    )
    accuracy_metrics = _accuracy_metrics(analytics.accuracy)
    score = sum(component.score for component in components)
    report = QualityScoreReport(
        generated_at=datetime.now(UTC),
        total_score=score,
        max_score=100,
        overall_status=_overall_status(score),
        classification_confidence=_classification_confidence(spec, analytics),
        components=components,
        warnings=warnings,
        accuracy_metrics=accuracy_metrics,
        metadata={
            "boundary_area_km2": round(boundary_area_km2, 3),
            "valid_coverage_start": round(valid_t1, 4),
            "valid_coverage_end": round(valid_t2, 4),
            "invalid_fraction_start": round(invalid_t1, 4),
            "invalid_fraction_end": round(invalid_t2, 4),
            "average_scene_cloud_cover": (
                None if avg_cloud is None else round(avg_cloud, 2)
            ),
            "scene_count": len(sources),
            "datasets": sorted({source.dataset for source in sources}),
        },
    )
    logger.info(
        "Calculated GeoWatch Quality Score {} / {} for {}",
        report.rounded_score,
        report.max_score,
        spec.location.name,
    )
    return report


def write_quality_outputs(
    output_dir: Path, report: QualityScoreReport
) -> dict[str, Path]:
    """Write machine-readable and human-readable quality outputs."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "quality_score.json"
        markdown_path = output_dir / "quality_score.md"
        csv_path = output_dir / "quality_score_components.csv"
        json_path.write_text(
            json.dumps(report.as_dict(), indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(report.as_markdown(), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "key",
                    "title",
                    "weight",
                    "score",
                    "status",
                    "summary",
                    "reasons",
                    "warnings",
                ),
            )
            writer.writeheader()
            for component in report.components:
                writer.writerow(
                    {
                        "key": component.key,
                        "title": component.title,
                        "weight": component.weight,
                        "score": f"{component.score:.2f}",
                        "status": component.status,
                        "summary": component.summary,
                        "reasons": " | ".join(component.reasons),
                        "warnings": " | ".join(component.warnings),
                    }
                )
    except OSError as exc:
        logger.exception("Could not write quality outputs to {}", output_dir)
        raise RuntimeError(f"Could not write quality outputs: {output_dir}") from exc
    logger.info("Wrote quality outputs to {}", output_dir)
    return {
        "quality_json": json_path,
        "quality_markdown": markdown_path,
        "quality_csv": csv_path,
    }


def load_quality_report(path: Path) -> QualityScoreReport:
    """Load a previously exported quality report from JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        components = tuple(
            QualityComponent(
                key=str(item["key"]),
                title=str(item["title"]),
                weight=int(item["weight"]),
                score=float(item["score"]),
                status=str(item["status"]),
                summary=str(item["summary"]),
                reasons=tuple(str(value) for value in item.get("reasons", [])),
                warnings=tuple(str(value) for value in item.get("warnings", [])),
            )
            for item in payload.get("components", [])
        )
        return QualityScoreReport(
            generated_at=datetime.fromisoformat(payload["generated_at"]),
            total_score=float(payload["total_score"]),
            max_score=int(payload["max_score"]),
            overall_status=str(payload["overall_status"]),
            classification_confidence=str(payload["classification_confidence"]),
            components=components,
            warnings=tuple(str(value) for value in payload.get("warnings", [])),
            accuracy_metrics={
                str(key): dict(value)
                for key, value in payload.get("accuracy_metrics", {}).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.exception("Could not load quality report {}", path)
        raise RuntimeError(f"Could not read quality report: {path}") from exc


def _boundary_component(
    spec: RunSpecification,
    boundary_path: Path,
    area_km2: float,
    boundary: Any,
) -> QualityComponent:
    source = (spec.location.boundary_source or "").lower()
    if (
        "local" in source
        or "verified" in source
        or boundary_path.suffix.lower()
        in {
            ".geojson",
            ".gpkg",
            ".shp",
        }
    ):
        source_points = 10.0
        source_text = "local or preserved boundary source"
    elif "geoboundaries" in source:
        source_points = 9.0
        source_text = "geoBoundaries source"
    elif "gadm" in source:
        source_points = 8.5
        source_text = "GADM source"
    elif "openstreetmap" in source or "nominatim" in source or "osm" in source:
        source_points = 8.0
        source_text = "OpenStreetMap administrative source"
    elif "natural earth" in source:
        source_points = 6.0
        source_text = "Natural Earth fallback source"
    elif "bbox" in source:
        source_points = 3.5
        source_text = "bounding-box fallback source"
    else:
        source_points = 7.0
        source_text = "user-confirmed boundary source"
    geometry_points = 4.0 if bool(boundary.geometry.is_valid) else 0.0
    crs_points = 3.0 if bool(boundary.crs) else 0.0
    area_plausible = math.isfinite(area_km2) and 0.1 <= area_km2 <= 20_000_000.0
    area_points = 2.0 if area_plausible else 0.0
    confirmation_points = 1.0 if spec.location.boundary_path is not None else 0.0
    warnings: list[str] = []
    if not boundary.geometry.is_valid:
        warnings.append("Boundary geometry is invalid and should be repaired.")
    if not boundary.crs:
        warnings.append("Boundary CRS is missing.")
    if not area_plausible:
        warnings.append("Boundary area failed the AOI sanity check.")
    score = min(
        20.0,
        source_points
        + geometry_points
        + crs_points
        + area_points
        + confirmation_points,
    )
    return QualityComponent(
        key="boundary",
        title="Boundary confidence",
        weight=20,
        score=score,
        status=_component_status(score, 20.0),
        summary=(
            f"Boundary source is {source_text}; area is {area_km2:,.2f} km2 and "
            f"geometry validity is {boundary.geometry.is_valid}."
        ),
        reasons=(
            f"Source assessment: {source_text}.",
            f"Geometry valid: {boundary.geometry.is_valid}.",
            f"CRS present: {bool(boundary.crs)}.",
            f"AOI area sanity check passed: {area_plausible}.",
        ),
        warnings=tuple(warnings),
    )


def _imagery_component(
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
) -> QualityComponent:
    warnings: list[str] = []
    if availability is None:
        scene_count = len(sources)
        score = min(15.0, 6.0 + min(scene_count, 3) * 3.0)
        if scene_count < 2:
            warnings.append("Very few scenes were available for scoring.")
        return QualityComponent(
            key="imagery",
            title="Imagery availability",
            weight=15,
            score=score,
            status=_component_status(score, 15.0),
            summary=f"{scene_count} source scene(s) were available.",
            reasons=(f"Scene count: {scene_count}.",),
            warnings=tuple(warnings),
        )
    minimum = max(availability.minimum_scenes_per_year, 1)
    years = tuple(sorted(availability.years))
    adequate = sum(
        1 for item in availability.years.values() if item.scene_count >= minimum
    )
    mean_scene_ratio = sum(
        min(item.scene_count / minimum, 2.0) for item in availability.years.values()
    ) / max(len(availability.years), 1)
    mean_coverage = sum(
        item.aoi_coverage for item in availability.years.values()
    ) / max(len(availability.years), 1)
    score = min(
        15.0,
        (adequate / max(len(years), 1)) * 7.0
        + (mean_scene_ratio / 2.0) * 4.0
        + mean_coverage * 4.0,
    )
    if availability.used_fallback:
        score = max(score - min(len(availability.fallback_messages), 2) * 1.5, 0.0)
        warnings.extend(availability.fallback_messages)
    if mean_coverage < 0.95:
        warnings.append("Planned AOI coverage is below the ideal threshold.")
    return QualityComponent(
        key="imagery",
        title="Imagery availability",
        weight=15,
        score=score,
        status=_component_status(score, 15.0),
        summary=(
            f"{adequate}/{len(years)} requested year(s) met the minimum scene policy "
            f"for {availability.dataset}."
        ),
        reasons=(
            f"Dataset: {availability.dataset}.",
            f"Minimum scenes per year: {minimum}.",
            f"Average planned AOI coverage: {mean_coverage:.1%}.",
        ),
        warnings=tuple(warnings),
    )


def _cloud_component(
    valid_t1: float,
    valid_t2: float,
    invalid_t1: float,
    invalid_t2: float,
    avg_cloud: float | None,
) -> QualityComponent:
    average_valid = (valid_t1 + valid_t2) / 2.0
    average_invalid = (invalid_t1 + invalid_t2) / 2.0
    cloud_factor = 0.75 if avg_cloud is None else max(0.0, 1.0 - (avg_cloud / 100.0))
    score = min(
        20.0, average_valid * 11.0 + (1.0 - average_invalid) * 5.0 + cloud_factor * 4.0
    )
    warnings: list[str] = []
    if average_valid < 0.75:
        warnings.append(
            "Cloud-free valid AOI coverage is below the recommended threshold."
        )
    if average_invalid > 0.30:
        warnings.append("Cloud or nodata coverage is high inside the AOI.")
    return QualityComponent(
        key="cloud_nodata",
        title="Cloud-free coverage",
        weight=20,
        score=score,
        status=_component_status(score, 20.0),
        summary=(
            f"AOI-valid coverage is {average_valid:.1%} on average, with "
            f"{average_invalid:.1%} invalid pixels inside the boundary."
        ),
        reasons=(
            f"Start valid AOI coverage: {valid_t1:.1%}.",
            f"End valid AOI coverage: {valid_t2:.1%}.",
            "Cloud, shadow, snow, fill, saturation, and nodata masking were considered "
            "through raster validity and scene-level cloud metadata.",
        ),
        warnings=tuple(warnings),
    )


def _sensor_component(sources: Sequence[SceneMetadata]) -> QualityComponent:
    datasets = sorted({source.dataset for source in sources})
    warnings: list[str] = []
    if not datasets:
        return QualityComponent(
            key="sensor",
            title="Sensor consistency",
            weight=10,
            score=3.0,
            status="Low",
            summary="No scene metadata were available to confirm sensor consistency.",
            reasons=("Sensor metadata were unavailable.",),
            warnings=("Sensor consistency could not be verified.",),
        )
    if len(datasets) == 1:
        score = 10.0
        summary = f"All compared scenes use {datasets[0]}."
    elif all(dataset.startswith("landsat-") for dataset in datasets):
        score = 6.5
        summary = "Multiple Landsat missions were used across the comparison."
        warnings.append(
            "Cross-mission Landsat comparisons should be interpreted carefully "
            "unless harmonization is documented."
        )
    else:
        score = 4.0
        summary = "Mixed sensor families were used across the comparison."
        warnings.append(
            "Cross-sensor comparison lowers run reliability unless harmonization "
            "is explicitly validated."
        )
    return QualityComponent(
        key="sensor",
        title="Sensor consistency",
        weight=10,
        score=score,
        status=_component_status(score, 10.0),
        summary=summary,
        reasons=(f"Datasets observed: {', '.join(datasets)}.",),
        warnings=tuple(warnings),
    )


def _season_component(
    spec: RunSpecification,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
) -> QualityComponent:
    warnings: list[str] = []
    score = 10.0
    requested = (spec.temporal.start_month, spec.temporal.end_month)
    if availability is not None:
        effective = (
            availability.effective_start_month,
            availability.effective_end_month,
        )
        if effective != requested:
            score -= 3.0
            warnings.append(
                "Seasonal window expanded from months "
                f"{requested[0]}-{requested[1]} to "
                f"{effective[0]}-{effective[1]}."
            )
    months = sorted(
        {
            source.acquired_at.month
            for source in sources
            if source.acquired_at is not None
        }
    )
    if months:
        spread = max(months) - min(months)
        if spread > 2:
            score -= 2.0
            warnings.append(
                "Scene acquisition months vary noticeably across the comparison."
            )
        if any(month < requested[0] or month > requested[1] for month in months):
            score -= 2.0
            warnings.append(
                "One or more scene dates fall outside the originally requested season."
            )
    score = max(score, 0.0)
    month_text = ", ".join(str(month) for month in months) if months else "unknown"
    return QualityComponent(
        key="season",
        title="Season consistency",
        weight=10,
        score=score,
        status=_component_status(score, 10.0),
        summary=(
            f"Requested months {requested[0]}-{requested[1]}; observed scene "
            f"months: {month_text}."
        ),
        reasons=(
            f"Requested seasonal window: {requested[0]}-{requested[1]}.",
            f"Observed acquisition months: {month_text}.",
        ),
        warnings=tuple(warnings),
    )


def _processing_component(
    spec: RunSpecification,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    downloads: Mapping[str, Path],
) -> QualityComponent:
    requested_indices = set(spec.analysis.indices) | {"ndvi", "ndbi", "ndwi"}
    generated_indices = set(analytics.index_results)
    index_fraction = len(generated_indices & requested_indices) / max(
        len(requested_indices), 1
    )
    map_count = len(maps)
    report_hits = sum(
        1
        for key in (
            "html_report",
            "pdf_report",
            "dashboard",
            "interpretation",
        )
        if key in downloads
    )
    score = min(
        15.0,
        index_fraction * 5.0
        + (3.0 if analytics.change_results else 0.0)
        + (
            2.0
            if analytics.classification_results
            or spec.analysis.classification == "none"
            else 0.0
        )
        + min(map_count, 3) * 1.0
        + min(report_hits, 4) * 0.5
        + (
            2.0
            if "quality_json" in downloads or "quality_markdown" in downloads
            else 0.0
        ),
    )
    warnings: list[str] = []
    if index_fraction < 0.75:
        warnings.append("Not all requested indices were generated.")
    if not analytics.change_results:
        warnings.append("No change-detection products were generated.")
    if report_hits < 4:
        warnings.append("One or more publication reports are missing.")
    return QualityComponent(
        key="processing",
        title="Report completeness",
        weight=15,
        score=score,
        status=_component_status(score, 15.0),
        summary=(
            f"{len(generated_indices)}/{len(requested_indices)} requested indices, "
            f"{len(analytics.change_results)} change method(s), "
            f"{map_count} map theme(s), "
            f"and {report_hits} report artifact(s) were exported."
        ),
        reasons=(
            f"Generated indices: {', '.join(sorted(generated_indices)) or 'none'}.",
            "Generated change methods: "
            f"{', '.join(sorted(analytics.change_results)) or 'none'}.",
            f"Generated maps: {map_count}.",
        ),
        warnings=tuple(warnings),
    )


def _classification_component(
    spec: RunSpecification,
    analytics: AnalyticsReport,
) -> QualityComponent:
    method = spec.analysis.classification
    if method == "none":
        return QualityComponent(
            key="classification",
            title="Classification confidence",
            weight=10,
            score=7.0,
            status="Not applicable",
            summary="No LULC classification was requested for this run.",
            reasons=("Classification method was set to none.",),
        )
    if method in {"kmeans", "isodata"}:
        return QualityComponent(
            key="classification",
            title="Classification confidence",
            weight=10,
            score=3.5 if analytics.classification_results else 2.0,
            status="Exploratory",
            summary=(
                "Unsupervised LULC was generated and should not be treated as "
                "validated land-cover accuracy."
            ),
            reasons=(
                f"Classification method: {method}.",
                "Classification outputs generated: "
                f"{bool(analytics.classification_results)}.",
            ),
            warnings=(
                "LULC was generated using unsupervised classification. Treat "
                "class labels as exploratory.",
            ),
        )
    if analytics.accuracy:
        overall = [
            assessment.overall_accuracy for assessment in analytics.accuracy.values()
        ]
        kappas = [assessment.kappa for assessment in analytics.accuracy.values()]
        mean_overall = sum(overall) / len(overall)
        mean_kappa = sum(kappas) / len(kappas)
        score = min(10.0, 4.5 + (mean_overall * 4.0) + max(mean_kappa, 0.0) * 1.5)
        return QualityComponent(
            key="classification",
            title="Classification confidence",
            weight=10,
            score=score,
            status=_component_status(score, 10.0),
            summary=(
                "Supervised classification has validation metrics with mean "
                f"overall accuracy {mean_overall:.2%} and mean kappa "
                f"{mean_kappa:.3f}."
            ),
            reasons=(
                f"Classification method: {method}.",
                f"Accuracy assessments available: {len(analytics.accuracy)}.",
            ),
        )
    return QualityComponent(
        key="classification",
        title="Classification confidence",
        weight=10,
        score=5.0 if analytics.classification_results else 2.5,
        status="Limited",
        summary=(
            "Supervised classification outputs exist, but no independent "
            "validation metrics were provided."
        ),
        reasons=(
            f"Classification method: {method}.",
            "Classification outputs generated: "
            f"{bool(analytics.classification_results)}.",
        ),
        warnings=(
            "Supervised classification did not include independent validation metrics.",
        ),
    )


def _classification_confidence(
    spec: RunSpecification,
    analytics: AnalyticsReport,
) -> str:
    method = spec.analysis.classification
    if method in {"kmeans", "isodata"}:
        return "Exploratory"
    if method == "none":
        return "Not applicable"
    if analytics.accuracy:
        return "Validated"
    if analytics.classification_results:
        return "Limited"
    return "Unavailable"


def _accuracy_metrics(
    accuracy: Mapping[str, AccuracyAssessment],
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for name, item in accuracy.items():
        metrics[name] = {
            "overall_accuracy": round(item.overall_accuracy, 4),
            "kappa": round(item.kappa, 4),
            "per_class_accuracy": {
                key: round(value, 4) for key, value in item.per_class_accuracy.items()
            },
        }
    return metrics


def _boundary_mask(boundary_path: Path, scene: RasterLayer) -> np.ndarray:
    """Return the approved AOI mask on the raster grid."""
    boundary = load_vector_geometry(boundary_path)
    geometry = reproject_geometry(boundary.geometry, boundary.crs, scene.grid.crs)
    return geometry_mask_for_grid(geometry, scene.grid)


def _valid_coverage(scene: RasterLayer, mask: np.ndarray) -> float:
    """Return the fraction of AOI pixels with finite non-cloud data."""
    denominator = int(mask.sum())
    if denominator == 0:
        return 0.0
    metadata_fraction = scene.metadata.get("valid_aoi_fraction")
    if isinstance(metadata_fraction, (int, float)) and math.isfinite(metadata_fraction):
        return float(max(0.0, min(1.0, metadata_fraction)))
    valid = np.isfinite(scene.data[0])
    if scene.cloud_mask is not None:
        valid &= ~scene.cloud_mask
    return float((valid & mask).sum() / denominator)


def _average_cloud_cover(sources: Sequence[SceneMetadata]) -> float | None:
    """Return the mean scene cloud cover when available."""
    values = [
        value
        for value in (source.cloud_cover for source in sources)
        if value is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _boundary_area_km2(geometry: Any, crs_value: object) -> float:
    """Calculate geodesic boundary area in square kilometres."""
    crs = CRS.from_user_input(crs_value or "EPSG:4326")
    if not crs.is_geographic:
        geod = Geod(ellps="WGS84")
        transformed = reproject_geometry(geometry, str(crs), "EPSG:4326")
        return float(abs(geod.geometry_area_perimeter(transformed)[0]) / 1_000_000.0)
    geod = Geod(ellps="WGS84")
    return float(abs(geod.geometry_area_perimeter(geometry)[0]) / 1_000_000.0)


def _component_status(score: float, weight: float) -> str:
    """Map a weighted score to a human-readable band."""
    ratio = 0.0 if weight <= 0 else score / weight
    if ratio >= 0.85:
        return "High"
    if ratio >= 0.60:
        return "Medium"
    if ratio >= 0.35:
        return "Limited"
    return "Low"


def _overall_status(score: float) -> str:
    """Map the overall score to a report-friendly quality label."""
    if score >= 85.0:
        return "High"
    if score >= 70.0:
        return "Good"
    if score >= 55.0:
        return "Moderate"
    if score >= 40.0:
        return "Limited"
    return "Low"


def _unique(values: Sequence[str] | Any) -> tuple[str, ...]:
    """Return unique non-empty strings while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
