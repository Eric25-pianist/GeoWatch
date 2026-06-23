"""Rule-based scientific interpretation for GeoWatch reports."""

# ruff: noqa: E501

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger
from pyproj import CRS, Geod

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import (
    AnalyticsReport,
    ClassificationResult,
    SignedChangeResult,
)
from geowatch.application.availability import AvailabilityPlan
from geowatch.application.models import RunSpecification
from geowatch.processing.models import RasterLayer
from geowatch.utils.geometry import (
    geometry_mask_for_grid,
    load_vector_geometry,
    reproject_geometry,
)
from geowatch.validation.quality_score import (
    QualityScoreReport,
    calculate_quality_score,
)


@dataclass(frozen=True)
class InterpretationSection:
    """One titled interpretation section."""

    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class InterpretationReport:
    """Structured rule-based interpretation generated from GeoWatch outputs."""

    generated_at: datetime
    sections: tuple[InterpretationSection, ...]

    def as_markdown(self) -> str:
        """Render the interpretation as standalone Markdown."""
        lines = [
            "# GeoWatch Interpretation",
            "",
            f"- Generated: {self.generated_at.isoformat()}",
            "- Method: offline deterministic rule-based interpretation",
            "",
        ]
        for section in self.sections:
            lines.extend([f"## {section.title}", ""])
            for paragraph in section.paragraphs:
                lines.extend([paragraph, ""])
        return "\n".join(lines).rstrip() + "\n"

    def as_html(self) -> str:
        """Render the interpretation as embeddable HTML."""
        chunks: list[str] = []
        for section in self.sections:
            paragraphs = "".join(
                f"<p>{html.escape(paragraph)}</p>" for paragraph in section.paragraphs
            )
            chunks.append(
                f'<article class="interpretation-card"><h3>{html.escape(section.title)}</h3>{paragraphs}</article>'
            )
        return "".join(chunks)


def write_interpretation(
    output_path: Path,
    *,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None = None,
    quality_report: QualityScoreReport | None = None,
) -> Path:
    """Generate and write a standalone Markdown interpretation report."""
    report = generate_interpretation(
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
        quality_report=quality_report,
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.as_markdown(), encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not write interpretation report {}", output_path)
        raise RuntimeError(
            f"Could not write interpretation report: {output_path}"
        ) from exc
    logger.info("Wrote rule-based interpretation to {}", output_path)
    return output_path


def generate_interpretation(
    *,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None = None,
    quality_report: QualityScoreReport | None = None,
) -> InterpretationReport:
    """Generate deterministic analyst-style interpretation from structured results."""
    metrics = _quality_metrics(boundary_path, scene_t1, scene_t2)
    resolved_quality = quality_report or calculate_quality_score(
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
    )
    sections = (
        _executive_section(spec, analytics, sources, availability),
        _vegetation_section(spec, scene_t2, analytics),
        _water_section(analytics),
        _urban_section(analytics),
        _soil_agriculture_section(analytics),
        _hotspot_section(analytics),
        _quality_section(
            spec,
            metrics,
            analytics,
            sources,
            availability,
            resolved_quality,
        ),
        _recommendations_section(analytics, metrics),
    )
    return InterpretationReport(
        generated_at=datetime.now(UTC),
        sections=tuple(section for section in sections if section.paragraphs),
    )


def render_interpretation_html(report: InterpretationReport) -> str:
    """Render a complete interpretation block for HTML reports."""
    return (
        '<section class="interpretation-block">'
        "<h2>Analyst Interpretation</h2>"
        f"{report.as_html()}"
        "</section>"
    )


def _executive_section(
    spec: RunSpecification,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
) -> InterpretationSection:
    mission = _mission_text(sources)
    methods = ", ".join(sorted(analytics.change_results)) or "spectral comparison"
    location = spec.location.name
    period = f"{spec.temporal.start_year}-{spec.temporal.end_year}"
    paragraphs = [
        (
            f"GeoWatch analysed {location} for {period} using {mission}. "
            f"The workflow compared seasonally consistent endpoint imagery with {methods} "
            "and polygon masking to the confirmed area of interest."
        )
    ]
    ndvi = analytics.index_results.get("ndvi")
    primary_change = _primary_change_fraction(analytics)
    if ndvi is not None and math.isfinite(ndvi.statistics.difference.mean):
        direction = _direction(ndvi.statistics.difference.mean, 0.015)
        paragraphs.append(
            f"The mean NDVI signal {direction} by {ndvi.statistics.difference.mean:+.4f}. "
            f"The primary change surface marks approximately {primary_change:.1%} of valid pixels as changed when a thresholded change layer is available."
        )
    else:
        paragraphs.append(
            "The available outputs support a structured review of spectral and land-cover indicators, but no NDVI summary was available for a vegetation-centred headline."
        )
    if availability is not None and availability.fallback_messages:
        paragraphs.append(
            "Scene selection required fallback handling: "
            + "; ".join(availability.fallback_messages)
            + ". These choices should be considered when comparing subtle changes."
        )
    return InterpretationSection("Executive interpretation", tuple(paragraphs))


def _vegetation_section(
    spec: RunSpecification,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
) -> InterpretationSection:
    paragraphs: list[str] = []
    ndvi = analytics.index_results.get("ndvi")
    if ndvi is not None:
        diff = ndvi.statistics.difference.mean
        paragraphs.append(
            f"NDVI changed from a mean of {ndvi.statistics.t1.mean:.4f} in {spec.temporal.start_year} to {ndvi.statistics.t2.mean:.4f} in {spec.temporal.end_year}, with a mean difference of {diff:+.4f}. This should be read as spectral vegetation signal change, not direct ecological ground truth."
        )
    if analytics.signed_change is not None:
        signed_text = _signed_change_sentence(analytics.signed_change, scene_t2)
        paragraphs.append(
            f"Using the documented NDVI threshold of +/-{analytics.signed_change.threshold:.4f}, {signed_text}. Gain and loss classes identify pixels whose vegetation index moved beyond the selected threshold."
        )
    if not paragraphs:
        paragraphs.append(
            "Vegetation-specific interpretation is limited because NDVI or NDVI gain/loss statistics were not available in this run."
        )
    return InterpretationSection("Vegetation interpretation", tuple(paragraphs))


def _water_section(analytics: AnalyticsReport) -> InterpretationSection:
    paragraphs: list[str] = []
    for index_name in ("ndwi", "mndwi"):
        result = analytics.index_results.get(index_name)
        if result is None:
            continue
        paragraphs.append(
            f"{index_name.upper()} mean change is {result.statistics.difference.mean:+.4f}. Positive values may indicate stronger water or moisture-related spectral response, while negative values may indicate reduced water signal or seasonal/background effects."
        )
    if not paragraphs:
        paragraphs.append(
            "Water interpretation is limited because NDWI or MNDWI statistics were not available. Water conclusions should therefore be based on the exported maps only after visual and reference-data review."
        )
    return InterpretationSection("Water interpretation", tuple(paragraphs))


def _urban_section(analytics: AnalyticsReport) -> InterpretationSection:
    paragraphs: list[str] = []
    ndbi = analytics.index_results.get("ndbi")
    if ndbi is not None:
        paragraphs.append(
            f"NDBI mean change is {ndbi.statistics.difference.mean:+.4f}. Higher NDBI values can be consistent with stronger built-up or impervious-surface signal, but mixed pixels, bare soil, and dry surfaces can produce similar spectral responses."
        )
    lulc_t1 = analytics.classification_results.get("lulc_t1")
    lulc_t2 = analytics.classification_results.get("lulc_t2")
    if lulc_t1 is not None and lulc_t2 is not None:
        urban_delta = _class_delta(lulc_t1, lulc_t2, ("Urban", "Built-up", "Developed"))
        if urban_delta is not None:
            paragraphs.append(
                f"The exploratory LULC output shows a built-up/urban class pixel change of {urban_delta:+,}. This is a classification signal and should be validated before being used as a definitive urban expansion estimate."
            )
    if not paragraphs:
        paragraphs.append(
            "Built-up interpretation is limited because neither NDBI nor comparable urban LULC class statistics were available."
        )
    return InterpretationSection("Built-up and urban interpretation", tuple(paragraphs))


def _soil_agriculture_section(analytics: AnalyticsReport) -> InterpretationSection:
    paragraphs: list[str] = []
    bsi = analytics.index_results.get("bsi")
    if bsi is not None:
        paragraphs.append(
            f"BSI mean change is {bsi.statistics.difference.mean:+.4f}. BSI can highlight exposed soil or dry bright surfaces, but it does not by itself prove crop conversion or soil degradation."
        )
    lulc_t1 = analytics.classification_results.get("lulc_t1")
    lulc_t2 = analytics.classification_results.get("lulc_t2")
    if lulc_t1 is not None and lulc_t2 is not None:
        for label in ("Agriculture", "Bare Soil", "Vegetation"):
            delta = _class_delta(lulc_t1, lulc_t2, (label,))
            if delta is not None:
                paragraphs.append(
                    f"The exploratory LULC class '{label}' changed by {delta:+,} pixels between endpoints."
                )
    if not paragraphs:
        paragraphs.append(
            "Bare-soil and agriculture interpretation is limited because BSI or relevant LULC class summaries were not available."
        )
    paragraphs.append(
        "Crop-specific claims require crop calendars, field samples, or independent reference labels; GeoWatch only reports spectral and classification evidence from the supplied imagery."
    )
    return InterpretationSection(
        "Bare soil and agriculture interpretation", tuple(paragraphs)
    )


def _hotspot_section(analytics: AnalyticsReport) -> InterpretationSection:
    hotspot_artifacts = [
        name for name in analytics.artifacts if "hotspot" in name.lower()
    ]
    primary = _primary_change_fraction(analytics)
    paragraphs = [
        "Hotspot outputs, when generated, identify spatial clustering of change intensity rather than a land-cover class. A positive Getis-Ord Gi* z-score means nearby pixels have unusually high change scores compared with the broader analysis surface."
    ]
    if hotspot_artifacts:
        paragraphs.append(
            "This run includes hotspot-related artifacts: "
            + ", ".join(sorted(hotspot_artifacts))
            + ". These should be inspected with the boundary and basemap to understand whether clusters follow real urban, vegetation, or water features."
        )
    else:
        paragraphs.append(
            f"No dedicated hotspot statistics were provided to the interpretation engine. The primary thresholded change fraction is {primary:.1%}, but clustering should be interpreted from the exported hotspot map only if it exists."
        )
    return InterpretationSection("Hotspot interpretation", tuple(paragraphs))


def _quality_section(
    spec: RunSpecification,
    metrics: Mapping[str, float | bool | str],
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
    quality_report: QualityScoreReport,
) -> InterpretationSection:
    valid_t1 = _as_float(metrics.get("valid_t1"))
    valid_t2 = _as_float(metrics.get("valid_t2"))
    nodata_t1 = _as_float(metrics.get("nodata_t1"))
    nodata_t2 = _as_float(metrics.get("nodata_t2"))
    sensor_count = len({source.dataset for source in sources})
    boundary_component = quality_report.component("boundary")
    cloud_component = quality_report.component("cloud_nodata")
    season_component = quality_report.component("season")
    paragraphs = [
        (
            f"GeoWatch Quality Score for this run is {quality_report.rounded_score}/{quality_report.max_score}. The run is rated {quality_report.overall_status.lower()} overall, with boundary confidence {boundary_component.status.lower() if boundary_component else 'unknown'}, cloud-free coverage {cloud_component.status.lower() if cloud_component else 'unknown'}, season consistency {season_component.status.lower() if season_component else 'unknown'}, and classification confidence {quality_report.classification_confidence.lower()}."
        ),
        (
            f"AOI-relative valid coverage is {valid_t1:.1%} for the start endpoint and {valid_t2:.1%} for the end endpoint. Cloud/no-data coverage is {nodata_t1:.1%} and {nodata_t2:.1%}, respectively."
        ),
        (
            "Sensor consistency is "
            + (
                "good because all source scenes use one dataset."
                if sensor_count <= 1
                else "mixed; cross-sensor differences should be considered carefully."
            )
        ),
        (
            f"The comparison uses months {spec.temporal.start_month}-{spec.temporal.end_month}. Seasonal consistency reduces phenology bias but does not eliminate short-term rainfall, irrigation, tide, or atmospheric effects."
        ),
        (
            f"Boundary source: {spec.location.boundary_source or 'user-confirmed boundary'}. Boundary quality affects all clipped areas, class summaries, and visual map footprints."
        ),
    ]
    if any("landsat-7" in source.dataset for source in sources):
        paragraphs.append(
            "Landsat 7 imagery can contain SLC-off striping after May 2003; compositing and valid-pixel checks reduce this issue but do not replace manual QA."
        )
    if analytics.accuracy:
        paragraphs.append(
            "A supervised accuracy assessment object is present, so accuracy metrics should be read together with its sampling design and reference labels."
        )
    else:
        paragraphs.append(
            "No independent reference samples were provided to the interpretation engine; therefore no classification accuracy claim is made."
        )
    if quality_report.warnings:
        paragraphs.append("Quality warnings: " + "; ".join(quality_report.warnings))
    if availability is not None and availability.fallback_messages:
        paragraphs.append(
            "Acquisition fallback was used: "
            + "; ".join(availability.fallback_messages)
        )
    return InterpretationSection("Data quality and uncertainty", tuple(paragraphs))


def _recommendations_section(
    analytics: AnalyticsReport,
    metrics: Mapping[str, float | bool | str],
) -> InterpretationSection:
    valid_t1 = _as_float(metrics.get("valid_t1"))
    valid_t2 = _as_float(metrics.get("valid_t2"))
    recommendations = [
        "Review the change and index maps in QGIS or ArcGIS together with high-resolution reference imagery before making planning or policy decisions.",
        "Collect independent validation samples or field observations if LULC classes will be used as final land-cover evidence.",
        "Use supervised Random Forest, SVM, or XGBoost only when reliable labeled training and validation samples are available.",
    ]
    if min(valid_t1, valid_t2) < 0.8:
        recommendations.append(
            "Consider widening the seasonal window or adding more scenes because valid AOI coverage is below 80% for at least one endpoint."
        )
    if analytics.signed_change is not None:
        recommendations.append(
            "Inspect NDVI gain/loss patches manually to separate vegetation recovery, crop-cycle effects, wetness changes, and mixed urban vegetation pixels."
        )
    return InterpretationSection("Recommended next steps", tuple(recommendations))


def _quality_metrics(
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> dict[str, float | bool | str]:
    """Calculate AOI area and AOI-relative valid/nodata coverage."""
    try:
        boundary = load_vector_geometry(boundary_path)
        geod = Geod(ellps="WGS84")
        geometry_wgs84 = reproject_geometry(
            boundary.geometry, boundary.crs, "EPSG:4326"
        )
        area_m2, _ = geod.geometry_area_perimeter(geometry_wgs84)
        projected_t1 = reproject_geometry(
            boundary.geometry, boundary.crs, scene_t1.grid.crs
        )
        projected_t2 = reproject_geometry(
            boundary.geometry, boundary.crs, scene_t2.grid.crs
        )
        mask_t1 = geometry_mask_for_grid(projected_t1, scene_t1.grid)
        mask_t2 = geometry_mask_for_grid(projected_t2, scene_t2.grid)
        valid_t1, nodata_t1 = _valid_and_nodata(scene_t1, mask_t1)
        valid_t2, nodata_t2 = _valid_and_nodata(scene_t2, mask_t2)
        return {
            "area_km2": abs(float(area_m2)) / 1_000_000.0,
            "valid_t1": valid_t1,
            "valid_t2": valid_t2,
            "nodata_t1": nodata_t1,
            "nodata_t2": nodata_t2,
            "aligned": _grids_aligned(scene_t1, scene_t2),
        }
    except (OSError, ValueError) as exc:
        logger.warning("Interpretation QA metric calculation failed: {}", exc)
        return {
            "area_km2": float("nan"),
            "valid_t1": _array_valid_fraction(scene_t1),
            "valid_t2": _array_valid_fraction(scene_t2),
            "nodata_t1": 1.0 - _array_valid_fraction(scene_t1),
            "nodata_t2": 1.0 - _array_valid_fraction(scene_t2),
            "aligned": _grids_aligned(scene_t1, scene_t2),
        }


def _valid_and_nodata(scene: RasterLayer, inside: np.ndarray) -> tuple[float, float]:
    denominator = int(inside.sum())
    if denominator == 0:
        return 0.0, 1.0
    finite = np.isfinite(scene.data[0])
    if scene.cloud_mask is not None:
        finite &= ~scene.cloud_mask
    valid = float((finite & inside).sum() / denominator)
    return valid, 1.0 - valid


def _array_valid_fraction(scene: RasterLayer) -> float:
    finite = np.isfinite(scene.data[0])
    if scene.cloud_mask is not None:
        finite &= ~scene.cloud_mask
    return float(finite.sum() / finite.size) if finite.size else 0.0


def _grids_aligned(scene_t1: RasterLayer, scene_t2: RasterLayer) -> bool:
    grid_t1 = scene_t1.grid
    grid_t2 = scene_t2.grid
    return bool(
        grid_t1.crs == grid_t2.crs
        and grid_t1.width == grid_t2.width
        and grid_t1.height == grid_t2.height
        and np.allclose(grid_t1.transform, grid_t2.transform, rtol=0.0, atol=1e-9)
    )


def _mission_text(sources: Sequence[SceneMetadata]) -> str:
    datasets = sorted({source.dataset for source in sources})
    if not datasets:
        return "the available satellite scene metadata"
    return ", ".join(_dataset_label(dataset) for dataset in datasets)


def _dataset_label(dataset: str) -> str:
    labels = {
        "sentinel-2-l2a": "Sentinel-2 Level-2A",
        "landsat-5-c2-l2": "Landsat 5 Collection 2 Level 2",
        "landsat-7-c2-l2": "Landsat 7 Collection 2 Level 2",
        "landsat-8-c2-l2": "Landsat 8 Collection 2 Level 2",
        "landsat-9-c2-l2": "Landsat 9 Collection 2 Level 2",
    }
    return labels.get(dataset, dataset)


def _primary_change_fraction(analytics: AnalyticsReport) -> float:
    for result in analytics.change_results.values():
        if result.threshold is not None:
            return result.threshold.change_fraction
    return 0.0


def _direction(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "increased"
    if value < -tolerance:
        return "decreased"
    return "remained broadly stable"


def _signed_change_sentence(
    signed: SignedChangeResult,
    scene: RasterLayer,
) -> str:
    pixel_area_km2 = _pixel_area_m2(scene) / 1_000_000.0
    total = sum(max(0, count) for count in signed.counts.values())
    if total == 0:
        return "no signed NDVI classes contain valid pixels"
    parts = []
    for name in signed.class_names:
        count = signed.counts.get(name, 0)
        fraction = count / total
        area = count * pixel_area_km2
        parts.append(f"{name.lower()} covers {area:,.2f} km2 ({fraction:.1%})")
    return ", ".join(parts)


def _pixel_area_m2(scene: RasterLayer) -> float:
    a, b, _, d, e, _ = scene.grid.transform
    native_area = abs((a * e) - (b * d))
    try:
        crs = CRS.from_user_input(scene.grid.crs)
    except Exception:  # pragma: no cover - pyproj can raise several CRS errors.
        return float(native_area)
    if crs.axis_info and crs.axis_info[0].unit_conversion_factor:
        factor = crs.axis_info[0].unit_conversion_factor
        return float(native_area * factor * factor)
    return float(native_area)


def _class_delta(
    t1: ClassificationResult,
    t2: ClassificationResult,
    labels: Sequence[str],
) -> int | None:
    for label in labels:
        if label in t1.counts or label in t2.counts:
            return int(t2.counts.get(label, 0) - t1.counts.get(label, 0))
    return None


def _as_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else 0.0
