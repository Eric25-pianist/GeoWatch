# ruff: noqa: E501
"""Offline interactive HTML dashboard generation for GeoWatch publications."""

from __future__ import annotations

import html
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import numpy as np
from loguru import logger
from pyproj import CRS, Geod

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import (
    AnalyticsReport,
    ChangeDetectionResult,
    ClassificationResult,
)
from geowatch.application.availability import AvailabilityPlan
from geowatch.application.models import RunSpecification
from geowatch.processing.models import RasterLayer
from geowatch.reporting.interpretation import (
    InterpretationReport,
    generate_interpretation,
    render_interpretation_html,
)
from geowatch.reporting.models import MapArtifact
from geowatch.utils.geometry import (
    geometry_mask_for_grid,
    load_vector_geometry,
    reproject_geometry,
)
from geowatch.validation.quality_score import (
    QualityScoreReport,
    calculate_quality_score,
)

_LULC_COLORS: dict[str, str] = {
    "Water": "#277da1",
    "Urban": "#e76f51",
    "Vegetation": "#43aa8b",
    "Agriculture": "#e9c46a",
    "Bare Soil": "#b08968",
    "Forest": "#2d6a4f",
    "Wetlands": "#4d908e",
    "Snow/Ice": "#b8d8e8",
    "Bright Surface / Uncertain": "#adb5bd",
}

_CHANGE_COLORS: dict[str, str] = {
    "Loss": "#d55e00",
    "No change": "#a7adb4",
    "Gain": "#009e73",
}


class _SpatialMetrics(TypedDict):
    """Calculated spatial and coverage measures used by dashboard cards."""

    area_km2: float
    coverage_t1: float
    coverage_t2: float
    nodata_t1: float
    nodata_t2: float
    quality_ok: bool
    quality_text: str


def write_dashboard(
    output_path: Path,
    *,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None = None,
    downloads: Mapping[str, Path] | None = None,
    interpretation: InterpretationReport | None = None,
    quality_report: QualityScoreReport | None = None,
) -> Path:
    """Write a portable, dependency-free dashboard for one GeoWatch comparison."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document = render_dashboard(
            output_path=output_path,
            spec=spec,
            boundary_path=boundary_path,
            scene_t1=scene_t1,
            scene_t2=scene_t2,
            analytics=analytics,
            maps=maps,
            sources=sources,
            availability=availability,
            downloads=downloads or {},
            interpretation=interpretation,
            quality_report=quality_report,
        )
        output_path.write_text(document, encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.exception("Could not write dashboard {}", output_path)
        raise RuntimeError(f"Could not generate dashboard: {output_path}") from exc
    logger.info("Wrote interactive dashboard to {}", output_path)
    return output_path


def render_dashboard(
    *,
    output_path: Path,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None = None,
    downloads: Mapping[str, Path] | None = None,
    interpretation: InterpretationReport | None = None,
    quality_report: QualityScoreReport | None = None,
) -> str:
    """Render a complete dashboard document without writing it to disk."""
    metrics = _spatial_metrics(boundary_path, scene_t1, scene_t2)
    resolved_quality = quality_report or calculate_quality_score(
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
        maps=maps,
        downloads=downloads or {},
    )
    interpretation_report = interpretation or generate_interpretation(
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
        quality_report=resolved_quality,
    )
    mission = _mission_text(sources, scene_t1, scene_t2)
    scene_count = len({source.scene_id for source in sources})
    generated = datetime.now(UTC)
    map_gallery = _render_map_gallery(maps, output_path)
    slider = _render_slider(maps.get("before_after"), output_path, spec)
    summary_cards = _render_summary_cards(metrics, analytics, resolved_quality)
    index_table = _render_index_table(analytics)
    lulc = _render_lulc(analytics, scene_t2)
    signed_change = _render_signed_change(analytics, scene_t2)
    transition = _render_transition_matrix(analytics)
    scenes = _render_scene_table(sources)
    provenance = _render_provenance(
        spec,
        scene_t1,
        scene_t2,
        sources,
        availability,
        boundary_path,
        generated,
    )
    limitations = _render_limitations(spec, sources)
    download_links = _render_downloads(
        output_path,
        maps,
        analytics,
        downloads or {},
    )
    quality_class = (
        "quality-good"
        if metrics["quality_ok"] and resolved_quality.rounded_score >= 70
        else "quality-warn"
    )
    quality_text = html.escape(_quality_notice_text(resolved_quality))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{html.escape(spec.location.name)} | GeoWatch Dashboard</title>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body>
  <header class="topbar">
    <div class="brand-block">
      <span class="eyebrow">GeoWatch Project</span>
      <h1>{html.escape(spec.location.name)} change dashboard</h1>
      <p>{spec.temporal.start_year} to {spec.temporal.end_year} · {html.escape(mission)}</p>
    </div>
    <dl class="run-facts">
      <div><dt>Scenes</dt><dd>{scene_count}</dd></div>
      <div><dt>AOI area</dt><dd>{_format_area(metrics["area_km2"])}</dd></div>
      <div><dt>Projection</dt><dd>{html.escape(scene_t1.grid.crs)}</dd></div>
      <div><dt>Generated</dt><dd>{generated.date().isoformat()}</dd></div>
    </dl>
  </header>

  <nav class="section-nav" aria-label="Dashboard sections">
    <a href="#overview">Overview</a><a href="#compare">Compare</a>
    <a href="#maps">Maps</a><a href="#analysis">Analysis</a>
    <a href="#interpretation">Interpretation</a>
    <a href="#provenance">Provenance</a><a href="#downloads">Downloads</a>
  </nav>

  <main>
    <section id="overview" class="section-block">
      <div class="section-heading"><p class="kicker">Run overview</p><h2>What the project produced</h2></div>
      <div class="summary-grid">{summary_cards}</div>
      <div class="quality-notice {quality_class}"><strong>Spatial QA</strong><span>{quality_text}</span></div>
    </section>

    <section id="compare" class="section-block">
      <div class="section-heading"><p class="kicker">Visual comparison</p><h2>Before and after</h2><p>Drag the handle to reveal either endpoint. Both images use the same grid, extent, and display dimensions.</p></div>
      {slider}
    </section>

    <section id="maps" class="section-block">
      <div class="section-heading"><p class="kicker">Cartographic outputs</p><h2>Publication map gallery</h2><p>Select any map to open the full-resolution export.</p></div>
      {map_gallery}
    </section>

    <section id="analysis" class="section-block">
      <div class="section-heading"><p class="kicker">Measured results</p><h2>Indices, land cover, and change</h2></div>
      <div class="analysis-grid">
        <article class="data-panel"><h3>Spectral index statistics</h3>{index_table}</article>
        <article class="data-panel"><h3>NDVI gain, stability, and loss</h3>{signed_change}</article>
        <article class="data-panel"><h3>Land-cover area</h3>{lulc}</article>
      </div>
      <article class="matrix-panel"><h3>LULC transition matrix</h3>{transition}</article>
    </section>

    <section id="interpretation" class="section-block">
      <div class="section-heading"><p class="kicker">Analyst narrative</p><h2>What the results mean</h2><p>Generated offline from the measured GeoWatch statistics. The language is intentionally cautious and does not claim accuracy without reference validation.</p></div>
      <div class="interpretation-grid">{render_interpretation_html(interpretation_report)}</div>
    </section>

    <section class="section-block">
      <div class="section-heading"><p class="kicker">How it was made</p><h2>Methodology</h2></div>
      <ol class="workflow-list">
        <li><strong>Boundary validation</strong><span>The selected administrative geometry was checked, repaired when safe, projected, and used as the final pixel mask.</span></li>
        <li><strong>Imagery search</strong><span>Scenes were ranked for AOI coverage, cloud cover, seasonal proximity, mission consistency, and analytical-band completeness.</span></li>
        <li><strong>QA masking and compositing</strong><span>Cloud, shadow, cirrus, snow, saturation, fill, and nodata pixels were excluded before seasonal compositing.</span></li>
        <li><strong>Spatial preparation</strong><span>Endpoint imagery was reprojected to one grid, aligned, and clipped to the confirmed polygon boundary.</span></li>
        <li><strong>Analytics</strong><span>Spectral indices, change methods, NDVI gain/loss, and LULC products were calculated from surface reflectance.</span></li>
        <li><strong>Publication</strong><span>GeoTIFFs, statistics, professional maps, reports, and this offline dashboard were exported together.</span></li>
      </ol>
      <p class="caveat-inline">Unsupervised K-Means and ISODATA classifications are exploratory. GeoWatch does not claim classification accuracy without independent reference labels.</p>
    </section>

    <section id="provenance" class="section-block">
      <div class="section-heading"><p class="kicker">Traceability</p><h2>Data provenance</h2></div>
      {provenance}
      <article class="table-panel"><h3>Acquisition scenes</h3>{scenes}</article>
    </section>

    <section class="section-block">
      <div class="section-heading"><p class="kicker">Responsible interpretation</p><h2>Limitations and caveats</h2></div>
      {limitations}
    </section>

    <section id="downloads" class="section-block">
      <div class="section-heading"><p class="kicker">Project files</p><h2>Downloads and exports</h2><p>Links are relative to this report so the project folder remains portable.</p></div>
      {download_links}
    </section>
  </main>

  <footer><strong>GeoWatch Project</strong><span>Terminal-first geospatial change analysis · {generated.date().isoformat()}</span></footer>
  <script>{_DASHBOARD_JS}</script>
</body>
</html>"""


def _spatial_metrics(
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> _SpatialMetrics:
    """Calculate AOI area, valid coverage, nodata, and boundary-mask QA metrics."""
    boundary = load_vector_geometry(boundary_path)
    geometry_wgs84 = reproject_geometry(boundary.geometry, boundary.crs, "EPSG:4326")
    geod = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(geometry_wgs84)
    coverages: list[float] = []
    outside_counts: list[int] = []
    for scene in (scene_t1, scene_t2):
        projected = reproject_geometry(boundary.geometry, boundary.crs, scene.grid.crs)
        inside = geometry_mask_for_grid(projected, scene.grid)
        valid = np.isfinite(scene.data[0])
        denominator = int(inside.sum())
        coverages.append(
            float((valid & inside).sum() / denominator) if denominator else 0.0
        )
        outside_counts.append(int((valid & ~inside).sum()))
    aligned = _grids_aligned(scene_t1, scene_t2)
    quality_ok = aligned and not any(outside_counts) and boundary.geometry.is_valid
    if quality_ok:
        quality_text = (
            "Boundary geometry is valid, endpoint grids are aligned, and no valid "
            "pixels occur outside the approved AOI."
        )
    else:
        quality_text = (
            f"Review required: boundary valid={boundary.geometry.is_valid}, "
            f"grids aligned={aligned}, outside pixels={sum(outside_counts):,}."
        )
    return {
        "area_km2": abs(float(area_m2)) / 1_000_000.0,
        "coverage_t1": coverages[0],
        "coverage_t2": coverages[1],
        "nodata_t1": 1.0 - coverages[0],
        "nodata_t2": 1.0 - coverages[1],
        "quality_ok": quality_ok,
        "quality_text": quality_text,
    }


def _grids_aligned(scene_t1: RasterLayer, scene_t2: RasterLayer) -> bool:
    """Return whether endpoint scenes use one matching spatial grid."""
    first = scene_t1.grid
    second = scene_t2.grid
    return bool(
        first.crs == second.crs
        and first.width == second.width
        and first.height == second.height
        and np.allclose(first.transform, second.transform, rtol=0.0, atol=1e-9)
    )


def _render_summary_cards(
    metrics: _SpatialMetrics,
    analytics: AnalyticsReport,
    quality_report: QualityScoreReport,
) -> str:
    """Render compact dashboard metric cards."""
    change = _primary_change(analytics)
    change_value = "Unavailable"
    change_detail = "No change product was generated"
    if change is not None:
        fraction = change.threshold.change_fraction if change.threshold else None
        change_value = (
            f"{fraction:.1%}"
            if fraction is not None
            else f"{change.statistics.mean:.3f}"
        )
        change_detail = f"{_humanize(change.method)} change result"
    signed_value = "Unavailable"
    signed_detail = "No signed NDVI product"
    if analytics.signed_change is not None:
        counts = analytics.signed_change.counts
        total = sum(counts.values())
        gain = counts.get("Gain", 0)
        loss = counts.get("Loss", 0)
        signed_value = f"{(gain + loss) / total:.1%}" if total else "0.0%"
        signed_detail = "NDVI gain or loss pixels"
    cards = (
        (
            "GeoWatch quality",
            f"{quality_report.rounded_score}/{quality_report.max_score}",
            f"{quality_report.overall_status} overall run quality",
        ),
        ("AOI area", _format_area(metrics["area_km2"]), "Geodesic polygon area"),
        (
            "Valid coverage",
            f"{float(metrics['coverage_t1']):.1%} / {float(metrics['coverage_t2']):.1%}",
            "Start / end AOI coverage",
        ),
        (
            "Cloud and nodata",
            f"{float(metrics['nodata_t1']):.1%} / {float(metrics['nodata_t2']):.1%}",
            "Start / end invalid AOI pixels",
        ),
        ("Primary change", change_value, change_detail),
        ("NDVI movement", signed_value, signed_detail),
        (
            "Classification",
            quality_report.classification_confidence,
            (
                "Independent validation metrics"
                if analytics.accuracy
                else "No unsupported accuracy claims"
            ),
        ),
    )
    return "".join(
        f'<article class="metric-card"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong><small>{html.escape(detail)}</small></article>'
        for label, value, detail in cards
    )


def _quality_notice_text(report: QualityScoreReport) -> str:
    """Render the overview quality sentence for the dashboard."""
    parts = [
        f"GeoWatch Quality Score {report.rounded_score}/{report.max_score}",
        f"overall {report.overall_status.lower()}",
    ]
    for key in (
        "boundary",
        "cloud_nodata",
        "sensor",
        "season",
        "classification",
        "processing",
    ):
        component = report.component(key)
        if component is None:
            continue
        parts.append(f"{component.title.lower()} {component.status.lower()}")
    if report.warnings:
        parts.append(f"{len(report.warnings)} warning(s) recorded")
    return ". ".join(parts) + "."


def _render_slider(
    artifact: MapArtifact | None,
    output_path: Path,
    spec: RunSpecification,
) -> str:
    """Render the before/after swipe viewer or a graceful missing state."""
    if artifact is None:
        return _empty_state(
            "Before/after map artifacts were not generated for this run."
        )
    before = artifact.files.get("slider_before")
    after = artifact.files.get("slider_after")
    if before is None or after is None or not before.exists() or not after.exists():
        return _empty_state(
            "Matched endpoint images are unavailable. The static comparison map remains in the gallery."
        )
    before_href = _relative_href(before, output_path.parent)
    after_href = _relative_href(after, output_path.parent)
    return f"""<div class="comparison-shell" data-compare>
  <div class="comparison-stage">
    <img src="{after_href}" alt="After imagery for {html.escape(spec.location.name)} in {spec.temporal.end_year}" draggable="false">
    <div class="comparison-before" data-before><img src="{before_href}" alt="Before imagery for {html.escape(spec.location.name)} in {spec.temporal.start_year}" draggable="false"></div>
    <div class="comparison-handle" data-handle aria-hidden="true"><span>↔</span></div>
    <span class="comparison-label label-before">Before · {spec.temporal.start_year}</span>
    <span class="comparison-label label-after">After · {spec.temporal.end_year}</span>
    <input type="range" min="0" max="100" value="50" aria-label="Reveal before or after imagery" data-slider>
  </div>
  <p class="comparison-caption">Natural-color surface-reflectance composites. Light-gray pixels have no valid observation after QA masking.</p>
</div>"""


def _render_map_gallery(maps: Mapping[str, MapArtifact], output_path: Path) -> str:
    """Render available publication maps in a responsive gallery."""
    order = (
        "before_after",
        "change_detection",
        "ndvi_gain_loss",
        "ndvi",
        "ndwi",
        "ndbi",
        "lulc",
        "hotspot_analysis",
    )
    cards: list[str] = []
    for name in order:
        artifact = maps.get(name)
        if artifact is None:
            continue
        image_path = artifact.files.get("png_300")
        if image_path is None or not image_path.exists():
            continue
        href = _relative_href(image_path, output_path.parent)
        cards.append(f"""<figure class="map-card">
  <a href="{href}" target="_blank" rel="noopener" title="Open full-resolution {html.escape(artifact.title)}">
    <img src="{href}" alt="{html.escape(artifact.title)}" loading="lazy">
  </a>
  <figcaption><strong>{html.escape(artifact.title)}</strong><span>{html.escape(artifact.description)}</span></figcaption>
</figure>""")
    if not cards:
        return _empty_state("No publication map images are available for this run.")
    return '<div class="map-grid">' + "".join(cards) + "</div>"


def _render_index_table(analytics: AnalyticsReport) -> str:
    """Render T1, T2, and difference statistics for available indices."""
    if not analytics.index_results:
        return _empty_state("No spectral index statistics are available.")
    rows = []
    for name, result in analytics.index_results.items():
        rows.append(
            f'<tr><th scope="row">{html.escape(name.upper())}</th><td>{result.statistics.t1.mean:.4f}</td><td>{result.statistics.t2.mean:.4f}</td><td>{result.statistics.difference.mean:+.4f}</td></tr>'
        )
    return _table(
        ("Index", "Start mean", "End mean", "Difference"),
        "".join(rows),
    )


def _render_signed_change(analytics: AnalyticsReport, scene: RasterLayer) -> str:
    """Render signed NDVI area bars and a compact table."""
    signed = analytics.signed_change
    if signed is None or not signed.counts:
        return _empty_state("No NDVI gain/loss statistics are available.")
    pixel_area = _pixel_area_m2(scene)
    total = max(sum(signed.counts.values()), 1)
    bars: list[str] = []
    rows: list[str] = []
    for name in signed.class_names:
        count = signed.counts.get(name, 0)
        area_km2 = count * pixel_area / 1_000_000.0
        width = max((count / total) * 100.0, 0.5 if count else 0.0)
        color = _CHANGE_COLORS.get(name, "#607d8b")
        bars.append(
            f'<div class="bar-row"><span>{html.escape(name)}</span><div><i style="width:{width:.2f}%;background:{color}"></i></div><strong>{area_km2:,.2f} km²</strong></div>'
        )
        rows.append(
            f'<tr><th scope="row">{html.escape(name)}</th><td>{count:,}</td><td>{area_km2:,.2f}</td></tr>'
        )
    return (
        '<div class="bar-chart" aria-label="NDVI change area chart">'
        + "".join(bars)
        + "</div>"
        + _table(("Class", "Pixels", "Area km²"), "".join(rows))
        + f'<p class="data-note">Threshold: ±{signed.threshold:.4f} NDVI</p>'
    )


def _render_lulc(analytics: AnalyticsReport, scene: RasterLayer) -> str:
    """Render endpoint LULC class area bars and table when classification exists."""
    classification = analytics.classification_results.get("lulc_t2")
    if classification is None and analytics.classification_results:
        classification = next(reversed(analytics.classification_results.values()))
    if classification is None:
        return _empty_state("No LULC classification statistics are available.")
    pixel_area = _pixel_area_m2(scene)
    exploratory = classification.method in {"kmeans", "isodata"}
    labels = _classification_labels(classification, exploratory=exploratory)
    display_counts = [
        (label, classification.counts.get(source_name, 0))
        for source_name, label in labels
        if classification.counts.get(source_name, 0) > 0
    ]
    total = max(sum(count for _, count in display_counts), 1)
    bars: list[str] = []
    rows: list[str] = []
    for label, count in display_counts:
        area_km2 = count * pixel_area / 1_000_000.0
        width = max((count / total) * 100.0, 0.5)
        color = _LULC_COLORS.get(label, "#607d8b")
        bars.append(
            f'<div class="bar-row"><span>{html.escape(label)}</span><div><i style="width:{width:.2f}%;background:{color}"></i></div><strong>{area_km2:,.2f} km²</strong></div>'
        )
        rows.append(
            f'<tr><th scope="row">{html.escape(label)}</th><td>{count:,}</td><td>{area_km2:,.2f}</td></tr>'
        )
    if not rows:
        return _empty_state("The LULC output contains no classified AOI pixels.")
    note = (
        "Exploratory unsupervised classification; class labels require reference validation."
        if exploratory
        else "Supervised classification; consult the accuracy section for validation status."
    )
    return (
        '<div class="bar-chart" aria-label="LULC area chart">'
        + "".join(bars)
        + "</div>"
        + _table(("Class", "Pixels", "Area km²"), "".join(rows))
        + f'<p class="data-note">{html.escape(note)}</p>'
    )


def _classification_labels(
    classification: ClassificationResult, *, exploratory: bool
) -> list[tuple[str, str]]:
    """Return source and publication-safe LULC class names."""
    return [
        (
            name,
            (
                "Bright Surface / Uncertain"
                if exploratory and name == "Snow/Ice"
                else name
            ),
        )
        for name in classification.class_names
    ]


def _render_transition_matrix(analytics: AnalyticsReport) -> str:
    """Render the LULC transition matrix with horizontal overflow support."""
    transition = analytics.transition_result
    if transition.transition_matrix.size == 0:
        return _empty_state("No LULC transition matrix is available.")
    headers = ("From / to", *transition.class_names)
    rows: list[str] = []
    for index, name in enumerate(transition.class_names):
        cells = "".join(
            f"<td>{int(value):,}</td>" for value in transition.transition_matrix[index]
        )
        rows.append(f'<tr><th scope="row">{html.escape(name)}</th>{cells}</tr>')
    return (
        '<div class="matrix-scroll">'
        + _table(headers, "".join(rows))
        + '</div><p class="data-note">Cells report pixel transitions. Convert to area using the exported statistics when needed.</p>'
    )


def _render_scene_table(sources: Sequence[SceneMetadata]) -> str:
    """Render normalized scene acquisition metadata."""
    if not sources:
        return _empty_state("No scene catalog records are available.")
    rows = []
    for source in sources:
        acquired = (
            source.acquired_at.date().isoformat() if source.acquired_at else "Unknown"
        )
        cloud = (
            f"{source.cloud_cover:.1f}%"
            if source.cloud_cover is not None
            else "Unknown"
        )
        rows.append(
            f'<tr><th scope="row">{html.escape(source.scene_id)}</th><td>{html.escape(acquired)}</td><td>{cloud}</td><td>{html.escape(_format_dataset(source.dataset))}</td><td>{html.escape(source.provider)}</td></tr>'
        )
    return _table(("Scene ID", "Date", "Cloud", "Mission", "Provider"), "".join(rows))


def _render_provenance(
    spec: RunSpecification,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
    boundary_path: Path,
    generated: datetime,
) -> str:
    """Render data-source, processing, and AOI provenance fields."""
    providers = (
        ", ".join(sorted({source.provider for source in sources})) or "Unavailable"
    )
    missions = ", ".join(
        sorted({_format_dataset(source.dataset) for source in sources})
    ) or _mission_text(sources, scene_t1, scene_t2)
    dates = sorted(
        source.acquired_at.date().isoformat()
        for source in sources
        if source.acquired_at is not None
    )
    fallback = (
        "; ".join(availability.fallback_messages)
        if availability and availability.fallback_messages
        else "No acquisition fallback was required."
    )
    fields = (
        ("Provider", providers),
        ("Mission", missions),
        ("Acquisition dates", ", ".join(dates) or "Unavailable"),
        ("Bands", ", ".join(scene_t1.grid.band_names) or "See raster metadata"),
        ("Boundary source", spec.location.boundary_source or "User-provided boundary"),
        ("Boundary file", boundary_path.name),
        ("CRS", scene_t1.grid.crs),
        ("Composite", spec.imagery.composite_method),
        ("Season", f"Months {spec.temporal.start_month}-{spec.temporal.end_month}"),
        ("Processing date", generated.date().isoformat()),
        ("Search fallback", fallback),
    )
    return (
        '<dl class="provenance-grid">'
        + "".join(
            f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd></div>"
            for label, value in fields
        )
        + "</dl>"
    )


def _render_limitations(
    spec: RunSpecification, sources: Sequence[SceneMetadata]
) -> str:
    """Render scientifically cautious, sensor-aware limitations."""
    datasets: set[str] = {source.dataset for source in sources}
    items = [
        (
            "Cloud masking",
            "Thin cloud, haze, and cloud-edge contamination can remain after automated QA masking.",
        ),
        (
            "Seasonality",
            "Phenology, rainfall, tides, and acquisition timing can create apparent change even when land cover is stable.",
        ),
        (
            "Mixed pixels",
            "Each pixel may contain multiple materials, especially near coastlines, rivers, roads, and urban edges.",
        ),
        (
            "Classification",
            "Unsupervised LULC is exploratory. No accuracy claim is made without independent validation samples.",
        ),
        (
            "Boundary source",
            f"Administrative limits inherit the completeness, date, and legal interpretation of {spec.location.boundary_source or 'the supplied boundary source'}.",
        ),
    ]
    if len(datasets) > 1:
        items.append(
            (
                "Sensor differences",
                "Multiple missions may differ in spectral response, resolution, and acquisition geometry.",
            )
        )
    if "landsat-7-c2-l2" in datasets:
        items.append(
            (
                "Landsat 7 SLC-off",
                "Post-2003 scenes contain scan-line gaps; multi-scene compositing reduces but may not remove all missing coverage.",
            )
        )
    return (
        '<ul class="limitation-grid">'
        + "".join(
            f"<li><strong>{html.escape(title)}</strong><span>{html.escape(text)}</span></li>"
            for title, text in items
        )
        + "</ul>"
    )


def _render_downloads(
    output_path: Path,
    maps: Mapping[str, MapArtifact],
    analytics: AnalyticsReport,
    downloads: Mapping[str, Path],
) -> str:
    """Render existing project artifacts as portable relative download links."""
    groups: dict[str, list[tuple[str, Path]]] = {
        "Maps": [],
        "GIS and statistics": [],
        "Reports": [],
    }
    seen: set[str] = set()
    for artifact in maps.values():
        for format_name, path in artifact.files.items():
            if path.exists() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".pdf",
                ".svg",
            }:
                key = str(path.resolve())
                if key not in seen:
                    groups["Maps"].append((f"{artifact.title} · {format_name}", path))
                    seen.add(key)
    for label, path in {**analytics.artifacts, **downloads}.items():
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        group = (
            "Reports"
            if path.suffix.lower() in {".html", ".pdf", ".md"}
            else "GIS and statistics"
        )
        groups[group].append((_humanize(label), path))
        seen.add(key)
    sections: list[str] = []
    for group, items in groups.items():
        if not items:
            continue
        links = "".join(
            f'<a class="download-link" href="{_relative_href(path, output_path.parent)}" download><span>{html.escape(label)}</span><small>{html.escape(path.suffix.lstrip(".").upper() or "FILE")}</small></a>'
            for label, path in sorted(items, key=lambda item: item[0])
        )
        sections.append(
            f'<details class="download-group" open><summary>{html.escape(group)} <small>{len(items)} files</small></summary><div>{links}</div></details>'
        )
    if not sections:
        return _empty_state("No downloadable artifacts are currently available.")
    return '<div class="download-groups">' + "".join(sections) + "</div>"


def _primary_change(analytics: AnalyticsReport) -> ChangeDetectionResult | None:
    """Select a stable primary change result for dashboard summaries."""
    for name in ("mad", "irmad", "index_difference", "cva"):
        if name in analytics.change_results:
            return analytics.change_results[name]
    return next(iter(analytics.change_results.values()), None)


def _mission_text(
    sources: Sequence[SceneMetadata],
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> str:
    """Return a concise mission label from catalogs or raster provenance."""
    datasets: set[str] = {source.dataset for source in sources}
    if not datasets:
        datasets = {
            str(scene.metadata.get("dataset", ""))
            for scene in (scene_t1, scene_t2)
            if scene.metadata.get("dataset")
        }
    return (
        ", ".join(sorted(_format_dataset(value) for value in datasets))
        or "Mission unavailable"
    )


def _format_dataset(dataset: str) -> str:
    """Convert normalized dataset identifiers into publication labels."""
    labels = {
        "sentinel-2-l2a": "Sentinel-2 L2A",
        "landsat-5-c2-l2": "Landsat 5 Collection 2 L2",
        "landsat-7-c2-l2": "Landsat 7 Collection 2 L2",
        "landsat-8-c2-l2": "Landsat 8 Collection 2 L2",
        "landsat-9-c2-l2": "Landsat 9 Collection 2 L2",
    }
    return labels.get(dataset, dataset.replace("-", " ").title())


def _pixel_area_m2(scene: RasterLayer) -> float:
    """Calculate one pixel area in square metres from the scene grid and CRS."""
    a, b, _, d, e, _ = scene.grid.transform
    native_area = abs((a * e) - (b * d))
    try:
        crs = CRS.from_user_input(scene.grid.crs)
        factor = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 1.0
    except (ValueError, IndexError):
        logger.debug("Assuming metre grid units for dashboard area calculation")
        factor = 1.0
    return float(native_area * factor * factor)


def _relative_href(path: Path, base_directory: Path) -> str:
    """Return a URL-safe relative path for a portable local dashboard."""
    relative = Path(os.path.relpath(path, start=base_directory)).as_posix()
    return html.escape(quote(relative, safe="/._-"), quote=True)


def _format_area(value: float) -> str:
    """Format an AOI area with a sensible precision."""
    area = float(value)
    if not math.isfinite(area):
        return "Unavailable"
    return f"{area:,.1f} km²" if area >= 100.0 else f"{area:,.2f} km²"


def _humanize(value: str) -> str:
    """Convert an artifact identifier to a compact human label."""
    return value.replace("_", " ").replace("-", " ").strip().title()


def _table(headers: Sequence[str], rows: str) -> str:
    """Wrap pre-rendered rows in an accessible table."""
    header_cells = "".join(
        f'<th scope="col">{html.escape(header)}</th>' for header in headers
    )
    return f'<div class="table-scroll"><table><thead><tr>{header_cells}</tr></thead><tbody>{rows}</tbody></table></div>'


def _empty_state(message: str) -> str:
    """Render a consistent optional-output empty state."""
    return f'<div class="empty-state"><strong>Not available</strong><span>{html.escape(message)}</span></div>'


_DASHBOARD_CSS = r"""
:root{--ink:#102a43;--muted:#526579;--line:#ccd7df;--paper:#fff;--page:#edf1f3;--green:#007f5f;--blue:#277da1;--amber:#e9c46a;--orange:#e76f51;--shadow:0 8px 24px rgba(16,42,67,.08)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--page);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif;line-height:1.55;letter-spacing:0}a{color:inherit}button,input{font:inherit}.topbar{background:#102a43;color:#fff;padding:28px max(28px,calc((100vw - 1400px)/2));display:flex;align-items:flex-end;justify-content:space-between;gap:28px;border-bottom:5px solid var(--green)}.brand-block{max-width:760px}.eyebrow,.kicker{display:block;color:#75d5b8;font-size:.74rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.brand-block h1{font-size:clamp(1.75rem,3vw,2.8rem);line-height:1.08;margin:7px 0 9px}.brand-block p{margin:0;color:#d8e4ec}.run-facts{display:grid;grid-template-columns:repeat(2,minmax(115px,1fr));gap:14px 24px;margin:0}.run-facts div{border-left:2px solid #3aa886;padding-left:10px}.run-facts dt{font-size:.7rem;text-transform:uppercase;color:#b9c9d5;font-weight:700}.run-facts dd{margin:2px 0 0;font-weight:750}.section-nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);display:flex;gap:4px;padding:8px max(22px,calc((100vw - 1400px)/2));overflow:auto;white-space:nowrap}.section-nav a{text-decoration:none;padding:8px 12px;font-size:.84rem;font-weight:700;color:#334e68;border-radius:4px}.section-nav a:hover,.section-nav a:focus{background:#e4eeea;color:#00684e;outline:none}main{max-width:1400px;margin:auto;background:var(--paper);box-shadow:var(--shadow)}.section-block{padding:38px 42px;border-bottom:1px solid var(--line)}.section-heading{max-width:780px;margin-bottom:22px}.section-heading h2{font-size:1.62rem;line-height:1.2;margin:5px 0 7px}.section-heading p{color:var(--muted);margin:0}.summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.metric-card{border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:6px;padding:17px;background:#fbfcfd;min-height:126px;display:flex;flex-direction:column}.metric-card:nth-child(2n){border-top-color:var(--green)}.metric-card:nth-child(3n){border-top-color:var(--amber)}.metric-card span{font-size:.78rem;font-weight:800;color:var(--muted);text-transform:uppercase}.metric-card strong{font-size:1.48rem;margin:8px 0 3px;line-height:1.15}.metric-card small{color:var(--muted);margin-top:auto}.quality-notice{display:flex;gap:14px;align-items:center;margin-top:14px;padding:14px 16px;border-left:4px solid;border-radius:4px}.quality-good{background:#e9f6f1;border-color:var(--green)}.quality-warn{background:#fff4e5;border-color:#d97706}.quality-notice span{color:#334e68}.comparison-shell{max-width:1120px;margin:auto}.comparison-stage{position:relative;overflow:hidden;background:#dfe5e9;aspect-ratio:16/9;border:1px solid #8fa4b3;box-shadow:var(--shadow);touch-action:none}.comparison-stage>img,.comparison-before img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;user-select:none}.comparison-before{position:absolute;inset:0 auto 0 0;width:50%;overflow:hidden}.comparison-before img{width:100vw;max-width:1120px}.comparison-handle{position:absolute;top:0;bottom:0;left:50%;width:3px;background:#fff;box-shadow:0 0 0 1px rgba(16,42,67,.35);pointer-events:none}.comparison-handle span{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);display:grid;place-items:center;width:42px;height:42px;border-radius:50%;background:#fff;color:var(--ink);font-size:1.25rem;font-weight:900;box-shadow:0 3px 12px rgba(0,0,0,.3)}.comparison-stage input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize}.comparison-label{position:absolute;top:14px;padding:7px 10px;background:rgba(16,42,67,.86);color:#fff;border-radius:4px;font-size:.78rem;font-weight:800;pointer-events:none}.label-before{left:14px}.label-after{right:14px}.comparison-caption,.data-note{font-size:.82rem;color:var(--muted);margin:9px 0 0}.map-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.map-card{margin:0;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#fff}.map-card a{display:block;background:#e9eef1;aspect-ratio:16/10;overflow:hidden}.map-card img{width:100%;height:100%;object-fit:contain;display:block;transition:transform .2s ease}.map-card a:hover img{transform:scale(1.015)}.map-card figcaption{display:flex;flex-direction:column;padding:13px 15px}.map-card figcaption span{color:var(--muted);font-size:.83rem;margin-top:3px}.analysis-grid{display:grid;grid-template-columns:1.15fr 1fr 1fr;gap:18px;align-items:start}.data-panel,.table-panel,.matrix-panel{min-width:0}.data-panel h3,.table-panel h3,.matrix-panel h3,.download-group h3{font-size:1rem;margin:0 0 12px}.matrix-panel{margin-top:28px}.table-scroll,.matrix-scroll{overflow:auto;max-width:100%;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;background:#fff;font-size:.8rem}th,td{text-align:right;padding:8px 10px;border-bottom:1px solid #dce3e8;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#e8eef2;color:#243b53;font-size:.72rem;text-transform:uppercase}tbody tr:nth-child(even){background:#f7f9fa}.bar-chart{display:grid;gap:9px;margin-bottom:14px}.bar-row{display:grid;grid-template-columns:minmax(90px,1fr) 2fr auto;gap:8px;align-items:center;font-size:.76rem}.bar-row>div{height:9px;background:#e1e6ea;overflow:hidden;border-radius:2px}.bar-row i{display:block;height:100%}.bar-row strong{font-size:.72rem}.workflow-list{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;counter-reset:flow}.workflow-list li{counter-increment:flow;border-top:3px solid var(--green);padding:14px;background:#f7f9fa}.workflow-list li:before{content:counter(flow,decimal-leading-zero);display:block;color:var(--green);font-weight:900;margin-bottom:7px}.workflow-list strong,.workflow-list span{display:block}.workflow-list span{font-size:.82rem;color:var(--muted);margin-top:4px}.caveat-inline{border-left:4px solid var(--amber);background:#fff8df;padding:13px 15px;margin:15px 0 0}.provenance-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}.provenance-grid div{background:#fff;padding:12px}.provenance-grid dt{font-size:.7rem;text-transform:uppercase;color:var(--muted);font-weight:800}.provenance-grid dd{margin:4px 0 0;font-size:.86rem;overflow-wrap:anywhere}.table-panel{margin-top:24px}.limitation-grid{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.limitation-grid li{padding:15px;border-left:4px solid var(--orange);background:#f7f9fa}.limitation-grid strong,.limitation-grid span{display:block}.limitation-grid span{font-size:.84rem;color:var(--muted);margin-top:4px}.download-groups{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.download-group>div{display:grid;gap:7px}.download-link{display:flex;justify-content:space-between;gap:12px;text-decoration:none;padding:10px 11px;border:1px solid var(--line);border-radius:4px;background:#fff;font-size:.82rem}.download-link:hover,.download-link:focus{border-color:var(--green);background:#f0f8f5;outline:none}.download-link small{font-weight:800;color:var(--green)}.empty-state{border:1px dashed #9babb7;background:#f7f9fa;padding:22px;display:flex;flex-direction:column;align-items:flex-start}.empty-state span{color:var(--muted);font-size:.84rem;margin-top:3px}footer{max-width:1400px;margin:0 auto 40px;padding:20px 42px;background:#102a43;color:#fff;display:flex;justify-content:space-between;gap:20px;font-size:.82rem}footer span{color:#c7d5df}
@media(max-width:1050px){.analysis-grid{grid-template-columns:1fr 1fr}.analysis-grid .data-panel:first-child{grid-column:1/-1}.download-groups{grid-template-columns:1fr 1fr}.workflow-list{grid-template-columns:1fr 1fr}}
@media(max-width:760px){.topbar{align-items:flex-start;flex-direction:column;padding:24px 20px}.run-facts{width:100%}.section-block{padding:28px 20px}.summary-grid,.map-grid,.analysis-grid,.provenance-grid,.limitation-grid,.download-groups,.workflow-list{grid-template-columns:1fr}.analysis-grid .data-panel:first-child{grid-column:auto}.comparison-stage{aspect-ratio:4/3}.comparison-before img{max-width:none;width:calc(100vw - 40px)}footer{margin-bottom:0;padding:18px 20px;flex-direction:column}.bar-row{grid-template-columns:90px 1fr}.bar-row strong{grid-column:2}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.map-card img{transition:none}}
.comparison-before img{width:auto!important;max-width:none!important}
.download-group summary{cursor:pointer;font-size:1rem;font-weight:800;margin-bottom:12px}.download-group summary small{color:var(--muted);font-weight:600;margin-left:5px}.download-group>div{max-height:520px;overflow:auto;padding-right:4px}
.interpretation-grid .interpretation-block{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.interpretation-card{border:1px solid var(--line);border-left:4px solid var(--green);border-radius:6px;background:#fbfcfd;padding:16px}.interpretation-card h3{margin:0 0 8px;font-size:1rem;color:#102a43}.interpretation-card p{margin:0;color:#334e68;font-size:.9rem}.interpretation-card p+p{margin-top:9px}@media(max-width:1050px){.interpretation-grid .interpretation-block{grid-template-columns:1fr}}
"""

_DASHBOARD_JS = r"""
document.querySelectorAll('[data-compare]').forEach((root)=>{const stage=root.querySelector('.comparison-stage');const input=root.querySelector('[data-slider]');const before=root.querySelector('[data-before]');const beforeImage=before.querySelector('img');const handle=root.querySelector('[data-handle]');const update=()=>{const value=input.value+'%';before.style.width=value;handle.style.left=value;input.setAttribute('aria-valuenow',input.value)};const resize=()=>{beforeImage.style.width=stage.clientWidth+'px'};input.addEventListener('input',update);window.addEventListener('resize',resize);if('ResizeObserver' in window){new ResizeObserver(resize).observe(stage)}resize();update()});
"""
