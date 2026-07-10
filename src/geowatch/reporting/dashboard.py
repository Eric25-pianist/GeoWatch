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
      <div class="brand-row">
        <span class="brand-mark">GW</span>
        <span class="eyebrow">GeoWatch Project</span>
      </div>
      <h1>{html.escape(spec.location.name)} change dashboard</h1>
      <p class="hero-copy">{spec.temporal.start_year} to {spec.temporal.end_year} &middot; {html.escape(mission)}</p>
      <div class="hero-actions">
        <a href="#compare">Review comparison</a>
        <a href="#downloads">Open exports</a>
      </div>
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

  <footer><strong>GeoWatch Project</strong><span>Terminal-first geospatial change analysis &middot; {generated.date().isoformat()}</span></footer>
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
    <div class="comparison-handle" data-handle aria-hidden="true"><span>&#8596;</span></div>
    <span class="comparison-label label-before">Before &middot; {spec.temporal.start_year}</span>
    <span class="comparison-label label-after">After &middot; {spec.temporal.end_year}</span>
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
                    groups["Maps"].append((f"{artifact.title} - {format_name}", path))
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
:root {
  --ink: #111827;
  --muted: #607086;
  --line: #d7e0e8;
  --paper: #ffffff;
  --panel: #f8fafc;
  --page: #edf2f7;
  --navy: #0b1220;
  --navy-2: #12243a;
  --green: #13856b;
  --blue: #2563eb;
  --cyan: #0ea5b7;
  --amber: #d99b14;
  --orange: #df6b3d;
  --red: #c24135;
  --shadow: 0 16px 42px rgba(17, 24, 39, .11);
  --soft-shadow: 0 8px 22px rgba(17, 24, 39, .08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  min-width: 320px;
  background: linear-gradient(180deg, var(--navy) 0, var(--navy-2) 330px, var(--page) 330px);
  color: var(--ink);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  letter-spacing: 0;
}
a { color: inherit; }
button, input { font: inherit; }
.topbar {
  color: #fff;
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(360px, .65fr);
  gap: 28px;
  max-width: 1440px;
  margin: 0 auto;
  padding: 42px 42px 36px;
}
.brand-block { max-width: 850px; }
.brand-row {
  align-items: center;
  display: flex;
  gap: 12px;
  margin-bottom: 18px;
}
.brand-mark {
  align-items: center;
  background: linear-gradient(135deg, #39d0ff, #34d399);
  color: #04111f;
  display: inline-flex;
  font-weight: 900;
  height: 40px;
  justify-content: center;
  width: 40px;
}
.eyebrow, .kicker {
  color: #8ee5ca;
  display: block;
  font-size: .74rem;
  font-weight: 850;
  letter-spacing: 0;
  text-transform: uppercase;
}
.brand-block h1 {
  font-size: 2.65rem;
  line-height: 1.05;
  margin: 0 0 12px;
}
.hero-copy {
  color: #d9e8f4;
  font-size: 1.05rem;
  margin: 0;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 22px;
}
.hero-actions a {
  background: rgba(255, 255, 255, .11);
  border: 1px solid rgba(255, 255, 255, .28);
  color: #fff;
  font-size: .88rem;
  font-weight: 800;
  padding: 9px 13px;
  text-decoration: none;
}
.hero-actions a:hover, .hero-actions a:focus {
  background: rgba(255, 255, 255, .19);
  outline: 2px solid rgba(142, 229, 202, .55);
  outline-offset: 2px;
}
.run-facts {
  align-self: end;
  background: rgba(255, 255, 255, .09);
  border: 1px solid rgba(255, 255, 255, .22);
  display: grid;
  gap: 1px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 0;
}
.run-facts div {
  background: rgba(11, 18, 32, .42);
  min-height: 92px;
  padding: 16px;
}
.run-facts dt {
  color: #b9c9d8;
  font-size: .72rem;
  font-weight: 800;
  text-transform: uppercase;
}
.run-facts dd {
  color: #fff;
  font-weight: 850;
  margin: 5px 0 0;
  overflow-wrap: anywhere;
}
.section-nav {
  background: rgba(255, 255, 255, .97);
  border-bottom: 1px solid var(--line);
  box-shadow: 0 5px 18px rgba(17, 24, 39, .06);
  display: flex;
  gap: 6px;
  overflow: auto;
  padding: 10px max(22px, calc((100vw - 1440px) / 2 + 42px));
  position: sticky;
  top: 0;
  white-space: nowrap;
  z-index: 20;
}
.section-nav a {
  color: #334155;
  font-size: .84rem;
  font-weight: 780;
  padding: 8px 12px;
  text-decoration: none;
}
.section-nav a:hover, .section-nav a:focus {
  background: #e8f4ef;
  color: #0f766e;
  outline: none;
}
main {
  background: var(--paper);
  box-shadow: var(--shadow);
  margin: 0 auto;
  max-width: 1440px;
}
.section-block {
  border-bottom: 1px solid var(--line);
  padding: 42px;
}
.section-heading {
  margin-bottom: 22px;
  max-width: 850px;
}
.section-heading h2 {
  font-size: 1.65rem;
  line-height: 1.18;
  margin: 5px 0 7px;
}
.section-heading p {
  color: var(--muted);
  margin: 0;
}
.summary-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.metric-card {
  background: linear-gradient(180deg, #fff, #f8fafc);
  border: 1px solid var(--line);
  border-top: 4px solid var(--blue);
  box-shadow: var(--soft-shadow);
  display: flex;
  flex-direction: column;
  min-height: 132px;
  padding: 17px;
}
.metric-card:nth-child(2n) { border-top-color: var(--green); }
.metric-card:nth-child(3n) { border-top-color: var(--amber); }
.metric-card:nth-child(4n) { border-top-color: var(--cyan); }
.metric-card span {
  color: var(--muted);
  font-size: .76rem;
  font-weight: 850;
  text-transform: uppercase;
}
.metric-card strong {
  color: var(--ink);
  font-size: 1.45rem;
  line-height: 1.15;
  margin: 9px 0 5px;
  overflow-wrap: anywhere;
}
.metric-card small {
  color: var(--muted);
  margin-top: auto;
}
.quality-notice {
  align-items: flex-start;
  border: 1px solid var(--line);
  border-left: 5px solid;
  display: flex;
  gap: 14px;
  margin-top: 16px;
  padding: 15px 17px;
}
.quality-good {
  background: #eefaf5;
  border-left-color: var(--green);
}
.quality-warn {
  background: #fff7e6;
  border-left-color: var(--amber);
}
.quality-notice strong { white-space: nowrap; }
.quality-notice span { color: #334155; }
.comparison-shell {
  margin: auto;
  max-width: 1160px;
}
.comparison-stage {
  background: #dfe7ed;
  border: 1px solid #8297aa;
  box-shadow: var(--shadow);
  overflow: hidden;
  position: relative;
  touch-action: none;
  aspect-ratio: 16 / 9;
}
.comparison-stage > img,
.comparison-before img {
  height: 100%;
  inset: 0;
  object-fit: cover;
  position: absolute;
  user-select: none;
  width: 100%;
}
.comparison-before {
  inset: 0 auto 0 0;
  overflow: hidden;
  position: absolute;
  width: 50%;
}
.comparison-before img {
  max-width: none;
  width: 100%;
}
.comparison-handle {
  background: #fff;
  bottom: 0;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, .4);
  left: 50%;
  pointer-events: none;
  position: absolute;
  top: 0;
  width: 3px;
}
.comparison-handle span {
  align-items: center;
  background: #fff;
  border: 1px solid #cbd5e1;
  border-radius: 50%;
  box-shadow: 0 5px 18px rgba(0, 0, 0, .28);
  color: var(--ink);
  display: flex;
  font-size: 1.2rem;
  font-weight: 900;
  height: 42px;
  justify-content: center;
  left: 50%;
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 42px;
}
.comparison-stage input {
  cursor: ew-resize;
  height: 100%;
  inset: 0;
  opacity: 0;
  position: absolute;
  width: 100%;
}
.comparison-label {
  background: rgba(15, 23, 42, .86);
  color: #fff;
  font-size: .78rem;
  font-weight: 850;
  padding: 7px 10px;
  pointer-events: none;
  position: absolute;
  top: 14px;
}
.label-before { left: 14px; }
.label-after { right: 14px; }
.comparison-caption,
.data-note {
  color: var(--muted);
  font-size: .84rem;
  margin: 10px 0 0;
}
.map-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.map-card {
  background: #fff;
  border: 1px solid var(--line);
  box-shadow: var(--soft-shadow);
  margin: 0;
  overflow: hidden;
}
.map-card a {
  background: #edf2f7;
  display: block;
  overflow: hidden;
  aspect-ratio: 16 / 10;
}
.map-card img {
  display: block;
  height: 100%;
  object-fit: contain;
  transition: transform .18s ease;
  width: 100%;
}
.map-card a:hover img { transform: scale(1.012); }
.map-card.image-missing a {
  align-items: center;
  color: var(--muted);
  display: flex;
  justify-content: center;
}
.map-card.image-missing a::after {
  content: "Image unavailable";
  font-weight: 800;
}
.map-card.image-missing img { display: none; }
.map-card figcaption {
  display: flex;
  flex-direction: column;
  padding: 14px 16px;
}
.map-card figcaption span {
  color: var(--muted);
  font-size: .84rem;
  margin-top: 4px;
}
.analysis-grid {
  align-items: start;
  display: grid;
  gap: 18px;
  grid-template-columns: 1.2fr 1fr 1fr;
}
.data-panel,
.table-panel,
.matrix-panel {
  background: #fff;
  border: 1px solid var(--line);
  box-shadow: var(--soft-shadow);
  min-width: 0;
  padding: 18px;
}
.data-panel h3,
.table-panel h3,
.matrix-panel h3,
.download-group h3 {
  font-size: 1rem;
  margin: 0 0 12px;
}
.matrix-panel { margin-top: 22px; }
.table-scroll,
.matrix-scroll {
  border: 1px solid var(--line);
  max-width: 100%;
  overflow: auto;
}
table {
  background: #fff;
  border-collapse: collapse;
  font-size: .82rem;
  width: 100%;
}
th, td {
  border-bottom: 1px solid #e2e8f0;
  padding: 8px 10px;
  text-align: right;
  white-space: nowrap;
}
th:first-child, td:first-child { text-align: left; }
thead th {
  background: #e8eef5;
  color: #243244;
  font-size: .72rem;
  position: sticky;
  text-transform: uppercase;
  top: 0;
}
tbody tr:nth-child(even) { background: #f8fafc; }
.bar-chart {
  display: grid;
  gap: 9px;
  margin-bottom: 14px;
}
.bar-row {
  align-items: center;
  display: grid;
  font-size: .78rem;
  gap: 9px;
  grid-template-columns: minmax(96px, 1fr) 2fr auto;
}
.bar-row > div {
  background: #e2e8f0;
  height: 10px;
  overflow: hidden;
}
.bar-row i {
  display: block;
  height: 100%;
}
.bar-row strong { font-size: .74rem; }
.workflow-list {
  counter-reset: flow;
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(3, 1fr);
  list-style: none;
  margin: 0;
  padding: 0;
}
.workflow-list li {
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 4px solid var(--green);
  counter-increment: flow;
  padding: 15px;
}
.workflow-list li::before {
  color: var(--green);
  content: counter(flow, decimal-leading-zero);
  display: block;
  font-weight: 900;
  margin-bottom: 8px;
}
.workflow-list strong,
.workflow-list span {
  display: block;
}
.workflow-list span {
  color: var(--muted);
  font-size: .84rem;
  margin-top: 5px;
}
.caveat-inline {
  background: #fff8df;
  border-left: 5px solid var(--amber);
  margin: 16px 0 0;
  padding: 14px 16px;
}
.provenance-grid {
  background: var(--line);
  border: 1px solid var(--line);
  display: grid;
  gap: 1px;
  grid-template-columns: repeat(3, 1fr);
}
.provenance-grid div {
  background: #fff;
  padding: 13px;
}
.provenance-grid dt {
  color: var(--muted);
  font-size: .72rem;
  font-weight: 850;
  text-transform: uppercase;
}
.provenance-grid dd {
  font-size: .86rem;
  margin: 4px 0 0;
  overflow-wrap: anywhere;
}
.table-panel { margin-top: 24px; }
.limitation-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(2, 1fr);
  list-style: none;
  margin: 0;
  padding: 0;
}
.limitation-grid li {
  background: var(--panel);
  border-left: 5px solid var(--orange);
  padding: 15px;
}
.limitation-grid strong,
.limitation-grid span {
  display: block;
}
.limitation-grid span {
  color: var(--muted);
  font-size: .84rem;
  margin-top: 5px;
}
.download-groups {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(3, 1fr);
}
.download-group {
  background: #fff;
  border: 1px solid var(--line);
  box-shadow: var(--soft-shadow);
  padding: 16px;
}
.download-group summary {
  cursor: pointer;
  font-size: 1rem;
  font-weight: 850;
  margin-bottom: 12px;
}
.download-group summary small {
  color: var(--muted);
  font-weight: 650;
  margin-left: 5px;
}
.download-group > div {
  display: grid;
  gap: 7px;
  max-height: 520px;
  overflow: auto;
  padding-right: 4px;
}
.download-link {
  align-items: center;
  background: #fff;
  border: 1px solid var(--line);
  display: flex;
  font-size: .83rem;
  gap: 12px;
  justify-content: space-between;
  padding: 10px 11px;
  text-decoration: none;
}
.download-link:hover,
.download-link:focus {
  background: #f0f8f5;
  border-color: var(--green);
  outline: none;
}
.download-link small {
  color: var(--green);
  font-weight: 850;
}
.empty-state {
  align-items: flex-start;
  background: #f8fafc;
  border: 1px dashed #94a3b8;
  display: flex;
  flex-direction: column;
  padding: 22px;
}
.empty-state span {
  color: var(--muted);
  font-size: .84rem;
  margin-top: 4px;
}
.interpretation-grid .interpretation-block {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.interpretation-card {
  background: #fff;
  border: 1px solid var(--line);
  border-left: 5px solid var(--green);
  box-shadow: var(--soft-shadow);
  padding: 16px;
}
.interpretation-card h3 {
  color: var(--ink);
  font-size: 1rem;
  margin: 0 0 8px;
}
.interpretation-card p {
  color: #334155;
  font-size: .9rem;
  margin: 0;
}
.interpretation-card p + p { margin-top: 9px; }
footer {
  background: var(--navy);
  color: #fff;
  display: flex;
  font-size: .84rem;
  gap: 20px;
  justify-content: space-between;
  margin: 0 auto 40px;
  max-width: 1440px;
  padding: 20px 42px;
}
footer span { color: #c7d5df; }
@media (max-width: 1120px) {
  .topbar {
    grid-template-columns: 1fr;
    padding: 34px 28px;
  }
  .run-facts { max-width: 760px; }
  .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .analysis-grid { grid-template-columns: 1fr 1fr; }
  .analysis-grid .data-panel:first-child { grid-column: 1 / -1; }
  .download-groups,
  .workflow-list {
    grid-template-columns: 1fr 1fr;
  }
  .interpretation-grid .interpretation-block { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  body {
    background: linear-gradient(180deg, var(--navy) 0, var(--navy-2) 420px, var(--page) 420px);
  }
  .topbar { padding: 28px 20px; }
  .brand-block h1 { font-size: 2rem; }
  .run-facts,
  .summary-grid,
  .map-grid,
  .analysis-grid,
  .provenance-grid,
  .limitation-grid,
  .download-groups,
  .workflow-list {
    grid-template-columns: 1fr;
  }
  .section-block { padding: 30px 20px; }
  .section-nav { padding: 9px 20px; }
  .analysis-grid .data-panel:first-child { grid-column: auto; }
  .comparison-stage { aspect-ratio: 4 / 3; }
  .bar-row {
    grid-template-columns: 96px 1fr;
  }
  .bar-row strong { grid-column: 2; }
  footer {
    flex-direction: column;
    margin-bottom: 0;
    padding: 18px 20px;
  }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .map-card img { transition: none; }
}
"""

_DASHBOARD_JS = r"""
document.querySelectorAll('[data-compare]').forEach((root) => {
  const stage = root.querySelector('.comparison-stage');
  const input = root.querySelector('[data-slider]');
  const before = root.querySelector('[data-before]');
  const beforeImage = before ? before.querySelector('img') : null;
  const handle = root.querySelector('[data-handle]');
  if (!stage || !input || !before || !beforeImage || !handle) return;
  const update = () => {
    const value = input.value + '%';
    before.style.width = value;
    handle.style.left = value;
    input.setAttribute('aria-valuenow', input.value);
  };
  const resize = () => {
    beforeImage.style.width = stage.clientWidth + 'px';
    beforeImage.style.height = stage.clientHeight + 'px';
  };
  input.addEventListener('input', update);
  window.addEventListener('resize', resize);
  if ('ResizeObserver' in window) {
    new ResizeObserver(resize).observe(stage);
  }
  resize();
  update();
});
document.querySelectorAll('.map-card img').forEach((image) => {
  image.addEventListener('error', () => {
    const card = image.closest('.map-card');
    if (card) card.classList.add('image-missing');
    image.alt = 'Map image could not be loaded from the local output folder.';
  });
});
"""
