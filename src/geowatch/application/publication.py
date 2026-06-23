"""Real-data reports, exports, and validation summaries for terminal projects."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from loguru import logger
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import AnalyticsReport
from geowatch.application.availability import AvailabilityPlan
from geowatch.application.models import RunSpecification
from geowatch.application.project import ProjectLayout
from geowatch.portfolio.exporter import export_portfolio_package
from geowatch.processing.models import RasterLayer
from geowatch.reporting.dashboard import write_dashboard
from geowatch.reporting.exports import export_publication_tables
from geowatch.reporting.interpretation import (
    InterpretationReport,
    generate_interpretation,
    render_interpretation_html,
    write_interpretation,
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
    write_quality_outputs,
)


def write_professional_outputs(
    spec: RunSpecification,
    layout: ProjectLayout,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None = None,
) -> dict[str, Path]:
    """Write reports, tables, vector exports, and spatial QA evidence."""
    exports = export_publication_tables(
        analytics, sources, maps, layout.root / "exports"
    )
    vector_exports = _export_boundary(boundary_path, layout.root / "exports")
    planned_outputs = {
        **exports,
        **vector_exports,
        "validation_report": layout.root / "reports" / "validation_report.md",
        "interpretation": layout.root / "reports" / "interpretation.md",
        "html_report": layout.root / "reports" / "report.html",
        "pdf_report": layout.root / "reports" / "report.pdf",
        "dashboard": layout.root / "reports" / "dashboard.html",
        "quality_json": layout.root / "validation" / "quality_score.json",
        "quality_markdown": layout.root / "validation" / "quality_score.md",
        "quality_csv": layout.root / "validation" / "quality_score_components.csv",
    }
    quality_report = calculate_quality_score(
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
        maps=maps,
        downloads=planned_outputs,
    )
    quality_outputs = write_quality_outputs(layout.root / "validation", quality_report)
    validation = _write_validation_report(
        layout.root / "reports" / "validation_report.md",
        spec,
        boundary_path,
        scene_t1,
        scene_t2,
        analytics,
        availability,
    )
    interpretation_report = generate_interpretation(
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
        quality_report=quality_report,
    )
    interpretation = write_interpretation(
        layout.root / "reports" / "interpretation.md",
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        availability=availability,
        quality_report=quality_report,
    )
    html_report = _write_html_report(
        layout.root / "reports" / "report.html",
        spec,
        analytics,
        maps,
        sources,
        availability,
        interpretation_report,
        quality_report,
    )
    pdf_report = _write_pdf_report(
        layout.root / "reports" / "report.pdf",
        spec,
        analytics,
        maps,
        sources,
        availability,
        interpretation_report,
        quality_report,
    )
    dashboard = write_dashboard(
        layout.root / "reports" / "dashboard.html",
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        maps=maps,
        sources=sources,
        availability=availability,
        downloads={
            **exports,
            **vector_exports,
            **quality_outputs,
            "html_report": html_report,
            "pdf_report": pdf_report,
            "validation_report": validation,
            "interpretation": interpretation,
        },
        interpretation=interpretation_report,
        quality_report=quality_report,
    )
    portfolio_exports = export_portfolio_package(
        output_dir=layout.root / "portfolio_exports",
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        maps=maps,
        sources=sources,
        downloads={
            **exports,
            **vector_exports,
            **quality_outputs,
            "html_report": html_report,
            "pdf_report": pdf_report,
            "validation_report": validation,
            "interpretation": interpretation,
            "dashboard": dashboard,
        },
        interpretation=interpretation_report,
        quality_report=quality_report,
    )
    return {
        **exports,
        **vector_exports,
        **quality_outputs,
        "validation_report": validation,
        "interpretation": interpretation,
        "html_report": html_report,
        "pdf_report": pdf_report,
        "dashboard": dashboard,
        **portfolio_exports,
    }


def write_annual_master_report(
    spec: RunSpecification,
    layout: ProjectLayout,
    availability: AvailabilityPlan,
    comparison_reports: Mapping[str, Path],
) -> dict[str, Path]:
    """Write an annual timeline linking every adjacent-year publication."""
    reports_dir = layout.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "annual_master_report.html"
    pdf_path = reports_dir / "annual_master_report.pdf"
    timeline_rows = []
    pdf_rows: list[list[str]] = [["Year", "Scenes", "Dates", "Cloud (%)"]]
    for year, item in sorted(availability.years.items()):
        dates = ", ".join(item.acquired_dates)
        clouds = ", ".join(
            "unknown" if value is None else f"{value:.1f}" for value in item.cloud_cover
        )
        timeline_rows.append(
            f"<tr><td>{year}</td><td>{item.scene_count}</td>"
            f"<td>{html.escape(dates)}</td><td>{html.escape(clouds)}</td></tr>"
        )
        pdf_rows.append([str(year), str(item.scene_count), dates, clouds])
    report_links = []
    for label, report in comparison_reports.items():
        relative = report.relative_to(layout.root).as_posix()
        report_links.append(
            f'<li><a href="../{html.escape(relative)}">{html.escape(label)}</a></li>'
        )
    fallbacks = (
        "; ".join(availability.fallback_messages)
        if availability.fallback_messages
        else "No fallback required."
    )
    html_path.write_text(
        "".join(
            (
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<title>GeoWatch Annual Master Report</title>",
                "<style>body{font:15px Arial;max-width:1000px;margin:32px auto;",
                "color:#17202a}table{border-collapse:collapse;width:100%}",
                "td,th{border:1px solid #aaa;padding:8px}",
                "th{background:#e8eef3}</style>",
                "</head><body>",
                f"<h1>{html.escape(spec.location.name)} Annual Change Timeline</h1>",
                f"<p>{spec.temporal.start_year}-{spec.temporal.end_year} | "
                "Author: GeoWatch Project</p>",
                f"<p><strong>Common policy:</strong> {availability.dataset}, months "
                f"{availability.effective_start_month}-"
                f"{availability.effective_end_month}, "
                f"scene-cloud ceiling {availability.effective_cloud_cover:.0f}%.</p>",
                "<p><strong>Fallback disclosure:</strong> "
                f"{html.escape(fallbacks)}</p>",
                "<h2>Acquisition Timeline</h2><table><tr><th>Year</th><th>Scenes</th>",
                "<th>Dates</th><th>Cloud (%)</th></tr>",
                "".join(timeline_rows),
                "</table><h2>Adjacent-year Publications</h2><ul>",
                "".join(report_links),
                "</ul><h2>Interpretation</h2><p>Each comparison retains the selected "
                "indices and change algorithms. Unsupervised LULC remains exploratory "
                "and is not presented as independently validated accuracy.</p>"
                "</body></html>",
            )
        ),
        encoding="utf-8",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"{spec.location.name} Annual Change Timeline", styles["Title"]),
        Paragraph(
            f"{spec.temporal.start_year}-{spec.temporal.end_year} | GeoWatch Project",
            styles["Heading2"],
        ),
        Paragraph(
            f"Common policy: {availability.dataset}, months "
            f"{availability.effective_start_month}-{availability.effective_end_month}, "
            f"scene-cloud ceiling {availability.effective_cloud_cover:.0f}%.",
            styles["BodyText"],
        ),
        Paragraph(f"Fallback disclosure: {fallbacks}", styles["BodyText"]),
        Spacer(1, 12),
        Table(pdf_rows, repeatRows=1),
        Spacer(1, 12),
        Paragraph("Adjacent-year reports", styles["Heading2"]),
    ]
    story.extend(
        Paragraph(f"{label}: {report}", styles["BodyText"])
        for label, report in comparison_reports.items()
    )
    SimpleDocTemplate(str(pdf_path), pagesize=A4, author="GeoWatch Project").build(
        story
    )
    logger.info("Wrote annual master publication to {}", reports_dir)
    return {"annual_html": html_path, "annual_pdf": pdf_path}


def _write_validation_report(
    path: Path,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    availability: AvailabilityPlan | None,
) -> Path:
    boundary = load_vector_geometry(boundary_path)
    projected = reproject_geometry(boundary.geometry, boundary.crs, scene_t1.grid.crs)
    mask_t1 = geometry_mask_for_grid(projected, scene_t1.grid)
    projected_t2 = reproject_geometry(
        boundary.geometry, boundary.crs, scene_t2.grid.crs
    )
    mask_t2 = geometry_mask_for_grid(projected_t2, scene_t2.grid)
    outside_t1 = int((~mask_t1 & np.isfinite(scene_t1.data[0])).sum())
    outside_t2 = int((~mask_t2 & np.isfinite(scene_t2.data[0])).sum())
    aligned = _grids_are_spatially_aligned(scene_t1, scene_t2)
    invalid_t1 = _inside_invalid_fraction(scene_t1, mask_t1)
    invalid_t2 = _inside_invalid_fraction(scene_t2, mask_t2)
    fallback = (
        "; ".join(availability.fallback_messages)
        if availability and availability.fallback_messages
        else "None"
    )
    valid_t1 = scene_t1.metadata.get("valid_aoi_fraction", "unknown")
    valid_t2 = scene_t2.metadata.get("valid_aoi_fraction", "unknown")
    saturated_t1 = scene_t1.metadata.get("saturated_pixels_masked", "unknown")
    saturated_t2 = scene_t2.metadata.get("saturated_pixels_masked", "unknown")
    lines = [
        "# GeoWatch Spatial and Scientific Validation Report",
        "",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Location: {spec.location.name}, {spec.location.country}",
        f"- Boundary: `{boundary_path}`",
        f"- Boundary valid: {boundary.geometry.is_valid}",
        f"- Boundary CRS: {boundary.crs}",
        f"- Raster CRS: {scene_t1.grid.crs}",
        f"- Raster grids aligned: {aligned}",
        f"- Valid T1 pixels outside boundary: {outside_t1}",
        f"- Valid T2 pixels outside boundary: {outside_t2}",
        f"- T1 cloud/nodata fraction: {invalid_t1:.2%}",
        f"- T2 cloud/nodata fraction: {invalid_t2:.2%}",
        f"- T1 valid AOI fraction: {valid_t1}",
        f"- T2 valid AOI fraction: {valid_t2}",
        f"- T1 saturated AOI pixels masked: {saturated_t1}",
        f"- T2 saturated AOI pixels masked: {saturated_t2}",
        f"- Reflectance values outside 0-1 after compositing (T1/T2): "
        f"{scene_t1.metadata.get('reflectance_values_outside_0_1', 'unknown')} / "
        f"{scene_t2.metadata.get('reflectance_values_outside_0_1', 'unknown')}",
        f"- Availability fallback: {fallback}",
        "",
        "## Scientific Interpretation",
        "",
        (
            "Spectral results are measurements from atmospherically corrected "
            "surface reflectance. Unsupervised LULC is exploratory and is not "
            "reported as validated accuracy without independent reference samples."
        ),
        "",
        "## Index Means",
        "",
    ]
    for name, result in analytics.index_results.items():
        lines.append(
            f"- {name.upper()}: T1={result.statistics.t1.mean:.4f}, "
            f"T2={result.statistics.t2.mean:.4f}, "
            f"change={result.statistics.difference.mean:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _grids_are_spatially_aligned(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> bool:
    """Compare spatial grids without treating NaN nodata values as unequal."""
    grid_t1 = scene_t1.grid
    grid_t2 = scene_t2.grid
    return bool(
        grid_t1.crs == grid_t2.crs
        and grid_t1.width == grid_t2.width
        and grid_t1.height == grid_t2.height
        and np.allclose(
            grid_t1.transform,
            grid_t2.transform,
            rtol=0.0,
            atol=1e-9,
        )
    )


def _write_html_report(
    path: Path,
    spec: RunSpecification,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
    interpretation: InterpretationReport,
    quality_report: QualityScoreReport,
) -> Path:
    map_links = []
    for artifact in maps.values():
        preferred = artifact.files.get("png_300") or next(iter(artifact.files.values()))
        relative = preferred.relative_to(path.parent.parent).as_posix()
        map_links.append(
            f'<figure><img src="../{html.escape(relative)}" '
            f'alt="{html.escape(artifact.title)}">'
            f"<figcaption>{html.escape(artifact.title)}</figcaption></figure>"
        )
    index_rows = "".join(
        f"<tr><td>{name.upper()}</td><td>{result.statistics.t1.mean:.4f}</td>"
        f"<td>{result.statistics.t2.mean:.4f}</td><td>{result.statistics.difference.mean:.4f}</td></tr>"
        for name, result in analytics.index_results.items()
    )
    source_rows = "".join(_source_html(source) for source in sources)
    fallback_text = html.escape(
        "; ".join(availability.fallback_messages)
        if availability and availability.fallback_messages
        else "No search fallback was required."
    )
    signed_text = _signed_change_html(analytics)
    document = "\n".join(
        (
            '<!doctype html><html><head><meta charset="utf-8">',
            "<title>GeoWatch Report</title>",
            "<style>body{font:15px Arial;margin:0;color:#17202a}",
            "header{background:#16324f;color:white;padding:28px}",
            "main{max-width:1100px;margin:auto;padding:24px}",
            "figure{margin:24px 0}img{max-width:100%;border:1px solid #aaa}",
            "table{border-collapse:collapse;width:100%}",
            "td,th{border:1px solid #bbb;padding:7px}",
            "th{background:#e8eef3}</style></head><body>",
            f"<header><h1>{html.escape(spec.location.name)} Change Detection</h1>",
            f"<p>{spec.temporal.start_year} to {spec.temporal.end_year} | "
            f"{html.escape(spec.location.country)}</p></header><main>",
            "<h2>Executive Summary</h2><p>Generated from real satellite surface "
            "reflectance aligned to one projected grid and polygon-masked to the "
            "confirmed administrative boundary.</p>",
            _quality_html(quality_report),
            "<h2>Methodology and Preprocessing</h2><p>Scene selection prioritized "
            "AOI coverage, mission consistency, cloud cover, and seasonal proximity. "
            "Surface reflectance was scaled, QA cloud/shadow/snow/fill and saturation "
            "were masked, dates were composited, aligned, and clipped by polygon.</p>",
            f"<h3>Scene-selection fallback</h3><p>{fallback_text}</p>",
            render_interpretation_html(interpretation),
            f"<h2>Satellite Sources</h2><ul>{source_rows}</ul>",
            "<h2>Spectral Indices</h2><table><tr><th>Index</th>"
            "<th>Start mean</th><th>End mean</th><th>Difference</th></tr>",
            index_rows,
            f"</table><h2>NDVI change areas</h2>{signed_text}<h2>Maps</h2>",
            "".join(map_links),
            "<h2>Limitations</h2><p>Cloud cover, sensor history, boundaries, "
            "seasonal consistency, mixed land-cover pixels, and boundary quality "
            "affect interpretation. Unsupervised LULC is exploratory; no accuracy "
            "is claimed without independent reference labels.</p>"
            "<h2>Credits and Provenance</h2><p>Author: GeoWatch Project. Boundary: "
            f"{html.escape(spec.location.boundary_source or 'user-confirmed source')}. "
            f"Projection and scene identifiers are printed on maps and in tables. "
            f"Processed {datetime.now(UTC).date().isoformat()}.</p>"
            "</main></body></html>",
        )
    )
    path.write_text(document, encoding="utf-8")
    return path


def _write_pdf_report(
    path: Path,
    spec: RunSpecification,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    sources: Sequence[SceneMetadata],
    availability: AvailabilityPlan | None,
    interpretation: InterpretationReport,
    quality_report: QualityScoreReport,
) -> Path:
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"{spec.location.name} Satellite Change Report", styles["Title"]),
        Paragraph(
            f"{spec.temporal.start_year} to {spec.temporal.end_year} | "
            f"{spec.location.country}",
            styles["Heading2"],
        ),
        Spacer(1, 12),
        Paragraph("Author: GeoWatch Project", styles["BodyText"]),
        Paragraph("GeoWatch Quality Score", styles["Heading2"]),
        Paragraph(
            f"{quality_report.rounded_score}/{quality_report.max_score} | "
            f"{quality_report.overall_status} | "
            f"classification confidence: {quality_report.classification_confidence}",
            styles["BodyText"],
        ),
        Paragraph(
            "; ".join(
                f"{component.title}: {component.status}"
                for component in quality_report.components
            ),
            styles["BodyText"],
        ),
        Paragraph("Methodology", styles["Heading2"]),
        Paragraph(
            "Real surface-reflectance scenes were selected for AOI coverage, mission "
            "consistency, cloud cover, and seasonal proximity. QA cloud, shadow, snow, "
            "fill, saturation, and outside-boundary pixels were masked before "
            "analysis.",
            styles["BodyText"],
        ),
        Paragraph("Scene-selection fallback", styles["Heading3"]),
        Paragraph(
            (
                "; ".join(availability.fallback_messages)
                if availability and availability.fallback_messages
                else "No search fallback was required."
            ),
            styles["BodyText"],
        ),
        Paragraph("Satellite Sources", styles["Heading2"]),
    ]
    story.extend(_interpretation_pdf_story(interpretation, styles))
    story.extend(
        Paragraph(
            _source_text(source),
            styles["BodyText"],
        )
        for source in sources
    )
    rows = [["Index", "Start", "End", "Difference"]]
    rows.extend(
        [
            name.upper(),
            f"{result.statistics.t1.mean:.4f}",
            f"{result.statistics.t2.mean:.4f}",
            f"{result.statistics.difference.mean:.4f}",
        ]
        for name, result in analytics.index_results.items()
    )
    story.extend(
        [Spacer(1, 12), Paragraph("Index Statistics", styles["Heading2"]), Table(rows)]
    )
    if analytics.signed_change is not None:
        signed = analytics.signed_change
        story.extend(
            [
                Spacer(1, 12),
                Paragraph("NDVI Gain / No-change / Loss", styles["Heading2"]),
                Paragraph(
                    f"Threshold: +/-{signed.threshold:.4f} NDVI. "
                    + ", ".join(
                        f"{key}: {value} pixels" for key, value in signed.counts.items()
                    ),
                    styles["BodyText"],
                ),
            ]
        )
    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Limitations", styles["Heading2"]),
            Paragraph(
                "Unsupervised LULC is exploratory. No accuracy value is claimed "
                "without independent reference data.",
                styles["BodyText"],
            ),
        ]
    )
    for warning in quality_report.warnings:
        story.append(Paragraph(f"Warning: {warning}", styles["BodyText"]))
    for artifact in maps.values():
        image_path = artifact.files.get("png_300")
        if image_path is None:
            continue
        story.extend(
            [
                PageBreak(),
                Paragraph(artifact.title, styles["Heading1"]),
                Image(str(image_path), width=6.7 * inch, height=5.18 * inch),
                Paragraph(artifact.description, styles["BodyText"]),
            ]
        )
    SimpleDocTemplate(
        str(path), pagesize=A4, author="GeoWatch Project", title=path.stem
    ).build(story)
    return path


def _interpretation_pdf_story(
    interpretation: InterpretationReport,
    styles: Mapping[str, Any],
) -> list[Paragraph | Spacer]:
    """Build PDF paragraphs for the rule-based interpretation."""
    body_style = styles["BodyText"]
    heading_style = styles["Heading2"]
    story: list[Paragraph | Spacer] = [
        Paragraph("Analyst Interpretation", heading_style),
    ]
    for section in interpretation.sections:
        story.append(Paragraph(section.title, styles["Heading3"]))
        for paragraph in section.paragraphs[:2]:
            story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 6))
    return story


def _inside_invalid_fraction(scene: RasterLayer, inside: np.ndarray) -> float:
    """Calculate invalid coverage using AOI pixels as the denominator."""
    denominator = int(inside.sum())
    if denominator == 0:
        return 1.0
    invalid = ~np.isfinite(scene.data[0])
    if scene.cloud_mask is not None:
        invalid |= scene.cloud_mask
    return float((invalid & inside).sum() / denominator)


def _source_text(source: SceneMetadata) -> str:
    """Format one source record consistently for reports."""
    acquired = (
        source.acquired_at.date().isoformat() if source.acquired_at else "unknown date"
    )
    return (
        f"{source.scene_id} | {source.dataset} | {acquired} | "
        f"cloud={source.cloud_cover} | {source.provider}"
    )


def _source_html(source: SceneMetadata) -> str:
    """Escape one source record for HTML."""
    return f"<li>{html.escape(_source_text(source))}</li>"


def _signed_change_html(analytics: AnalyticsReport) -> str:
    """Render signed NDVI counts without inventing accuracy claims."""
    if analytics.signed_change is None:
        return "<p>Signed NDVI change was not requested.</p>"
    signed = analytics.signed_change
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in signed.counts.items()
    )
    return (
        f"<p>Documented threshold: +/-{signed.threshold:.4f} NDVI.</p>"
        f"<table><tr><th>Class</th><th>Pixels</th></tr>{rows}</table>"
    )


def _quality_html(report: QualityScoreReport) -> str:
    """Render the run-quality summary for the static HTML report."""
    rows = "".join(
        f"<tr><td>{html.escape(component.title)}</td>"
        f"<td>{html.escape(component.status)}</td>"
        f"<td>{component.score:.1f}/{component.weight}</td></tr>"
        for component in report.components
    )
    warnings = (
        "".join(f"<li>{html.escape(warning)}</li>" for warning in report.warnings)
        if report.warnings
        else "<li>No run-quality warnings were generated.</li>"
    )
    return (
        "<h2>GeoWatch Quality Score</h2>"
        f"<p><strong>{report.rounded_score}/{report.max_score}</strong> | "
        f"{html.escape(report.overall_status)} | "
        "classification confidence: "
        f"{html.escape(report.classification_confidence)}</p>"
        "<table><tr><th>Component</th><th>Status</th><th>Score</th></tr>"
        f"{rows}</table><h3>Warnings</h3><ul>{warnings}</ul>"
    )


def _export_boundary(path: Path, output_dir: Path) -> dict[str, Path]:
    frame = gpd.read_file(path)
    geojson = output_dir / "administrative_boundary.geojson"
    shapefile = output_dir / "administrative_boundary.shp"
    geopackage = output_dir / "administrative_boundary.gpkg"
    frame.to_file(geojson, driver="GeoJSON")
    frame.to_file(shapefile, driver="ESRI Shapefile")
    frame.to_file(geopackage, driver="GPKG", layer="administrative_boundary")
    logger.info("Exported administrative boundary in three GIS formats.")
    return {
        "boundary_geojson": geojson,
        "boundary_shapefile": shapefile,
        "boundary_gpkg": geopackage,
    }
