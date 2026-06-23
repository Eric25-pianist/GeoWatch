"""Phase 5 cartography and reporting orchestration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from jinja2 import BaseLoader, Environment, select_autoescape
from loguru import logger
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import AnalyticsReport
from geowatch.analytics.pipeline import run_analytics_pipeline
from geowatch.application.models import (
    AnalysisSpec,
    ImagerySpec,
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.config.models import ProjectConfig
from geowatch.portfolio.exporter import export_portfolio_package
from geowatch.reporting.cartography import render_cartography_suite
from geowatch.reporting.dashboard import write_dashboard
from geowatch.reporting.demo import DemoPublicationInputs, build_demo_publication_inputs
from geowatch.reporting.exports import export_publication_tables
from geowatch.reporting.interpretation import (
    InterpretationReport,
    generate_interpretation,
    render_interpretation_html,
    write_interpretation,
)
from geowatch.reporting.models import MapArtifact, PublicationBundle
from geowatch.utils.paths import ensure_parent
from geowatch.validation.quality_score import (
    QualityScoreReport,
    calculate_quality_score,
    write_quality_outputs,
)


def build_phase5_publication(config: ProjectConfig) -> PublicationBundle:
    """Generate the Phase 5 example publication outputs."""
    inputs = build_demo_publication_inputs(config)
    analytics_report = run_analytics_pipeline(
        inputs.scene_t1,
        inputs.scene_t2,
        output_root=config.outputs.root,
        classification_method="random_forest",
        training_labels_t1=inputs.training_labels_t1,
        training_labels_t2=inputs.training_labels_t2,
        reference_labels_t1=inputs.reference_labels_t1,
        reference_labels_t2=inputs.reference_labels_t2,
    )

    maps_dir = config.outputs.maps / "phase5"
    reports_dir = config.outputs.reports
    exports_dir = config.outputs.exports / "phase5"
    validation_dir = config.outputs.root / "validation"
    maps_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    map_artifacts = render_cartography_suite(
        config,
        inputs.scene_t1,
        inputs.scene_t2,
        analytics_report,
        output_dir=maps_dir,
    )
    exports = export_publication_tables(
        analytics_report,
        inputs.sources,
        map_artifacts,
        exports_dir,
    )
    dashboard_spec, dashboard_boundary = _dashboard_specification(config)
    planned_outputs = {
        **exports,
        "html_report": reports_dir / "report.html",
        "pdf_report": reports_dir / "report.pdf",
        "dashboard": reports_dir / "dashboard.html",
        "interpretation": reports_dir / "interpretation.md",
        "quality_json": validation_dir / "quality_score.json",
        "quality_markdown": validation_dir / "quality_score.md",
        "quality_csv": validation_dir / "quality_score_components.csv",
    }
    quality_report = calculate_quality_score(
        spec=dashboard_spec,
        boundary_path=dashboard_boundary,
        scene_t1=inputs.scene_t1,
        scene_t2=inputs.scene_t2,
        analytics=analytics_report,
        sources=inputs.sources,
        maps=map_artifacts,
        downloads=planned_outputs,
    )
    quality_outputs = write_quality_outputs(validation_dir, quality_report)
    interpretation_report = generate_interpretation(
        spec=dashboard_spec,
        boundary_path=dashboard_boundary,
        scene_t1=inputs.scene_t1,
        scene_t2=inputs.scene_t2,
        analytics=analytics_report,
        sources=inputs.sources,
        quality_report=quality_report,
    )
    interpretation = write_interpretation(
        reports_dir / "interpretation.md",
        spec=dashboard_spec,
        boundary_path=dashboard_boundary,
        scene_t1=inputs.scene_t1,
        scene_t2=inputs.scene_t2,
        analytics=analytics_report,
        sources=inputs.sources,
        quality_report=quality_report,
    )
    html_report = _render_html_report(
        config=config,
        publication_inputs=inputs,
        analytics_report=analytics_report,
        map_artifacts=map_artifacts,
        interpretation=interpretation_report,
        quality_report=quality_report,
        output_path=reports_dir / "report.html",
    )
    pdf_report = _render_pdf_report(
        config=config,
        publication_inputs=inputs,
        analytics_report=analytics_report,
        map_artifacts=map_artifacts,
        interpretation=interpretation_report,
        quality_report=quality_report,
        output_path=reports_dir / "report.pdf",
    )
    dashboard = write_dashboard(
        reports_dir / "dashboard.html",
        spec=dashboard_spec,
        boundary_path=dashboard_boundary,
        scene_t1=inputs.scene_t1,
        scene_t2=inputs.scene_t2,
        analytics=analytics_report,
        maps=map_artifacts,
        sources=inputs.sources,
        downloads={
            **exports,
            **quality_outputs,
            "html_report": html_report,
            "pdf_report": pdf_report,
            "interpretation": interpretation,
        },
        interpretation=interpretation_report,
        quality_report=quality_report,
    )
    portfolio_exports = export_portfolio_package(
        output_dir=config.outputs.root / "portfolio_exports",
        spec=dashboard_spec,
        boundary_path=dashboard_boundary,
        scene_t1=inputs.scene_t1,
        scene_t2=inputs.scene_t2,
        analytics=analytics_report,
        maps=map_artifacts,
        sources=inputs.sources,
        downloads={
            **exports,
            **quality_outputs,
            "html_report": html_report,
            "pdf_report": pdf_report,
            "dashboard": dashboard,
            "interpretation": interpretation,
        },
        interpretation=interpretation_report,
        quality_report=quality_report,
    )
    bundle = PublicationBundle(
        project_name=config.project_name,
        generated_at=datetime.now(UTC),
        aoi=config.aoi,
        sources=inputs.sources,
        analytics_report=analytics_report,
        maps=map_artifacts,
        exports=exports,
        html_report=html_report,
        pdf_report=pdf_report,
        dashboard=dashboard,
        interpretation=interpretation,
        build_report=Path("BUILD_REPORT.md"),
        portfolio_exports=portfolio_exports,
        example_outputs={
            "html_report": html_report,
            "pdf_report": pdf_report,
            "dashboard": dashboard,
            "interpretation": interpretation,
            "quality_report": quality_outputs["quality_markdown"],
            "portfolio_directory": config.outputs.root / "portfolio_exports",
            "map_directory": maps_dir,
            "export_directory": exports_dir,
        },
    )
    logger.info("Built Phase 5 publication bundle for {}", config.project_name)
    return bundle


def _dashboard_specification(
    config: ProjectConfig,
) -> tuple[RunSpecification, Path]:
    """Adapt the compatibility publication configuration for the dashboard."""
    if config.aoi.bbox is None:
        if config.aoi.path is None:
            raise ValueError("Dashboard generation requires a configured AOI.")
        boundary_path = config.aoi.path
    else:
        west, south, east, north = config.aoi.bbox
        boundary_path = config.outputs.vectors / "dashboard_aoi.geojson"
        boundary_path.parent.mkdir(parents=True, exist_ok=True)
        boundary_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"source": "configured_bbox"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [west, south],
                                        [east, south],
                                        [east, north],
                                        [west, north],
                                        [west, south],
                                    ]
                                ],
                            },
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    start_year = config.dates.start_date.year
    end_year = config.dates.end_date.year
    temporal = TemporalSpec.model_construct(
        start_year=start_year,
        end_year=end_year,
        start_month=config.dates.start_date.month,
        end_month=config.dates.end_date.month,
        mode="endpoints",
        interval_years=1,
    )
    specification = RunSpecification.model_construct(
        schema_version="1.0",
        location=LocationSpec(
            name=config.project_name,
            country="Not specified",
            boundary_path=boundary_path,
            boundary_source=f"Configured {config.aoi.kind} AOI",
        ),
        temporal=temporal,
        imagery=ImagerySpec(
            max_cloud_cover=config.processing.max_cloud_cover,
        ),
        analysis=AnalysisSpec(),
        outputs=OutputSpec(
            root=config.outputs.root, map_theme=config.outputs.map_theme
        ),
    )
    return specification, boundary_path


def write_build_report(
    bundle: PublicationBundle,
    validation_summary: Mapping[str, object],
    path: Path,
) -> Path:
    """Write the final build report to disk."""
    path = ensure_parent(path)
    markdown = _render_build_report_markdown(bundle, validation_summary)
    path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote build report to {}", path)
    return path


def write_phase_report(
    bundle: PublicationBundle,
    validation_summary: Mapping[str, object],
    path: Path,
) -> Path:
    """Write the final phase completion report to disk."""
    path = ensure_parent(path)
    markdown = _render_phase_report_markdown(bundle, validation_summary)
    path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote phase report to {}", path)
    return path


def _render_html_report(
    *,
    config: ProjectConfig,
    publication_inputs: DemoPublicationInputs,
    analytics_report: AnalyticsReport,
    map_artifacts: Mapping[str, MapArtifact],
    interpretation: InterpretationReport,
    quality_report: QualityScoreReport,
    output_path: Path,
) -> Path:
    """Render the interactive HTML report."""
    output_path = ensure_parent(output_path)
    template = Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(["html", "xml"]),
    ).from_string(_HTML_TEMPLATE)
    tables = _build_html_tables(
        config=config,
        publication_inputs=publication_inputs,
        analytics_report=analytics_report,
        map_artifacts=map_artifacts,
    )
    charts = _build_plotly_charts(analytics_report)
    leaflet_layers = _build_leaflet_layers(output_path.parent, map_artifacts)
    html = template.render(
        project_name=config.project_name,
        generated_at=datetime.now(UTC).isoformat(),
        tables=tables,
        charts=json.dumps(charts),
        leaflet_layers=json.dumps(leaflet_layers),
        report_summary=analytics_report.summary(),
        aoi_description=_describe_aoi(config),
        interpretation_html=render_interpretation_html(interpretation),
        quality_score=quality_report.rounded_score,
        quality_status=quality_report.overall_status,
        quality_components=quality_report.components,
        quality_warnings=quality_report.warnings,
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote HTML report to {}", output_path)
    return output_path


def _render_pdf_report(
    *,
    config: ProjectConfig,
    publication_inputs: DemoPublicationInputs,
    analytics_report: AnalyticsReport,
    map_artifacts: Mapping[str, MapArtifact],
    interpretation: InterpretationReport,
    quality_report: QualityScoreReport,
    output_path: Path,
) -> Path:
    """Render the multi-page PDF report."""
    output_path = ensure_parent(output_path)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"{config.project_name} Phase 5 Report",
        author="GeoWatch",
    )
    styles = _pdf_styles()
    story: list[Any] = []
    story.extend(
        [
            Paragraph(
                f"{config.project_name} Phase 5 Report",
                styles["Title"],
            ),
            Spacer(1, 0.08 * inch),
            Paragraph("Cartography and Reporting", styles["Subtitle"]),
            Spacer(1, 0.12 * inch),
            Paragraph(analytics_report.summary(), styles["Body"]),
            Spacer(1, 0.12 * inch),
            Paragraph(
                f"GeoWatch Quality Score: {quality_report.rounded_score}/{quality_report.max_score} ({quality_report.overall_status})",
                styles["Body"],
            ),
            Spacer(1, 0.08 * inch),
        ]
    )
    story.extend(_pdf_table_block("AOI Information", _pdf_aoi_table(config), styles))
    story.append(Spacer(1, 0.1 * inch))
    story.extend(
        _pdf_table_block(
            "Satellite Sources",
            _pdf_sources_table(publication_inputs.sources),
            styles,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.extend(_pdf_interpretation_paragraphs(interpretation, styles))
    story.append(Spacer(1, 0.1 * inch))
    story.extend(_pdf_section("Methodology", _pdf_methodology_paragraphs(styles)))
    story.extend(_pdf_section("Preprocessing", _pdf_preprocessing_paragraphs(styles)))
    story.extend(
        _pdf_table_block("Indices", _pdf_indices_table(analytics_report), styles)
    )
    story.extend(
        _pdf_table_block(
            "Change Detection",
            _pdf_change_table(analytics_report),
            styles,
        )
    )
    story.extend(
        _pdf_table_block("LULC", _pdf_classification_table(analytics_report), styles)
    )
    story.extend(
        _pdf_section("Statistics", _pdf_statistics_paragraphs(analytics_report, styles))
    )
    story.extend(
        _pdf_section("Recommendations", _pdf_recommendations_paragraphs(styles))
    )
    story.extend(_pdf_section("Limitations", _pdf_limitations_paragraphs(styles)))
    story.extend(
        _pdf_section(
            "Appendix", _pdf_appendix_paragraphs(bundle=map_artifacts, styles=styles)
        )
    )

    for map_name in (
        "ndvi",
        "lulc",
        "change_detection",
        "hotspot_analysis",
        "before_after",
    ):
        artifact = map_artifacts[map_name]
        image_path = artifact.files.get("png_300")
        if image_path is not None and image_path.exists():
            story.append(Spacer(1, 0.12 * inch))
            story.append(Paragraph(artifact.title, styles["Section"]))
            story.append(Image(str(image_path), width=7.1 * inch, height=4.5 * inch))

    doc.build(story, onFirstPage=_pdf_page_decorator, onLaterPages=_pdf_page_decorator)
    logger.info("Wrote PDF report to {}", output_path)
    return output_path


def _render_build_report_markdown(
    bundle: PublicationBundle,
    validation_summary: Mapping[str, object],
) -> str:
    """Render the final build report markdown."""
    lines = [
        "# GeoWatch Build Report",
        "",
        "## Folder Structure",
        "",
        "- `outputs/analytics/`",
        "- `outputs/maps/phase5/`",
        "- `outputs/reports/`",
        "- `outputs/exports/phase5/`",
        "- `logs/`",
        "",
        "## Files Generated",
        "",
    ]
    lines.extend(f"- {path}" for path in _flatten_publication_paths(bundle))
    lines.extend(
        [
            "",
            "## Dependencies",
            "",
            "- NumPy, SciPy, PyProj, Matplotlib, Pandas, Jinja2, ReportLab, XlsxWriter",
            "- Typer, Loguru, Pydantic, PyYAML, scikit-learn",
            "",
            "## Algorithms",
            "",
            "- Spectral indices: NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI, NDMI, GNDVI, NBR",
            "- Change detection: index differencing, CVA, PCA, MAD, IRMAD, ratioing, magnitude mapping",
            "- Hotspot analysis: Getis-Ord Gi*",
            "- Cartography: continuous, discrete, and comparison map rendering",
            "- Reporting: HTML, PDF, CSV, JSON, Excel",
            "- Portfolio packaging: infographic, showcase maps, short PDF, README snippet, metadata JSON",
            "",
            "## Tests",
            "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in validation_summary.items())
    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            "- The publication bundle uses synthetic demo scenes for repeatable validation.",
            "- The interactive HTML report loads Plotly and Leaflet from public CDNs.",
            "- PDF output is optimized for narrative review rather than deep layout authoring.",
            "",
            "## Usage Examples",
            "",
            "```powershell",
            "geowatch publish --config configs/default.yaml",
            "geowatch validate configs/default.yaml",
            "python -m pytest",
            "```",
            "",
            "## Performance Notes",
            "",
            "- Map rendering is CPU-bound and scales with the chosen DPI.",
            "- The 600 DPI map exports are intentionally heavier than the 300 DPI versions.",
            "- PDF generation is fastest when the report image set stays compact.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_phase_report_markdown(
    bundle: PublicationBundle,
    validation_summary: Mapping[str, object],
) -> str:
    """Render the final phase completion report markdown."""
    lines = [
        "# GeoWatch Phase 5 Report",
        "",
        "- Phase: 5 - Cartography and Reporting",
        "- Status: PASS",
        f"- Generated: {bundle.generated_at.isoformat()}",
        "",
        "## Completed Scope",
        "",
        "- Generated professional GIS-style maps for NDVI, NDBI, NDWI, LULC, change detection, hotspot analysis, and before/after comparison.",
        "- Exported maps to PNG, PDF, and SVG at 300 and 600 DPI.",
        "- Built a Leaflet-based interactive HTML report with embedded Plotly charts and HTML tables.",
        "- Built a multi-page PDF report with publication-ready sectioning and imagery.",
        "- Generated a portfolio_exports package with infographic, showcase maps, a short PDF brief, and reusable metadata.",
        "- Exported summary tables to CSV, JSON, and Excel.",
        "- Generated the final build report and example outputs.",
        "",
        "## Validation Results",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in validation_summary.items())
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- HTML report: `{bundle.html_report}`",
            f"- PDF report: `{bundle.pdf_report}`",
            f"- Interpretation: `{bundle.interpretation}`",
            f"- Build report: `{bundle.build_report}`",
            f"- Export directory: `{bundle.example_outputs['export_directory']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_html_tables(
    *,
    config: ProjectConfig,
    publication_inputs: DemoPublicationInputs,
    analytics_report: AnalyticsReport,
    map_artifacts: Mapping[str, MapArtifact],
) -> dict[str, str]:
    """Build HTML tables for the publication report."""
    aoi_df = pd.DataFrame(
        [
            {
                "project": config.project_name,
                "aoi_kind": config.aoi.kind,
                "bbox": config.aoi.bbox,
                "crs": config.aoi.crs,
            }
        ]
    )
    sources_df = pd.DataFrame(
        [
            {
                "scene_id": source.scene_id,
                "provider": source.provider,
                "dataset": source.dataset,
                "cloud_cover": source.cloud_cover,
                "acquired_at": (
                    source.acquired_at.isoformat() if source.acquired_at else None
                ),
            }
            for source in publication_inputs.sources
        ]
    )
    indices_df = pd.DataFrame(
        [
            {
                "index": name,
                "t1_mean": result.statistics.t1.mean,
                "t2_mean": result.statistics.t2.mean,
                "difference_mean": result.statistics.difference.mean,
            }
            for name, result in analytics_report.index_results.items()
        ]
    )
    change_df = pd.DataFrame(
        [
            {
                "method": name,
                "score_mean": result.statistics.mean,
                "change_fraction": (
                    result.threshold.change_fraction
                    if result.threshold is not None
                    else 0.0
                ),
            }
            for name, result in analytics_report.change_results.items()
        ]
    )
    lulc_df = pd.DataFrame(
        [
            {
                "scene": name,
                "method": result.method,
                "model_name": result.model_name,
                "counts": json.dumps(result.counts),
            }
            for name, result in analytics_report.classification_results.items()
        ]
    )
    statistics_df = pd.DataFrame(
        [
            {
                "artifact": name,
                "title": artifact.title,
                "files": len(artifact.files),
            }
            for name, artifact in map_artifacts.items()
        ]
    )
    return {
        "aoi": aoi_df.to_html(index=False, border=0, classes="table"),
        "sources": sources_df.to_html(index=False, border=0, classes="table"),
        "indices": indices_df.to_html(index=False, border=0, classes="table"),
        "change": change_df.to_html(index=False, border=0, classes="table"),
        "lulc": lulc_df.to_html(index=False, border=0, classes="table"),
        "statistics": statistics_df.to_html(index=False, border=0, classes="table"),
    }


def _build_plotly_charts(analytics_report: AnalyticsReport) -> list[dict[str, str]]:
    """Build Plotly chart payloads for the HTML report."""
    index_names = list(analytics_report.index_results)
    index_values = [
        analytics_report.index_results[name].statistics.difference.mean
        for name in index_names
    ]
    change_methods = list(analytics_report.change_results)
    change_fractions = [
        (
            threshold.change_fraction
            if (threshold := analytics_report.change_results[name].threshold)
            is not None
            else 0.0
        )
        for name in change_methods
    ]
    lulc_result = analytics_report.classification_results["lulc_t2"]
    class_names = list(lulc_result.class_names)
    class_counts = [lulc_result.counts[name] for name in class_names]
    return [
        {
            "div_id": "index-chart",
            "title": "Spectral Index Differences",
            "data": json.dumps(
                [
                    {
                        "type": "bar",
                        "x": index_names,
                        "y": index_values,
                        "marker": {"color": "#2563eb"},
                    }
                ]
            ),
            "layout": json.dumps(
                {
                    "height": 320,
                    "margin": {"l": 50, "r": 20, "t": 30, "b": 70},
                    "yaxis": {"title": "Difference mean"},
                }
            ),
        },
        {
            "div_id": "change-chart",
            "title": "Change Fractions",
            "data": json.dumps(
                [
                    {
                        "type": "bar",
                        "x": change_methods,
                        "y": change_fractions,
                        "marker": {"color": "#f97316"},
                    }
                ]
            ),
            "layout": json.dumps(
                {
                    "height": 320,
                    "margin": {"l": 50, "r": 20, "t": 30, "b": 70},
                    "yaxis": {"title": "Fraction"},
                }
            ),
        },
        {
            "div_id": "lulc-chart",
            "title": "LULC Class Counts",
            "data": json.dumps(
                [
                    {
                        "type": "bar",
                        "x": class_names,
                        "y": class_counts,
                        "marker": {"color": "#16a34a"},
                    }
                ]
            ),
            "layout": json.dumps(
                {
                    "height": 320,
                    "margin": {"l": 50, "r": 20, "t": 30, "b": 70},
                    "yaxis": {"title": "Pixels"},
                }
            ),
        },
    ]


def _build_leaflet_layers(
    report_dir: Path,
    map_artifacts: Mapping[str, MapArtifact],
) -> list[dict[str, object]]:
    """Build Leaflet image overlays for the interactive map."""
    layers: list[dict[str, object]] = []
    for name in ("ndvi", "lulc", "change_detection", "hotspot_analysis"):
        artifact = map_artifacts[name]
        overlay_path = artifact.files.get("overlay_png")
        bounds = artifact.metadata.get("bounds_wgs84")
        if overlay_path is None or bounds is None:
            continue
        rel_path = os.path.relpath(overlay_path, report_dir)
        west, south, east, north = cast(tuple[float, float, float, float], bounds)
        layers.append(
            {
                "name": artifact.title,
                "path": rel_path.replace("\\", "/"),
                "bounds": [[south, west], [north, east]],
            }
        )
    return layers


def _pdf_styles() -> dict[str, ParagraphStyle]:
    """Build PDF paragraph styles."""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#243b53"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontSize=13,
            leading=15,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#102a43"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12,
            alignment=TA_LEFT,
        )
    )
    return {
        "Title": styles["Title"],
        "Subtitle": styles["Subtitle"],
        "Section": styles["Section"],
        "Body": styles["Body"],
    }


def _pdf_table_block(
    title: str,
    table: Table,
    styles: Mapping[str, ParagraphStyle],
) -> list[Any]:
    """Create a PDF section with a heading and a table."""
    return [Paragraph(title, styles["Section"]), Spacer(1, 0.05 * inch), table]


def _pdf_aoi_table(config: ProjectConfig) -> Table:
    """Build an AOI summary table for the PDF report."""
    rows = [
        ["Project", config.project_name],
        ["AOI kind", config.aoi.kind],
        ["CRS", config.aoi.crs],
    ]
    if config.aoi.bbox is not None:
        rows.append(["BBox", ", ".join(f"{value:.4f}" for value in config.aoi.bbox)])
    return _styled_table(rows)


def _pdf_sources_table(sources: Sequence[SceneMetadata]) -> Table:
    """Build a satellite sources table for the PDF report."""
    rows = [["Scene ID", "Provider", "Dataset", "Cloud Cover"]]
    for source in sources:
        rows.append(
            [
                source.scene_id,
                source.provider,
                source.dataset,
                (
                    f"{source.cloud_cover:.1f}%"
                    if source.cloud_cover is not None
                    else "n/a"
                ),
            ]
        )
    return _styled_table(rows, header=True)


def _pdf_indices_table(analytics_report: AnalyticsReport) -> Table:
    """Build an indices summary table for the PDF report."""
    rows = [["Index", "T1 Mean", "T2 Mean", "Difference Mean"]]
    for name, result in analytics_report.index_results.items():
        rows.append(
            [
                name.upper(),
                f"{result.statistics.t1.mean:.4f}",
                f"{result.statistics.t2.mean:.4f}",
                f"{result.statistics.difference.mean:.4f}",
            ]
        )
    return _styled_table(rows, header=True)


def _pdf_change_table(analytics_report: AnalyticsReport) -> Table:
    """Build a change detection summary table for the PDF report."""
    rows = [["Method", "Mean", "Change Fraction"]]
    for name, result in analytics_report.change_results.items():
        rows.append(
            [
                name,
                f"{result.statistics.mean:.4f}",
                (
                    f"{result.threshold.change_fraction:.2%}"
                    if result.threshold
                    else "0.00%"
                ),
            ]
        )
    return _styled_table(rows, header=True)


def _pdf_classification_table(analytics_report: AnalyticsReport) -> Table:
    """Build a classification summary table for the PDF report."""
    rows = [["Scene", "Method", "Model"]]
    for name, result in analytics_report.classification_results.items():
        rows.append([name, result.method, result.model_name])
    return _styled_table(rows, header=True)


def _pdf_methodology_paragraphs(
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF methodology paragraphs."""
    return [
        Paragraph(
            "GeoWatch synthesizes a repeatable demo scene, runs the Phase 4 analytics pipeline, and then "
            "renders publication-grade map products using Matplotlib, ReportLab, Plotly, and Leaflet.",
            styles["Body"],
        )
    ]


def _pdf_preprocessing_paragraphs(
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF preprocessing paragraphs."""
    return [
        Paragraph(
            "The demo publication uses projected synthetic scenes with fixed bands, consistent AOI bounds, "
            "and classification labels designed to exercise the cartographic outputs.",
            styles["Body"],
        )
    ]


def _pdf_statistics_paragraphs(
    analytics_report: AnalyticsReport,
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF statistics paragraphs."""
    return [
        Paragraph(
            f"Generated {len(analytics_report.index_results)} spectral index bundles, "
            f"{len(analytics_report.change_results)} change surfaces, and "
            f"{len(analytics_report.classification_results)} classification outputs.",
            styles["Body"],
        )
    ]


def _pdf_interpretation_paragraphs(
    interpretation: InterpretationReport,
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF paragraphs for the analyst interpretation."""
    paragraphs: list[Paragraph] = [
        Paragraph("Analyst Interpretation", styles["Section"])
    ]
    for section in interpretation.sections:
        paragraphs.append(Paragraph(section.title, styles["Section"]))
        for text in section.paragraphs[:2]:
            paragraphs.append(Paragraph(text, styles["Body"]))
    return paragraphs


def _pdf_recommendations_paragraphs(
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF recommendations paragraphs."""
    return [
        Paragraph(
            "Use the 600 DPI map exports for board-level review and the HTML report for interactive exploration.",
            styles["Body"],
        )
    ]


def _pdf_limitations_paragraphs(
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF limitation paragraphs."""
    return [
        Paragraph(
            "The publication bundle uses synthetic data and remote CDNs for the interactive assets.",
            styles["Body"],
        )
    ]


def _pdf_appendix_paragraphs(
    *,
    bundle: Mapping[str, MapArtifact],
    styles: Mapping[str, ParagraphStyle],
) -> list[Paragraph]:
    """Return PDF appendix paragraphs."""
    return [
        Paragraph(
            f"Map outputs generated: {', '.join(bundle.keys())}.",
            styles["Body"],
        )
    ]


def _styled_table(rows: list[list[str]], *, header: bool = False) -> Table:
    """Create a styled ReportLab table."""
    table = Table(rows, repeatRows=1 if header else 0, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f8fb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd2d9")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _pdf_page_decorator(canvas: Any, doc: Any) -> None:
    """Add page numbers and footer branding to the PDF report."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#334e68"))
    canvas.drawString(
        doc.leftMargin, 0.35 * inch, "GeoWatch Phase 5 Cartography and Reporting"
    )
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.35 * inch,
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


def _pdf_section(title: str, paragraphs: Sequence[Paragraph]) -> list[Any]:
    """Create a reusable PDF section block."""
    flowables: list[Any] = [
        Paragraph(title, _pdf_styles()["Section"]),
        Spacer(1, 0.04 * inch),
    ]
    flowables.extend(paragraphs)
    return flowables


def _describe_aoi(config: ProjectConfig) -> str:
    """Render a short AOI description for the HTML report."""
    if config.aoi.bbox is None:
        return f"{config.aoi.kind} AOI"
    west, south, east, north = config.aoi.bbox
    return f"{config.aoi.kind} AOI: {west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}"


def _flatten_publication_paths(bundle: PublicationBundle) -> list[str]:
    """Return a flat list of generated file paths for the build report."""
    paths: list[str] = [
        str(bundle.html_report),
        str(bundle.pdf_report),
        str(bundle.dashboard),
        str(bundle.interpretation),
        str(bundle.build_report),
    ]
    for name, artifact in bundle.maps.items():
        for key, path in artifact.files.items():
            paths.append(f"{name}:{key} -> {path}")
    for name, path in bundle.exports.items():
        paths.append(f"export:{name} -> {path}")
    for name, path in bundle.portfolio_exports.items():
        paths.append(f"portfolio:{name} -> {path}")
    return paths


_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ project_name }} Phase 5 Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f6f8fb; color: #102a43; }
    header { background: #102a43; color: white; padding: 24px 32px; }
    main { padding: 24px 32px 48px; }
    section { margin-bottom: 28px; background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 20px; }
    h1, h2, h3 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
    .chart { width: 100%; height: 320px; }
    .map { width: 100%; height: 520px; border: 1px solid #d9e2ec; border-radius: 4px; }
    .table { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
    .table th, .table td { border: 1px solid #d9e2ec; padding: 8px 10px; vertical-align: top; }
    .table th { background: #102a43; color: white; text-align: left; }
    .caption { color: #52606d; font-size: 0.92rem; }
    .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #d9e2ec; margin-right: 8px; font-size: 0.85rem; }
  </style>
</head>
<body>
  <header>
    <h1>{{ project_name }} Phase 5 Report</h1>
    <div class="caption">{{ generated_at }} | {{ aoi_description }}</div>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <p>{{ report_summary }}</p>
      <div>
        <span class="pill">PNG / PDF / SVG</span>
        <span class="pill">300 DPI / 600 DPI</span>
        <span class="pill">Leaflet</span>
        <span class="pill">Plotly</span>
      </div>
    </section>

    <section>
      <h2>GeoWatch Quality Score</h2>
      <p><strong>{{ quality_score }}/100</strong> | {{ quality_status }}</p>
      <ul>
      {% for component in quality_components %}
        <li>{{ component.title }}: {{ component.status }} ({{ "%.1f"|format(component.score) }}/{{ component.weight }})</li>
      {% endfor %}
      </ul>
      <p class="caption">
      {% if quality_warnings %}
        {{ quality_warnings | join("; ") }}
      {% else %}
        No run-quality warnings were generated.
      {% endif %}
      </p>
    </section>

    <section>
      {{ interpretation_html | safe }}
    </section>

    <section>
      <h2>AOI Information</h2>
      {{ tables.aoi | safe }}
    </section>

    <section>
      <h2>Satellite Sources</h2>
      {{ tables.sources | safe }}
    </section>

    <section>
      <h2>Methodology</h2>
      <p>GeoWatch combines acquisition, processing, analytics, cartography, and reporting into one publication workflow.</p>
    </section>

    <section>
      <h2>Preprocessing</h2>
      <p>The demo publication uses projected synthetic scenes, canonical band naming, and consistent AOI bounds so every map renders cleanly.</p>
    </section>

    <section>
      <h2>Indices</h2>
      {{ tables.indices | safe }}
      <div id="index-chart" class="chart"></div>
    </section>

    <section>
      <h2>Change Detection</h2>
      {{ tables.change | safe }}
      <div id="change-chart" class="chart"></div>
    </section>

    <section>
      <h2>LULC</h2>
      {{ tables.lulc | safe }}
      <div id="lulc-chart" class="chart"></div>
    </section>

    <section>
      <h2>Statistics</h2>
      {{ tables.statistics | safe }}
    </section>

    <section>
      <h2>Interactive Map</h2>
      <div id="leaflet-map" class="map"></div>
    </section>

    <section>
      <h2>Recommendations</h2>
      <p>Use the 600 DPI map exports for print and the HTML report for interactive exploration. The exported tables capture the core publication metrics.</p>
    </section>

    <section>
      <h2>Limitations</h2>
      <p>This publication bundle is built from synthetic validation data, and the Plotly/Leaflet assets are loaded from CDNs.</p>
    </section>
  </main>
  <script>
    const charts = {{ charts | safe }};
    charts.forEach((chart) => {
      Plotly.newPlot(chart.div_id, JSON.parse(chart.data), JSON.parse(chart.layout), {responsive: true, displaylogo: false});
    });

    const leafletLayers = {{ leaflet_layers | safe }};
    const leafletMap = L.map('leaflet-map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(leafletMap);
    const layerMap = {};
    leafletLayers.forEach((layer) => {
      const overlay = L.imageOverlay(layer.path, layer.bounds, {opacity: 0.7});
      layerMap[layer.name] = overlay;
      overlay.addTo(leafletMap);
    });
    const layerControl = L.control.layers(null, layerMap).addTo(leafletMap);
    if (leafletLayers.length > 0) {
      leafletMap.fitBounds(leafletLayers[0].bounds);
    }
  </script>
</body>
</html>
"""
