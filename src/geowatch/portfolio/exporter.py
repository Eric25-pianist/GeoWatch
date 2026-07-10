"""Portfolio-ready export packaging for GeoWatch runs."""

from __future__ import annotations

import csv
import json
import shutil
import textwrap
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from pyproj import Geod
from reportlab.lib import colors
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
    TableStyle,
)

from geowatch import __version__
from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import AnalyticsReport, ClassificationResult
from geowatch.application.models import RunSpecification
from geowatch.cartography.themes import get_map_theme
from geowatch.processing.models import RasterLayer
from geowatch.reporting.interpretation import InterpretationReport
from geowatch.reporting.models import MapArtifact
from geowatch.utils.geometry import load_vector_geometry, reproject_geometry
from geowatch.validation.quality_score import QualityScoreReport

_SHOWCASE_MAPS: tuple[tuple[str, str], ...] = (
    ("before_after", "02_before_after_comparison.png"),
    ("change_detection", "03_change_detection_map.png"),
    ("lulc", "04_lulc_map.png"),
    ("ndvi_gain_loss", "05_ndvi_gain_loss_map.png"),
)

_METHODOLOGY_FLOW = (
    "Boundary -> Imagery -> QA Masking -> Indices -> Classification -> "
    "Change Detection -> Reports"
)


def export_portfolio_package(
    *,
    output_dir: Path,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    maps: Mapping[str, MapArtifact],
    sources: Sequence[SceneMetadata],
    downloads: Mapping[str, Path],
    interpretation: InterpretationReport | None = None,
    quality_report: QualityScoreReport | None = None,
) -> dict[str, Path]:
    """Create a portfolio-ready summary package for one GeoWatch run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    theme = get_map_theme(spec.outputs.map_theme)
    missing_items: list[str] = []
    copied_maps = _copy_showcase_maps(output_dir, maps, missing_items)
    dashboard_launcher = _write_dashboard_launcher(
        output_dir / "06_dashboard.html",
        downloads.get("dashboard") or downloads.get("html_report"),
        output_dir,
        missing_items,
    )
    statistics_path = _write_key_statistics_csv(
        output_dir / "08_key_statistics.csv",
        spec=spec,
        boundary_path=boundary_path,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        quality_report=quality_report,
    )
    metadata_path = _write_metadata_json(
        output_dir / "10_project_metadata.json",
        spec=spec,
        boundary_path=boundary_path,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics=analytics,
        sources=sources,
        quality_report=quality_report,
        downloads=downloads,
        copied_maps=copied_maps,
        missing_items=missing_items,
    )
    readme_path = _write_readme_snippet(
        output_dir / "09_github_readme_snippet.md",
        spec=spec,
        sources=sources,
        copied_maps=copied_maps,
        quality_report=quality_report,
        missing_items=missing_items,
    )
    infographic_path = _write_summary_infographic(
        output_dir / "01_summary_infographic.png",
        spec=spec,
        boundary_path=boundary_path,
        analytics=analytics,
        sources=sources,
        quality_report=quality_report,
        copied_maps=copied_maps,
        theme_name=theme.label,
    )
    short_pdf_path = _write_short_portfolio_pdf(
        output_dir / "07_short_portfolio_report.pdf",
        spec=spec,
        analytics=analytics,
        sources=sources,
        quality_report=quality_report,
        interpretation=interpretation,
        infographic=infographic_path,
        copied_maps=copied_maps,
        metadata_path=metadata_path,
    )
    outputs = {
        "summary_infographic": infographic_path,
        **copied_maps,
        "dashboard": dashboard_launcher,
        "short_pdf": short_pdf_path,
        "key_statistics": statistics_path,
        "readme_snippet": readme_path,
        "metadata_json": metadata_path,
    }
    logger.info("Generated GeoWatch portfolio exports at {}", output_dir)
    return outputs


def _copy_showcase_maps(
    output_dir: Path,
    maps: Mapping[str, MapArtifact],
    missing_items: list[str],
) -> dict[str, Path]:
    """Copy the best already-rendered maps into the portfolio folder."""
    copied: dict[str, Path] = {}
    for artifact_name, filename in _SHOWCASE_MAPS:
        artifact = maps.get(artifact_name)
        if artifact is None:
            missing_items.append(f"{artifact_name}: map artifact unavailable")
            continue
        source = artifact.files.get("png_300") or artifact.files.get("png_600")
        if source is None or not source.exists():
            missing_items.append(f"{artifact_name}: PNG export unavailable")
            continue
        destination = output_dir / filename
        shutil.copy2(source, destination)
        copied[artifact_name] = destination
    return copied


def _write_dashboard_launcher(
    output_path: Path,
    dashboard_path: Path | None,
    output_dir: Path,
    missing_items: list[str],
) -> Path:
    """Write a launcher that opens the real dashboard from the portfolio folder."""
    if dashboard_path is None or not dashboard_path.exists():
        missing_items.append("dashboard: interactive report unavailable")
        output_path.write_text(
            (
                "<!doctype html><html><body><p>GeoWatch dashboard is unavailable "
                "for this run.</p></body></html>"
            ),
            encoding="utf-8",
        )
        return output_path
    relative = dashboard_path.relative_to(output_dir.parent).as_posix()
    output_path.write_text(
        "\n".join(
            (
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                f'<meta http-equiv="refresh" content="0; url=../{relative}">',
                "<title>GeoWatch Portfolio Dashboard</title>",
                (
                    "<style>body{font:16px Arial;margin:40px;color:#17202a}"
                    "a{color:#275dad}</style>"
                ),
                "</head>",
                "<body>",
                "<h1>GeoWatch Dashboard</h1>",
                (
                    "<p>Opening the full interactive dashboard from the project "
                    "reports folder.</p>"
                ),
                f'<p><a href="../{relative}">Open dashboard</a></p>',
                "</body></html>",
            )
        ),
        encoding="utf-8",
    )
    return output_path


def _write_key_statistics_csv(
    output_path: Path,
    *,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    quality_report: QualityScoreReport | None,
) -> Path:
    """Write concise portfolio statistics for quick review and sharing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_area_km2 = _pixel_area_km2(scene_t2)
    rows: list[dict[str, object]] = [
        {
            "metric": "location",
            "value": spec.location.name,
            "units": "",
            "note": "GeoWatch project location",
        },
        {
            "metric": "time_period",
            "value": f"{spec.temporal.start_year}-{spec.temporal.end_year}",
            "units": "",
            "note": "Compared years",
        },
        {
            "metric": "aoi_area_km2",
            "value": round(_boundary_area_km2(boundary_path), 3),
            "units": "km2",
            "note": "Boundary geodesic area",
        },
        {
            "metric": "mission",
            "value": _mission_text(sources, scene_t1=None, scene_t2=scene_t2),
            "units": "",
            "note": "Primary mission used",
        },
    ]
    if quality_report is not None:
        rows.append(
            {
                "metric": "quality_score",
                "value": quality_report.rounded_score,
                "units": "score",
                "note": quality_report.overall_status,
            }
        )
    for name, result in analytics.index_results.items():
        rows.append(
            {
                "metric": f"{name}_mean_change",
                "value": round(result.statistics.difference.mean, 6),
                "units": "index",
                "note": "T2 minus T1 mean",
            }
        )
    if analytics.signed_change is not None:
        for class_name, count in analytics.signed_change.counts.items():
            rows.append(
                {
                    "metric": f"ndvi_{class_name.lower().replace(' ', '_')}_area",
                    "value": round(count * pixel_area_km2, 4),
                    "units": "km2",
                    "note": "Signed NDVI change area",
                }
            )
    classification = _preferred_classification(analytics)
    if classification is not None:
        for class_name, count in classification.counts.items():
            class_key = class_name.lower().replace("/", "_").replace(" ", "_")
            rows.append(
                {
                    "metric": f"lulc_{class_key}_area",
                    "value": round(count * pixel_area_km2, 4),
                    "units": "km2",
                    "note": "Endpoint LULC area",
                }
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value", "units", "note"))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_metadata_json(
    output_path: Path,
    *,
    spec: RunSpecification,
    boundary_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    quality_report: QualityScoreReport | None,
    downloads: Mapping[str, Path],
    copied_maps: Mapping[str, Path],
    missing_items: Sequence[str],
) -> Path:
    """Write machine-readable portfolio metadata."""
    payload: dict[str, object] = {
        "location": spec.location.name,
        "country": spec.location.country,
        "years": {
            "start": spec.temporal.start_year,
            "end": spec.temporal.end_year,
        },
        "sensor": _mission_text(sources, scene_t1=scene_t1, scene_t2=scene_t2),
        "crs": scene_t1.grid.crs,
        "aoi_area_km2": round(_boundary_area_km2(boundary_path), 3),
        "quality_score": None
        if quality_report is None
        else quality_report.rounded_score,
        "quality_status": None
        if quality_report is None
        else quality_report.overall_status,
        "map_theme": spec.outputs.map_theme,
        "generation_date": datetime.now(UTC).isoformat(),
        "geowatch_version": __version__,
        "main_statistics": _main_statistics_dict(analytics, scene_t2),
        "output_paths": {
            "portfolio_exports": str(output_path.parent),
            "copied_maps": {key: str(path) for key, path in copied_maps.items()},
            "reports": {
                key: str(path)
                for key, path in downloads.items()
                if path.suffix.lower() in {".html", ".pdf", ".md"}
            },
        },
        "scene_ids": [source.scene_id for source in sources],
        "boundary_source": spec.location.boundary_source,
        "missing_items": list(missing_items),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _write_readme_snippet(
    output_path: Path,
    *,
    spec: RunSpecification,
    sources: Sequence[SceneMetadata],
    copied_maps: Mapping[str, Path],
    quality_report: QualityScoreReport | None,
    missing_items: Sequence[str],
) -> Path:
    """Write a GitHub-friendly Markdown snippet for the project showcase."""
    lines = [
        f"# GeoWatch Portfolio Export - {spec.location.name}",
        "",
        f"**Location:** {spec.location.name}, {spec.location.country}",
        f"**Time Period:** {spec.temporal.start_year} to {spec.temporal.end_year}",
        f"**Sensor / Mission:** {_mission_text(sources, scene_t1=None, scene_t2=None)}",
        "",
        "## Summary",
        "",
        (
            "This GeoWatch portfolio package presents a terminal-generated "
            "remote-sensing change analysis with professional maps, an offline "
            "dashboard, compact statistics, and portfolio-ready documentation."
        ),
        "",
    ]
    if quality_report is not None:
        lines.extend(
            [
                "## Quality Score",
                "",
                (
                    f"- GeoWatch Quality Score: {quality_report.rounded_score}/"
                    f"{quality_report.max_score}"
                ),
                f"- Overall status: {quality_report.overall_status}",
                "",
            ]
        )
    lines.extend(["## Key Outputs", ""])
    for _key, path in copied_maps.items():
        label = (
            path.stem.replace("_", " ")
            .replace("02 ", "")
            .replace("03 ", "")
            .replace("04 ", "")
            .replace("05 ", "")
        )
        lines.append(f"- {label.title()}: `{path.name}`")
    lines.extend(
        [
            "- Interactive dashboard: `06_dashboard.html`",
            "- Short portfolio PDF: `07_short_portfolio_report.pdf`",
            "- Key statistics: `08_key_statistics.csv`",
            "- Metadata JSON: `10_project_metadata.json`",
            "",
            "## Example Images",
            "",
        ]
    )
    for map_key in ("before_after", "change_detection", "lulc", "ndvi_gain_loss"):
        image_path = copied_maps.get(map_key)
        if image_path is not None:
            lines.extend(
                [
                    f"![{map_key.replace('_', ' ').title()}](./{image_path.name})",
                    "",
                ]
            )
    lines.extend(
        [
            "## Methods",
            "",
            "- Boundary validation and confirmation",
            "- Surface-reflectance imagery selection and QA masking",
            "- Spectral indices and change detection",
            "- LULC classification",
            "- Portfolio-ready cartography and reporting",
            "",
            "## Data Sources",
            "",
            f"- Mission: {_mission_text(sources, scene_t1=None, scene_t2=None)}",
            f"- Scene count: {len({source.scene_id for source in sources})}",
            "- Boundary: confirmed administrative geometry used as final clip mask",
            "",
            "## Limitations",
            "",
            "- Unsupervised LULC is exploratory unless validated with reference data.",
            (
                "- Spectral change should be interpreted with seasonal, "
                "atmospheric, and mixed-pixel context."
            ),
            "",
            "## Reproduce with GeoWatch",
            "",
            "```powershell",
            (
                "geowatch process "
                f"outputs/{spec.location.name.replace(' ', '_')}/project.yaml"
            ),
            "```",
            "",
        ]
    )
    if missing_items:
        lines.extend(["## Missing Optional Items", ""])
        lines.extend(f"- {item}" for item in missing_items)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_summary_infographic(
    output_path: Path,
    *,
    spec: RunSpecification,
    boundary_path: Path,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    quality_report: QualityScoreReport | None,
    copied_maps: Mapping[str, Path],
    theme_name: str,
) -> Path:
    """Compose a one-page infographic PNG from existing map outputs and summaries."""
    width, height = 1600, 1200
    canvas = PILImage.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(48, bold=True)
    heading_font = _load_font(28, bold=True)
    body_font = _load_font(22)
    small_font = _load_font(18)

    draw.rounded_rectangle((40, 36, 1560, 210), radius=28, fill="#16324f")
    draw.text((72, 64), spec.location.name, font=title_font, fill="white")
    draw.text(
        (72, 126),
        (
            f"{spec.temporal.start_year}-{spec.temporal.end_year} | "
            f"{_mission_text(sources, scene_t1=None, scene_t2=None)} | "
            f"{theme_name}"
        ),
        font=body_font,
        fill="#d8e6f2",
    )
    draw.text(
        (72, 164),
        (
            f"AOI area: {_boundary_area_km2(boundary_path):,.2f} km2 | "
            f"Projection: "
            f"{spec.outputs.target_crs if spec.outputs.target_crs else 'Auto'}"
        ),
        font=small_font,
        fill="#d8e6f2",
    )

    left_panel = (40, 240, 610, 1160)
    right_panel = (640, 240, 1560, 1160)
    draw.rounded_rectangle(left_panel, radius=24, fill="white", outline="#d3dde7")
    draw.rounded_rectangle(right_panel, radius=24, fill="white", outline="#d3dde7")

    draw.text((68, 270), "Key statistics", font=heading_font, fill="#16324f")
    stats_lines = _infographic_stat_lines(analytics, quality_report)
    current_y = 322
    for line in stats_lines:
        wrapped = textwrap.wrap(line, width=34)
        for part in wrapped:
            draw.text((72, current_y), f"- {part}", font=body_font, fill="#243b53")
            current_y += 34
        current_y += 10

    draw.text(
        (68, min(current_y + 18, 760)), "Methodology", font=heading_font, fill="#16324f"
    )
    method_y = min(current_y + 70, 810)
    for part in textwrap.wrap(_METHODOLOGY_FLOW, width=32):
        draw.text((72, method_y), part, font=body_font, fill="#2f4858")
        method_y += 34

    limit_text = (
        "Unsupervised LULC is exploratory unless validated with reference data."
    )
    draw.rounded_rectangle(
        (68, 930, 582, 1070), radius=18, fill="#fff6e8", outline="#f0bf63"
    )
    draw.text((86, 952), "Limitation note", font=heading_font, fill="#8a5a00")
    limit_y = 997
    for part in textwrap.wrap(limit_text, width=33):
        draw.text((88, limit_y), part, font=small_font, fill="#7a5600")
        limit_y += 26

    draw.text((670, 270), "Showcase maps", font=heading_font, fill="#16324f")
    _paste_showcase_thumbnails(canvas, copied_maps)

    note_y = 988
    draw.text(
        (668, note_y),
        "Data source and projection note",
        font=heading_font,
        fill="#16324f",
    )
    note_y += 46
    boundary_source = (
        spec.location.boundary_source or "User-confirmed administrative boundary"
    )
    notes = [
        f"Mission: {_mission_text(sources, scene_t1=None, scene_t2=None)}",
        f"Boundary source: {boundary_source}",
        "Projection and raster grid details are documented in the GeoWatch reports.",
        "GeoWatch Project attribution",
    ]
    for note in notes:
        for line in textwrap.wrap(note, width=78):
            draw.text((670, note_y), line, font=small_font, fill="#3d5368")
            note_y += 25
        note_y += 3

    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def _paste_showcase_thumbnails(
    canvas: PILImage.Image,
    copied_maps: Mapping[str, Path],
) -> None:
    """Paste up to four map thumbnails into a 2x2 showcase grid."""
    slots = (
        ("before_after", (668, 322, 1088, 630)),
        ("change_detection", (1110, 322, 1530, 630)),
        ("lulc", (668, 652, 1088, 960)),
        ("ndvi_gain_loss", (1110, 652, 1530, 960)),
    )
    for key, bounds in slots:
        path = copied_maps.get(key)
        if path is None or not path.exists():
            continue
        with PILImage.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((bounds[2] - bounds[0], bounds[3] - bounds[1]))
            x = bounds[0] + ((bounds[2] - bounds[0] - thumb.width) // 2)
            y = bounds[1] + ((bounds[3] - bounds[1] - thumb.height) // 2)
            canvas.paste(thumb, (x, y))


def _write_short_portfolio_pdf(
    output_path: Path,
    *,
    spec: RunSpecification,
    analytics: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    quality_report: QualityScoreReport | None,
    interpretation: InterpretationReport | None,
    infographic: Path,
    copied_maps: Mapping[str, Path],
    metadata_path: Path,
) -> Path:
    """Write a compact visual PDF brief for GitHub, CV, and presentation sharing."""
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph(f"{spec.location.name} GeoWatch Portfolio Brief", styles["Title"]),
        Paragraph(
            (
                f"{spec.temporal.start_year} to {spec.temporal.end_year} | "
                f"{_mission_text(sources, scene_t1=None, scene_t2=None)}"
            ),
            styles["Heading2"],
        ),
        Spacer(1, 0.15 * inch),
        Image(str(infographic), width=7.2 * inch, height=5.4 * inch),
        Spacer(1, 0.12 * inch),
        Paragraph(
            (
                "This short brief summarizes the strongest visual and "
                "statistical outputs from the GeoWatch run."
            ),
            styles["BodyText"],
        ),
    ]
    if quality_report is not None:
        story.extend(
            [
                Spacer(1, 0.1 * inch),
                Paragraph(
                    (
                        "GeoWatch Quality Score: "
                        f"{quality_report.rounded_score}/{quality_report.max_score} "
                        f"({quality_report.overall_status})"
                    ),
                    styles["BodyText"],
                ),
            ]
        )
    story.append(PageBreak())
    story.append(Paragraph("Key maps and results", styles["Heading1"]))
    for key in ("before_after", "change_detection", "lulc", "ndvi_gain_loss"):
        path = copied_maps.get(key)
        if path is None or not path.exists():
            continue
        story.extend(
            [
                Spacer(1, 0.08 * inch),
                Paragraph(key.replace("_", " ").title(), styles["Heading2"]),
                Image(str(path), width=6.9 * inch, height=4.2 * inch),
            ]
        )
    story.append(PageBreak())
    story.append(Paragraph("Statistics and interpretation", styles["Heading1"]))
    statistics_table = Table(
        _pdf_statistics_rows(analytics, quality_report), repeatRows=1
    )
    statistics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d4df")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f4f7fa")],
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(statistics_table)
    story.append(Spacer(1, 0.16 * inch))
    interpretation_text = _interpretation_excerpt(interpretation)
    for paragraph in interpretation_text:
        story.append(Paragraph(paragraph, styles["BodyText"]))
        story.append(Spacer(1, 0.08 * inch))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Methods, sources, and limitations", styles["Heading1"]))
    for paragraph in (
        (
            "Workflow: Boundary -> Imagery -> QA Masking -> Indices -> "
            "Classification -> Change Detection -> Reports."
        ),
        (
            "Mission and provider information are documented in the GeoWatch "
            f"reports and metadata file `{metadata_path.name}`."
        ),
        (
            "Unsupervised LULC is exploratory unless validated with independent "
            "reference data."
        ),
    ):
        story.append(Paragraph(paragraph, styles["BodyText"]))
        story.append(Spacer(1, 0.08 * inch))
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        author="GeoWatch Project",
        title=f"{spec.location.name} Portfolio Brief",
    )
    doc.build(story)
    return output_path


def _pdf_statistics_rows(
    analytics: AnalyticsReport,
    quality_report: QualityScoreReport | None,
) -> list[list[str]]:
    """Build a compact PDF table of high-value statistics."""
    rows = [["Metric", "Value", "Note"]]
    if quality_report is not None:
        rows.append(
            [
                "GeoWatch Quality Score",
                f"{quality_report.rounded_score}/{quality_report.max_score}",
                quality_report.overall_status,
            ]
        )
    primary_change = next(iter(analytics.change_results.values()), None)
    if primary_change is not None:
        rows.append(
            [
                f"{primary_change.method.upper()} changed fraction",
                (
                    f"{primary_change.threshold.change_fraction:.1%}"
                    if primary_change.threshold is not None
                    else f"{primary_change.statistics.mean:.4f}"
                ),
                "Primary change result",
            ]
        )
    if analytics.signed_change is not None:
        for key, count in analytics.signed_change.counts.items():
            rows.append([f"NDVI {key}", f"{count:,} pixels", "Signed NDVI change"])
    return rows


def _interpretation_excerpt(
    interpretation: InterpretationReport | None,
) -> list[str]:
    """Extract concise paragraphs for the short portfolio PDF."""
    if interpretation is None or not interpretation.sections:
        return [
            "Interpretation was not available for this portfolio export.",
            "Use the full GeoWatch reports for complete technical context.",
        ]
    paragraphs: list[str] = []
    for section in interpretation.sections[:3]:
        paragraphs.extend(section.paragraphs[:1])
    return paragraphs[:4]


def _boundary_area_km2(boundary_path: Path) -> float:
    """Calculate the geodesic area of the approved AOI boundary."""
    loaded = load_vector_geometry(boundary_path)
    geometry_wgs84 = reproject_geometry(loaded.geometry, loaded.crs, "EPSG:4326")
    area_m2, _ = Geod(ellps="WGS84").geometry_area_perimeter(geometry_wgs84)
    return abs(float(area_m2)) / 1_000_000.0


def _pixel_area_km2(scene: RasterLayer) -> float:
    """Approximate the projected pixel area in square kilometres."""
    transform = scene.grid.transform
    width = abs(float(transform[0]))
    height = abs(float(transform[4]))
    return (width * height) / 1_000_000.0


def _mission_text(
    sources: Sequence[SceneMetadata],
    *,
    scene_t1: RasterLayer | None,
    scene_t2: RasterLayer | None,
) -> str:
    """Build a readable sensor or mission label."""
    dataset_labels = {
        "sentinel-2-l2a": "Sentinel-2 L2A",
        "landsat-5-c2-l2": "Landsat 5 Collection 2 L2",
        "landsat-7-c2-l2": "Landsat 7 Collection 2 L2",
        "landsat-8-c2-l2": "Landsat 8 Collection 2 L2",
        "landsat-9-c2-l2": "Landsat 9 Collection 2 L2",
    }
    datasets = sorted({source.dataset for source in sources if source.dataset})
    if datasets:
        return ", ".join(dataset_labels.get(item, item) for item in datasets)
    for scene in (scene_t1, scene_t2):
        if scene is None:
            continue
        dataset = str(scene.metadata.get("dataset", "")).strip().lower()
        if dataset:
            return dataset_labels.get(dataset, dataset)
    return "Mission documented in reports"


def _preferred_classification(
    analytics: AnalyticsReport,
) -> ClassificationResult | None:
    """Return the preferred endpoint classification for portfolio summaries."""
    if "lulc_t2" in analytics.classification_results:
        return analytics.classification_results["lulc_t2"]
    if analytics.classification_results:
        return next(reversed(analytics.classification_results.values()))
    return None


def _main_statistics_dict(
    analytics: AnalyticsReport,
    scene_t2: RasterLayer,
) -> dict[str, object]:
    """Collect human-friendly primary statistics for metadata export."""
    stats: dict[str, object] = {}
    pixel_area_km2 = _pixel_area_km2(scene_t2)
    for name, result in analytics.index_results.items():
        stats[f"{name}_difference_mean"] = round(result.statistics.difference.mean, 6)
    if analytics.signed_change is not None:
        stats["ndvi_signed_counts"] = analytics.signed_change.counts
        stats["ndvi_signed_area_km2"] = {
            key: round(value * pixel_area_km2, 4)
            for key, value in analytics.signed_change.counts.items()
        }
    classification = _preferred_classification(analytics)
    if classification is not None:
        stats["lulc_counts"] = classification.counts
        stats["lulc_area_km2"] = {
            key: round(value * pixel_area_km2, 4)
            for key, value in classification.counts.items()
        }
    if analytics.change_results:
        primary = next(iter(analytics.change_results.values()))
        stats["primary_change_method"] = primary.method
        stats["primary_change_mean"] = round(primary.statistics.mean, 6)
        if primary.threshold is not None:
            stats["primary_change_fraction"] = round(
                primary.threshold.change_fraction,
                6,
            )
    return stats


def _infographic_stat_lines(
    analytics: AnalyticsReport,
    quality_report: QualityScoreReport | None,
) -> list[str]:
    """Build concise summary lines for the infographic."""
    lines: list[str] = []
    if quality_report is not None:
        lines.append(
            "GeoWatch Quality Score: "
            f"{quality_report.rounded_score}/{quality_report.max_score} "
            f"({quality_report.overall_status})"
        )
    primary = next(iter(analytics.change_results.values()), None)
    if primary is not None:
        if primary.threshold is not None:
            lines.append(
                f"Primary change ({primary.method.upper()}): "
                f"{primary.threshold.change_fraction:.1%} of AOI flagged as change"
            )
        else:
            lines.append(
                f"Primary change ({primary.method.upper()}): "
                f"mean score {primary.statistics.mean:.4f}"
            )
    if analytics.signed_change is not None:
        counts = analytics.signed_change.counts
        lines.append(
            "NDVI gain/loss/no-change: "
            f"{counts.get('Gain', 0):,} / {counts.get('Loss', 0):,} / "
            f"{counts.get('No change', 0):,} pixels"
        )
    classification = _preferred_classification(analytics)
    if classification is not None:
        for class_name in ("Urban", "Vegetation", "Water"):
            if class_name in classification.counts:
                lines.append(
                    f"{class_name} area: {classification.counts[class_name]:,} pixels"
                )
    if not lines:
        lines.append("Portfolio statistics are limited for this run.")
    return lines


def _load_font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable local font with a safe fallback."""
    candidates = (
        "C:/Windows/Fonts/DejaVuSans-Bold.ttf"
        if bold
        else "C:/Windows/Fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
