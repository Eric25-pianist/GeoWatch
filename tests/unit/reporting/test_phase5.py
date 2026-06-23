"""Unit tests for the Phase 5 publication workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geowatch.config.loader import load_config
from geowatch.reporting import (
    analyze_hotspots,
    build_demo_publication_inputs,
    build_phase5_publication,
    write_build_report,
    write_phase_report,
)
from geowatch.reporting.cartography import (
    _cartographic_class_names,
    _scene_date_text,
    _scene_display_label,
)


def test_demo_hotspot_analysis() -> None:
    """Hotspot analysis should highlight concentrated change."""
    score = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 12.0, 12.0, 12.0, 0.0],
            [0.0, 12.0, 12.0, 12.0, 0.0],
            [0.0, 12.0, 12.0, 12.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    hotspot = analyze_hotspots(score, window_size=3)

    assert hotspot.gi_star.shape == score.shape
    assert hotspot.hotspot_mask[2, 2]
    assert hotspot.statistics.valid_pixels == 25


def test_exploratory_lulc_uses_cautious_bright_surface_label() -> None:
    """Unsupervised maps must not claim snow without validated evidence."""
    class_names = ("Water", "Urban", "Snow/Ice")

    assert _cartographic_class_names(class_names, exploratory=True) == (
        "Water",
        "Urban",
        "Bright Surface / Uncertain",
    )
    assert _cartographic_class_names(class_names, exploratory=False) == class_names


def test_scene_display_label_formats_processed_scene_metadata(
    sample_config_path: Path,
) -> None:
    """Mission identifiers and repeated source dates should be normalized."""
    config = load_config(sample_config_path)
    scene = build_demo_publication_inputs(config).scene_t1
    scene.metadata["dataset"] = "landsat-8-c2-l2"
    scene.metadata["source_dates"] = [
        "2018-05-03T10:20:00Z",
        "2018-05-03T10:20:00Z",
        "2018-06-12T10:20:00Z",
        "not-a-date",
    ]

    assert _scene_display_label(scene) == (
        "Landsat 8 Collection 2 L2 | 3 May 2018 to 12 June 2018"
    )
    assert _scene_date_text(scene) == "3 May 2018 to 12 June 2018"


def test_phase5_publication_bundle_and_reports(
    tmp_path: Path,
    sample_config_path: Path,
) -> None:
    """The publication bundle should generate the full set of outputs."""
    config = load_config(sample_config_path)
    config.outputs.root = tmp_path / "outputs"
    config.outputs.maps = tmp_path / "outputs" / "maps"
    config.outputs.reports = tmp_path / "outputs" / "reports"
    config.outputs.statistics = tmp_path / "outputs" / "statistics"
    config.outputs.manifests = tmp_path / "outputs" / "manifests"
    config.outputs.exports = tmp_path / "outputs" / "exports"
    config.outputs.map_theme = "dark"

    inputs = build_demo_publication_inputs(config)
    assert inputs.scene_t1.data.shape == inputs.scene_t2.data.shape
    assert inputs.sources

    bundle = build_phase5_publication(config)
    validation_summary = {
        "ruff": "passed",
        "mypy": "passed",
        "pytest": "passed",
        "cli": "passed",
        "imports": "passed",
        "config": "passed",
    }
    build_report = write_build_report(
        bundle,
        validation_summary,
        tmp_path / "BUILD_REPORT.md",
    )
    phase_report = write_phase_report(
        bundle,
        validation_summary,
        tmp_path / "PHASE_REPORT.md",
    )

    assert bundle.html_report.exists()
    assert bundle.pdf_report.exists()
    assert bundle.dashboard.exists()
    assert bundle.interpretation.exists()
    assert bundle.portfolio_exports["summary_infographic"].exists()
    assert bundle.portfolio_exports["short_pdf"].exists()
    assert bundle.portfolio_exports["readme_snippet"].exists()
    assert (tmp_path / "outputs" / "validation" / "quality_score.json").exists()
    dashboard_text = bundle.dashboard.read_text(encoding="utf-8")
    assert "data-compare" in dashboard_text
    assert "Analyst Interpretation" in dashboard_text
    assert "GeoWatch Quality Score" in bundle.html_report.read_text(encoding="utf-8")
    assert "supervised accuracy assessment object is present" in dashboard_text
    assert bundle.maps["ndvi"].files["png_300"].exists()
    assert bundle.maps["ndvi"].files["jpeg_300"].exists()
    assert bundle.maps["ndvi"].metadata["map_theme"] == "dark"
    assert bundle.maps["before_after"].metadata["map_theme_label"] == "Dark Dashboard"
    assert bundle.maps["lulc"].files["overlay_png"].exists()
    assert bundle.exports["csv"].exists()
    assert bundle.exports["xlsx"].exists()
    assert "Executive Summary" in bundle.html_report.read_text(encoding="utf-8")
    assert "## Vegetation interpretation" in bundle.interpretation.read_text(
        encoding="utf-8"
    )
    assert build_report.exists()
    assert phase_report.exists()
