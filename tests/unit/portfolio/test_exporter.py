"""Unit tests for GeoWatch portfolio export packaging."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import (
    AnalyticsReport,
    ChangeDetectionResult,
    ClassificationResult,
    IndexStatistics,
    MapStatistics,
    SignedChangeResult,
    SpectralIndexResult,
    ThresholdResult,
    TransitionResult,
)
from geowatch.application.models import (
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.portfolio.exporter import export_portfolio_package
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.reporting.models import MapArtifact
from geowatch.validation.quality_score import calculate_quality_score


def test_portfolio_exporter_generates_shareable_package_with_missing_items(
    tmp_path: Path,
) -> None:
    """The portfolio package should succeed even when some optional maps are missing."""
    inputs = _portfolio_inputs(tmp_path)
    quality = calculate_quality_score(
        spec=inputs["spec"],
        boundary_path=inputs["boundary"],
        scene_t1=inputs["scene_t1"],
        scene_t2=inputs["scene_t2"],
        analytics=inputs["analytics"],
        sources=inputs["sources"],
        maps=inputs["maps"],
        downloads=inputs["downloads"],
    )

    exports = export_portfolio_package(
        output_dir=tmp_path / "portfolio_exports",
        spec=inputs["spec"],
        boundary_path=inputs["boundary"],
        scene_t1=inputs["scene_t1"],
        scene_t2=inputs["scene_t2"],
        analytics=inputs["analytics"],
        maps=inputs["maps"],
        sources=inputs["sources"],
        downloads=inputs["downloads"],
        quality_report=quality,
    )

    assert exports["summary_infographic"].exists()
    assert exports["short_pdf"].exists()
    assert exports["readme_snippet"].exists()
    assert exports["metadata_json"].exists()
    assert exports["dashboard"].exists()
    metadata = exports["metadata_json"].read_text(encoding="utf-8")
    readme = exports["readme_snippet"].read_text(encoding="utf-8")
    assert "missing_items" in metadata
    assert "lulc: map artifact unavailable" in metadata
    assert "Missing Optional Items" in readme
    assert "./02_before_after_comparison.png" in readme


def _portfolio_inputs(tmp_path: Path) -> dict[str, object]:
    """Build compact portfolio inputs with intentionally missing optional maps."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"Polygon","coordinates":[[[0,0],[0.02,0],[0.02,0.02],[0,0.02],[0,0]]]}}]}',
        encoding="utf-8",
    )
    grid = RasterGrid(
        crs="EPSG:3857",
        transform=(1000.0, 0.0, 0.0, 0.0, -1000.0, 2000.0),
        width=2,
        height=2,
        band_names=("blue", "green", "red", "nir", "swir1", "swir2"),
        nodata=np.nan,
    )
    base = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    data_t1 = np.stack([base + (index * 0.02) for index in range(6)])
    data_t2 = np.clip(data_t1 + 0.05, 0.0, 1.0)
    scene_t1 = RasterLayer(name="t1", data=data_t1, grid=grid)
    scene_t2 = RasterLayer(name="t2", data=data_t2, grid=grid)
    scene_t1.metadata["dataset"] = "sentinel-2-l2a"
    scene_t2.metadata["dataset"] = "sentinel-2-l2a"
    statistics = MapStatistics("ndvi", 4, 4, -0.1, 0.7, 0.3, 0.2)
    difference_statistics = MapStatistics("ndvi_difference", 4, 4, -0.2, 0.3, 0.05, 0.1)
    index = SpectralIndexResult(
        name="ndvi",
        t1=base,
        t2=base + 0.05,
        difference=np.full((2, 2), 0.05, dtype=np.float32),
        statistics=IndexStatistics(statistics, statistics, difference_statistics),
    )
    threshold = ThresholdResult(
        method="otsu",
        threshold=0.1,
        mask=np.array([[False, True], [False, False]]),
        score_statistics=difference_statistics,
        changed_pixels=1,
        change_fraction=0.25,
    )
    change = ChangeDetectionResult(
        method="mad",
        score=np.abs(index.difference),
        statistics=difference_statistics,
        threshold=threshold,
    )
    class_names = ("Water", "Urban", "Vegetation")
    classification = ClassificationResult(
        method="kmeans",
        labels=np.array([[0, 1], [2, 2]], dtype=np.int64),
        class_names=class_names,
        counts={"Water": 1, "Urban": 1, "Vegetation": 2},
        model_name="KMeans",
        feature_names=("ndvi",),
    )
    transition = TransitionResult(
        class_names=class_names,
        transition_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]]),
        change_matrix=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        changed_pixels=1,
    )
    signed = SignedChangeResult(
        name="ndvi_gain_loss",
        labels=np.array([[0, 1], [1, 2]], dtype=np.uint8),
        class_names=("Loss", "No change", "Gain"),
        threshold=0.1,
        counts={"Loss": 1, "No change": 2, "Gain": 1},
    )
    analytics = AnalyticsReport(
        phase=4,
        messages=("Complete",),
        index_results={"ndvi": index},
        change_results={"mad": change},
        classification_results={"lulc_t2": classification},
        transition_result=transition,
        accuracy={},
        artifacts={},
        signed_change=signed,
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Sample City",
            country="Sample Country",
            boundary_path=boundary,
            boundary_source="Test boundary",
        ),
        temporal=TemporalSpec(start_year=2018, end_year=2020),
        outputs=OutputSpec(root=tmp_path, map_theme="government"),
    )
    sources = (
        SceneMetadata(
            scene_id="S2-SAMPLE-2018",
            provider="planetary-computer",
            dataset="sentinel-2-l2a",
            acquired_at=datetime(2018, 7, 1, tzinfo=UTC),
            cloud_cover=5.0,
        ),
        SceneMetadata(
            scene_id="S2-SAMPLE-2020",
            provider="planetary-computer",
            dataset="sentinel-2-l2a",
            acquired_at=datetime(2020, 7, 2, tzinfo=UTC),
            cloud_cover=7.0,
        ),
    )
    before_after = tmp_path / "maps" / "before_after.png"
    change_map = tmp_path / "maps" / "change.png"
    before_after.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (700, 420), "#d9e3e8").save(before_after)
    Image.new("RGB", (700, 420), "#e7d2c6").save(change_map)
    maps = {
        "before_after": MapArtifact(
            name="before_after",
            title="Before / After Comparison",
            description="Matched endpoint imagery.",
            files={"png_300": before_after},
            statistics={},
        ),
        "change_detection": MapArtifact(
            name="change_detection",
            title="Change Detection",
            description="Primary change map.",
            files={"png_300": change_map},
            statistics={},
        ),
    }
    dashboard = tmp_path / "reports" / "dashboard.html"
    report = tmp_path / "reports" / "report.html"
    pdf = tmp_path / "reports" / "report.pdf"
    interpretation = tmp_path / "reports" / "interpretation.md"
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.write_text("<html><body>Dashboard</body></html>", encoding="utf-8")
    report.write_text("<html><body>Report</body></html>", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4\n% minimal\n")
    interpretation.write_text("# Interpretation\n", encoding="utf-8")
    downloads = {
        "dashboard": dashboard,
        "html_report": report,
        "pdf_report": pdf,
        "interpretation": interpretation,
    }
    return {
        "boundary": boundary,
        "scene_t1": scene_t1,
        "scene_t2": scene_t2,
        "analytics": analytics,
        "spec": spec,
        "sources": sources,
        "maps": maps,
        "downloads": downloads,
    }
