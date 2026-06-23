"""Unit tests for rule-based GeoWatch interpretation text."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

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
from geowatch.application.models import LocationSpec, RunSpecification, TemporalSpec
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.reporting.interpretation import (
    generate_interpretation,
    render_interpretation_html,
    write_interpretation,
)


def test_interpretation_is_data_driven_and_cautious(tmp_path: Path) -> None:
    """Use measured values without unsupported accuracy claims."""
    inputs = _inputs(tmp_path)

    report = generate_interpretation(**inputs)
    markdown = report.as_markdown()

    assert "Sample City" in markdown
    assert "2018-2020" in markdown
    assert "Sentinel-2 Level-2A" in markdown
    assert "NDVI changed from a mean of 0.2500" in markdown
    assert "gain covers" in markdown
    assert "NDBI mean change is +0.0800" in markdown
    assert "GeoWatch Quality Score for this run is" in markdown
    assert "No independent reference samples" in markdown
    assert "high accuracy" not in markdown.lower()
    assert "99%" not in markdown


def test_interpretation_handles_missing_optional_statistics(tmp_path: Path) -> None:
    """Missing optional outputs should produce useful caveats instead of crashing."""
    inputs = _inputs(tmp_path)
    analytics = inputs["analytics"]
    minimal = AnalyticsReport(
        phase=4,
        messages=analytics.messages,
        index_results={},
        change_results={},
        classification_results={},
        transition_result=analytics.transition_result,
        accuracy={},
        artifacts={},
        signed_change=None,
    )
    inputs["analytics"] = minimal

    report = generate_interpretation(**inputs)
    markdown = report.as_markdown()

    assert "NDVI summary was available" in markdown
    assert "Water interpretation is limited" in markdown
    assert "Built-up interpretation is limited" in markdown
    assert "No independent reference samples" in markdown


def test_write_interpretation_and_html_rendering(tmp_path: Path) -> None:
    """Interpretation should export Markdown and render embeddable HTML."""
    inputs = _inputs(tmp_path)

    path = write_interpretation(tmp_path / "reports" / "interpretation.md", **inputs)
    html = render_interpretation_html(generate_interpretation(**inputs))

    assert path.exists()
    assert "GeoWatch Interpretation" in path.read_text(encoding="utf-8")
    assert "interpretation-card" in html
    assert "Executive interpretation" in html


def _inputs(tmp_path: Path) -> dict[str, object]:
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"properties":{},"geometry":{"type":"Polygon","coordinates":'
        "[[[0,0],[0.02,0],[0.02,0.02],[0,0.02],[0,0]]]}}]}",
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
    data_t1 = np.ones((6, 2, 2), dtype=np.float32) * 0.2
    data_t2 = np.ones((6, 2, 2), dtype=np.float32) * 0.3
    scene_t1 = RasterLayer(name="t1", data=data_t1, grid=grid)
    scene_t2 = RasterLayer(name="t2", data=data_t2, grid=grid)
    analytics = _analytics()
    spec = RunSpecification.model_construct(
        location=LocationSpec(
            name="Sample City",
            country="Sample Country",
            boundary_path=boundary,
            boundary_source="Test boundary",
        ),
        temporal=TemporalSpec(start_year=2018, end_year=2020),
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
    return {
        "spec": spec,
        "boundary_path": boundary,
        "scene_t1": scene_t1,
        "scene_t2": scene_t2,
        "analytics": analytics,
        "sources": sources,
    }


def _analytics() -> AnalyticsReport:
    ndvi_stats = IndexStatistics(
        t1=MapStatistics("ndvi_t1", 4, 4, 0.1, 0.4, 0.25, 0.05),
        t2=MapStatistics("ndvi_t2", 4, 4, 0.2, 0.5, 0.35, 0.05),
        difference=MapStatistics("ndvi_diff", 4, 4, -0.1, 0.3, 0.1, 0.05),
    )
    ndbi_stats = IndexStatistics(
        t1=MapStatistics("ndbi_t1", 4, 4, -0.2, 0.2, 0.05, 0.04),
        t2=MapStatistics("ndbi_t2", 4, 4, -0.1, 0.3, 0.13, 0.04),
        difference=MapStatistics("ndbi_diff", 4, 4, -0.1, 0.2, 0.08, 0.04),
    )
    ndvi = SpectralIndexResult(
        name="ndvi",
        t1=np.zeros((2, 2), dtype=np.float32),
        t2=np.zeros((2, 2), dtype=np.float32),
        difference=np.zeros((2, 2), dtype=np.float32),
        statistics=ndvi_stats,
    )
    ndbi = SpectralIndexResult(
        name="ndbi",
        t1=np.zeros((2, 2), dtype=np.float32),
        t2=np.zeros((2, 2), dtype=np.float32),
        difference=np.zeros((2, 2), dtype=np.float32),
        statistics=ndbi_stats,
    )
    threshold = ThresholdResult(
        method="otsu",
        threshold=0.1,
        mask=np.array([[False, True], [False, False]]),
        score_statistics=ndvi_stats.difference,
        changed_pixels=1,
        change_fraction=0.25,
    )
    change = ChangeDetectionResult(
        method="index_differencing",
        score=np.zeros((2, 2), dtype=np.float32),
        statistics=ndvi_stats.difference,
        threshold=threshold,
    )
    class_names = ("Water", "Urban", "Vegetation", "Bare Soil")
    lulc_t1 = ClassificationResult(
        method="kmeans",
        labels=np.zeros((2, 2), dtype=np.int64),
        class_names=class_names,
        counts={"Water": 1, "Urban": 1, "Vegetation": 1, "Bare Soil": 1},
        model_name="KMeans",
        feature_names=("ndvi", "ndbi"),
    )
    lulc_t2 = ClassificationResult(
        method="kmeans",
        labels=np.ones((2, 2), dtype=np.int64),
        class_names=class_names,
        counts={"Water": 1, "Urban": 2, "Vegetation": 1, "Bare Soil": 0},
        model_name="KMeans",
        feature_names=("ndvi", "ndbi"),
    )
    transition = TransitionResult(
        class_names=class_names,
        transition_matrix=np.eye(4, dtype=np.int64),
        change_matrix=np.zeros((4, 4), dtype=np.int64),
        changed_pixels=1,
    )
    signed = SignedChangeResult(
        name="ndvi_gain_loss",
        labels=np.array([[0, 1], [1, 2]], dtype=np.uint8),
        class_names=("Loss", "No change", "Gain"),
        threshold=0.1,
        counts={"Loss": 1, "No change": 2, "Gain": 1},
    )
    return AnalyticsReport(
        phase=4,
        messages=("Complete",),
        index_results={"ndvi": ndvi, "ndbi": ndbi},
        change_results={"index_differencing": change},
        classification_results={"lulc_t1": lulc_t1, "lulc_t2": lulc_t2},
        transition_result=transition,
        accuracy={},
        artifacts={},
        signed_change=signed,
    )
