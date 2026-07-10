"""Unit tests for the offline interactive publication dashboard."""

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
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.reporting.cartography import _save_slider_images
from geowatch.reporting.dashboard import write_dashboard
from geowatch.reporting.models import MapArtifact


def test_dashboard_contains_slider_statistics_and_portable_links(
    tmp_path: Path,
) -> None:
    """A complete run should produce the full offline dashboard experience."""
    inputs = _dashboard_inputs(tmp_path)
    slider_files = _save_slider_images(
        inputs["scene_t1"], inputs["scene_t2"], tmp_path / "maps" / "before_after"
    )
    map_image = tmp_path / "maps" / "before_after" / "before_after_300dpi.png"
    Image.new("RGB", (800, 480), "#d9e3e8").save(map_image)
    maps = {
        "before_after": MapArtifact(
            name="before_after",
            title="Before / After Comparison",
            description="Matched endpoint imagery.",
            files={"png_300": map_image, **slider_files},
            statistics={},
        )
    }

    dashboard = write_dashboard(
        tmp_path / "reports" / "dashboard.html",
        spec=inputs["spec"],
        boundary_path=inputs["boundary"],
        scene_t1=inputs["scene_t1"],
        scene_t2=inputs["scene_t2"],
        analytics=inputs["analytics"],
        maps=maps,
        sources=inputs["sources"],
        downloads={"summary_csv": inputs["download"]},
    )

    document = dashboard.read_text(encoding="utf-8")
    assert "data-compare" in document
    assert "Before &middot; 2018" in document
    assert "Sentinel-2 L2A" in document
    assert "LULC transition matrix" in document
    assert "Unsupervised K-Means" in document
    assert "Analyst Interpretation" in document
    assert "GeoWatch Quality Score" in document
    assert "No independent reference samples" in document
    assert "NDVI changed from a mean" in document
    assert "../maps/before_after/before_slider.png" in document
    assert "../exports/summary.csv" in document
    assert "https://" not in document


def test_dashboard_handles_missing_optional_maps(tmp_path: Path) -> None:
    """Missing map products should render clear empty states instead of failing."""
    inputs = _dashboard_inputs(tmp_path)

    dashboard = write_dashboard(
        tmp_path / "reports" / "dashboard.html",
        spec=inputs["spec"],
        boundary_path=inputs["boundary"],
        scene_t1=inputs["scene_t1"],
        scene_t2=inputs["scene_t2"],
        analytics=inputs["analytics"],
        maps={},
        sources=(),
    )

    document = dashboard.read_text(encoding="utf-8")
    assert "Before/after map artifacts were not generated" in document
    assert "No publication map images are available" in document
    assert "No scene catalog records are available" in document
    assert "Sentinel-2 L2A" in document
    assert "Analyst Interpretation" in document
    assert "GeoWatch quality" in document
    assert "Water interpretation is limited" in document


def _dashboard_inputs(tmp_path: Path) -> dict[str, object]:
    """Build a compact but complete dashboard input bundle."""
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
        classification_results={"lulc_t1": classification, "lulc_t2": classification},
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
        outputs=OutputSpec(root=tmp_path),
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
    download = tmp_path / "exports" / "summary.csv"
    download.parent.mkdir(parents=True)
    download.write_text("metric,value\n", encoding="utf-8")
    return {
        "boundary": boundary,
        "scene_t1": scene_t1,
        "scene_t2": scene_t2,
        "analytics": analytics,
        "spec": spec,
        "sources": sources,
        "download": download,
    }
