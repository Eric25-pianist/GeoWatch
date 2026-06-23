"""Unit tests for the GeoWatch run-quality scoring framework."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import (
    AccuracyAssessment,
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
from geowatch.application.availability import AvailabilityPlan, YearAvailability
from geowatch.application.models import (
    AnalysisSpec,
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.validation.quality_score import (
    calculate_quality_score,
    load_quality_report,
    write_quality_outputs,
)


def test_quality_score_is_transparent_and_cautious(tmp_path: Path) -> None:
    """The score should expose components, warnings, and exploratory LULC status."""
    inputs = _inputs(tmp_path)

    report = calculate_quality_score(**inputs)

    assert 0 <= report.rounded_score <= 100
    assert report.component("boundary") is not None
    assert report.component("imagery") is not None
    assert report.classification_confidence == "Exploratory"
    assert (
        "LULC was generated using unsupervised classification"
        in report.format_terminal()
    )
    assert "Cloud-free coverage" in report.as_markdown()


def test_quality_outputs_round_trip_and_accuracy_metrics(tmp_path: Path) -> None:
    """Quality artifacts should export and reload with accuracy metadata."""
    inputs = _inputs(tmp_path)
    analytics = inputs["analytics"]
    inputs["analytics"] = AnalyticsReport(
        phase=analytics.phase,
        messages=analytics.messages,
        index_results=analytics.index_results,
        change_results=analytics.change_results,
        classification_results=analytics.classification_results,
        transition_result=analytics.transition_result,
        accuracy={
            "lulc_t2": AccuracyAssessment(
                class_names=("Water", "Urban", "Vegetation"),
                confusion_matrix=np.eye(3, dtype=np.int64),
                overall_accuracy=0.84,
                kappa=0.76,
                per_class_accuracy={
                    "Water": 0.9,
                    "Urban": 0.8,
                    "Vegetation": 0.82,
                },
            )
        },
        artifacts=analytics.artifacts,
        signed_change=analytics.signed_change,
    )
    inputs["spec"] = RunSpecification(
        location=inputs["spec"].location,
        temporal=inputs["spec"].temporal,
        analysis=AnalysisSpec(
            classification="random_forest", training_data=tmp_path / "labels.tif"
        ),
        outputs=OutputSpec(root=tmp_path),
    )
    report = calculate_quality_score(**inputs)
    paths = write_quality_outputs(tmp_path / "validation", report)
    loaded = load_quality_report(paths["quality_json"])

    assert paths["quality_markdown"].exists()
    assert paths["quality_csv"].exists()
    assert loaded.classification_confidence == "Validated"
    assert loaded.accuracy_metrics["lulc_t2"]["overall_accuracy"] == 0.84


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
    scene_t1 = RasterLayer(
        name="t1",
        data=np.ones((6, 2, 2), dtype=np.float32) * 0.2,
        grid=grid,
    )
    scene_t2 = RasterLayer(
        name="t2",
        data=np.array(
            [
                [[0.3, 0.3], [0.3, np.nan]],
            ]
            * 6,
            dtype=np.float32,
        ),
        grid=grid,
        cloud_mask=np.array([[False, False], [False, True]]),
    )
    analytics = _analytics()
    spec = RunSpecification(
        location=LocationSpec(
            name="Sample City",
            country="Sample Country",
            boundary_path=boundary,
            boundary_source="OpenStreetMap Nominatim",
        ),
        temporal=TemporalSpec(
            start_year=2018, end_year=2020, start_month=6, end_month=8
        ),
        analysis=AnalysisSpec(classification="kmeans"),
        outputs=OutputSpec(root=tmp_path),
    )
    sources = (
        SceneMetadata(
            scene_id="S2-SAMPLE-2018",
            provider="planetary-computer",
            dataset="sentinel-2-l2a",
            acquired_at=datetime(2018, 7, 1, tzinfo=UTC),
            cloud_cover=12.0,
        ),
        SceneMetadata(
            scene_id="S2-SAMPLE-2020",
            provider="planetary-computer",
            dataset="sentinel-2-l2a",
            acquired_at=datetime(2020, 7, 3, tzinfo=UTC),
            cloud_cover=18.0,
        ),
    )
    availability = AvailabilityPlan(
        dataset="sentinel-2-l2a",
        requested_start_month=6,
        requested_end_month=8,
        requested_cloud_cover=20.0,
        effective_start_month=6,
        effective_end_month=9,
        effective_cloud_cover=20.0,
        minimum_scenes_per_year=1,
        years={
            2018: YearAvailability(
                year=2018,
                scene_ids=("S2-SAMPLE-2018",),
                scene_count=1,
                cloud_cover=(12.0,),
                acquired_dates=("2018-07-01",),
                aoi_coverage=1.0,
            ),
            2020: YearAvailability(
                year=2020,
                scene_ids=("S2-SAMPLE-2020",),
                scene_count=1,
                cloud_cover=(18.0,),
                acquired_dates=("2020-07-03",),
                aoi_coverage=1.0,
            ),
        },
        fallback_messages=("common seasonal window expanded from months 6-8 to 6-9",),
    )
    return {
        "spec": spec,
        "boundary_path": boundary,
        "scene_t1": scene_t1,
        "scene_t2": scene_t2,
        "analytics": analytics,
        "sources": sources,
        "availability": availability,
        "maps": {},
        "downloads": {},
    }


def _analytics() -> AnalyticsReport:
    ndvi_stats = IndexStatistics(
        t1=MapStatistics("ndvi_t1", 4, 4, 0.1, 0.4, 0.25, 0.05),
        t2=MapStatistics("ndvi_t2", 4, 3, 0.2, 0.5, 0.34, 0.05),
        difference=MapStatistics("ndvi_diff", 4, 3, -0.1, 0.3, 0.08, 0.05),
    )
    ndvi = SpectralIndexResult(
        name="ndvi",
        t1=np.zeros((2, 2), dtype=np.float32),
        t2=np.zeros((2, 2), dtype=np.float32),
        difference=np.zeros((2, 2), dtype=np.float32),
        statistics=ndvi_stats,
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
    class_names = ("Water", "Urban", "Vegetation")
    classification = ClassificationResult(
        method="kmeans",
        labels=np.zeros((2, 2), dtype=np.int64),
        class_names=class_names,
        counts={"Water": 1, "Urban": 1, "Vegetation": 2},
        model_name="KMeans",
        feature_names=("ndvi",),
    )
    transition = TransitionResult(
        class_names=class_names,
        transition_matrix=np.eye(3, dtype=np.int64),
        change_matrix=np.zeros((3, 3), dtype=np.int64),
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
        index_results={"ndvi": ndvi},
        change_results={"index_differencing": change},
        classification_results={"lulc_t1": classification, "lulc_t2": classification},
        transition_result=transition,
        accuracy={},
        artifacts={},
        signed_change=signed,
    )
