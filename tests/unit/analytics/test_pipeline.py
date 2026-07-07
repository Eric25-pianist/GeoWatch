"""Unit tests for the Phase 4 analytics pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from geowatch.analytics import run_analytics_pipeline
from geowatch.processing.models import RasterLayer


def test_run_analytics_pipeline_writes_outputs(
    tmp_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    training_labels: NDArray[np.int64],
    reference_labels: NDArray[np.int64],
) -> None:
    """The analytics pipeline should write its report and artifacts."""
    report = run_analytics_pipeline(
        scene_t1,
        scene_t2,
        output_root=tmp_path,
        classification_method="random_forest",
        training_labels_t1=training_labels,
        training_labels_t2=training_labels,
        reference_labels_t1=reference_labels,
        reference_labels_t2=reference_labels,
    )

    assert report.phase == 4
    assert report.artifacts["analytics_report"].exists()
    assert report.artifacts["indices_npz"].exists()
    assert report.artifacts["transition_json"].exists()
    assert "Phase 4" in report.summary()
    report_text = report.artifacts["analytics_report"].read_text(encoding="utf-8")
    assert "GeoWatch Phase 4 Report" in report_text
    assert "Transition Matrix" in report_text


def test_run_analytics_pipeline_can_skip_classification(
    tmp_path: Path,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> None:
    """Users should be able to run indices and change detection without LULC."""
    report = run_analytics_pipeline(
        scene_t1,
        scene_t2,
        output_root=tmp_path,
        classification_method="none",
    )

    assert report.classification_results == {}
    assert report.transition_result.changed_pixels == 0
    assert report.artifacts["ndvi_gain_loss_cog"].exists()
    assert report.artifacts["area_statistics"].exists()
    assert "transition_json" not in report.artifacts
    report_text = report.artifacts["analytics_report"].read_text(encoding="utf-8")
    assert "LULC classification was disabled" in report_text
