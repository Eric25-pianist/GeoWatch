"""Unit tests for Phase 4 LULC classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geowatch.analytics import (
    ANALYTICS_CLASS_NAMES,
    assess_accuracy,
    build_transition_result,
    classify_lulc,
)
from geowatch.processing.models import RasterLayer


def test_unsupervised_classification(scene_t1: RasterLayer) -> None:
    """Unsupervised classification should generate a full label grid."""
    for method in ("kmeans", "isodata"):
        result = classify_lulc(scene_t1, method=method)
        assert result.labels.shape == scene_t1.data.shape[1:]
        assert result.labels.dtype == np.int64
        assert sum(result.counts.values()) == scene_t1.grid.width * scene_t1.grid.height
        assert result.feature_names[:6] == (
            "blue",
            "green",
            "red",
            "nir",
            "swir1",
            "swir2",
        )


def test_supervised_classification_and_transition(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    training_labels: NDArray[np.int64],
) -> None:
    """Supervised classifiers should support transition generation."""
    for method in ("random_forest", "xgboost"):
        classification_t1 = classify_lulc(
            scene_t1,
            method=method,
            training_labels=training_labels,
        )
        classification_t2 = classify_lulc(
            scene_t2,
            method=method,
            training_labels=training_labels,
        )
        transition = build_transition_result(classification_t1, classification_t2)

        assert classification_t1.labels.shape == scene_t1.data.shape[1:]
        assert classification_t2.labels.shape == scene_t2.data.shape[1:]
        assert transition.transition_matrix.shape == (
            len(ANALYTICS_CLASS_NAMES),
            len(ANALYTICS_CLASS_NAMES),
        )
        assert transition.changed_pixels == int(transition.change_matrix.sum())


def test_accuracy_assessment_with_string_reference() -> None:
    """Accuracy assessment should accept string-valued reference labels."""
    reference = np.array(
        [
            ["Water", "Urban"],
            ["Vegetation", "Bare Soil"],
        ],
        dtype=np.str_,
    )
    predicted = np.array([[0, 1], [2, 4]], dtype=np.int64)

    assessment = assess_accuracy(reference, predicted)

    assert assessment.overall_accuracy == 1.0
    assert assessment.kappa == 1.0
    assert assessment.confusion_matrix.trace() == 4
