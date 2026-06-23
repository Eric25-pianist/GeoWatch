"""Remote sensing analytics for GeoWatch Phase 4."""

from __future__ import annotations

from geowatch.analytics.change import detect_change_suite
from geowatch.analytics.classification import (
    assess_accuracy,
    build_transition_result,
    classify_lulc,
)
from geowatch.analytics.indices import compute_scene_indices, compute_spectral_indices
from geowatch.analytics.models import (
    ANALYTICS_CLASS_NAMES,
    ANALYTICS_INDEX_NAMES,
    AccuracyAssessment,
    AnalyticsReport,
    ChangeDetectionResult,
    ClassificationResult,
    IndexStatistics,
    MapStatistics,
    SpectralIndexResult,
    ThresholdResult,
    TransitionResult,
)
from geowatch.analytics.pipeline import run_analytics_pipeline
from geowatch.analytics.thresholding import apply_threshold

__all__ = [
    "ANALYTICS_CLASS_NAMES",
    "ANALYTICS_INDEX_NAMES",
    "AccuracyAssessment",
    "AnalyticsReport",
    "ChangeDetectionResult",
    "ClassificationResult",
    "IndexStatistics",
    "MapStatistics",
    "SpectralIndexResult",
    "ThresholdResult",
    "TransitionResult",
    "apply_threshold",
    "assess_accuracy",
    "build_transition_result",
    "classify_lulc",
    "compute_scene_indices",
    "compute_spectral_indices",
    "detect_change_suite",
    "run_analytics_pipeline",
]
