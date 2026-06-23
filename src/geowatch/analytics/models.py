"""Typed analytics models for remote sensing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

ANALYTICS_INDEX_NAMES: tuple[str, ...] = (
    "ndvi",
    "evi",
    "savi",
    "ndwi",
    "mndwi",
    "ndbi",
    "bsi",
    "ndmi",
    "gndvi",
    "nbr",
)

ANALYTICS_CLASS_NAMES: tuple[str, ...] = (
    "Water",
    "Urban",
    "Vegetation",
    "Agriculture",
    "Bare Soil",
    "Forest",
    "Wetlands",
    "Snow/Ice",
)


@dataclass(frozen=True)
class MapStatistics:
    """Summary statistics for a raster-like analytical map."""

    name: str
    total_pixels: int
    valid_pixels: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float

    @property
    def valid_fraction(self) -> float:
        """Return the fraction of finite pixels."""
        if self.total_pixels == 0:
            return 0.0
        return float(self.valid_pixels / self.total_pixels)


@dataclass(frozen=True)
class IndexStatistics:
    """Statistics for a spectral index at T1, T2, and difference time steps."""

    t1: MapStatistics
    t2: MapStatistics
    difference: MapStatistics


@dataclass(frozen=True)
class ThresholdResult:
    """Thresholding output for a change score map."""

    method: str
    threshold: float | NDArray[np.float32]
    mask: NDArray[np.bool_]
    score_statistics: MapStatistics
    changed_pixels: int
    change_fraction: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectralIndexResult:
    """T1/T2 spectral index product with summary statistics."""

    name: str
    t1: NDArray[np.float32]
    t2: NDArray[np.float32]
    difference: NDArray[np.float32]
    statistics: IndexStatistics


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Change detection result for a single algorithm."""

    method: str
    score: NDArray[np.float32]
    statistics: MapStatistics
    threshold: ThresholdResult | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassificationResult:
    """Land-use / land-cover classification output for one scene."""

    method: str
    labels: NDArray[np.int64]
    class_names: tuple[str, ...]
    counts: dict[str, int]
    model_name: str
    feature_names: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionResult:
    """Transition matrix pair for land-cover comparisons."""

    class_names: tuple[str, ...]
    transition_matrix: NDArray[np.int64]
    change_matrix: NDArray[np.int64]
    changed_pixels: int


@dataclass(frozen=True)
class AccuracyAssessment:
    """Accuracy assessment for a classified raster."""

    class_names: tuple[str, ...]
    confusion_matrix: NDArray[np.int64]
    overall_accuracy: float
    kappa: float
    per_class_accuracy: dict[str, float]


@dataclass(frozen=True)
class SignedChangeResult:
    """Categorical loss/no-change/gain output for a signed index difference."""

    name: str
    labels: NDArray[np.uint8]
    class_names: tuple[str, ...]
    threshold: float
    counts: dict[str, int]


@dataclass(frozen=True)
class AnalyticsReport:
    """High-level analytics result bundle for Phase 4."""

    phase: int
    messages: tuple[str, ...]
    index_results: dict[str, SpectralIndexResult]
    change_results: dict[str, ChangeDetectionResult]
    classification_results: dict[str, ClassificationResult]
    transition_result: TransitionResult
    accuracy: dict[str, AccuracyAssessment]
    artifacts: dict[str, Path]
    signed_change: SignedChangeResult | None = None

    def summary(self) -> str:
        """Render a concise human-readable summary."""
        lines = [f"GeoWatch Phase {self.phase} analytics report"]
        lines.extend(f"- {message}" for message in self.messages)
        lines.append(f"- Spectral indices: {len(self.index_results)}")
        lines.append(f"- Change methods: {len(self.change_results)}")
        lines.append(f"- Classification outputs: {len(self.classification_results)}")
        lines.append(f"- Transition changes: {self.transition_result.changed_pixels}")
        return "\n".join(lines)


def summarize_array(name: str, values: NDArray[np.generic]) -> MapStatistics:
    """Summarize an analytical raster-like array with finite-value statistics."""
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    total_pixels = int(array.size)
    valid_pixels = int(finite.size)
    if finite.size == 0:
        logger.debug("Array {} contains no finite values.", name)
        return MapStatistics(
            name=name,
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            minimum=float("nan"),
            maximum=float("nan"),
            mean=float("nan"),
            standard_deviation=float("nan"),
        )
    return MapStatistics(
        name=name,
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        mean=float(finite.mean()),
        standard_deviation=float(finite.std()),
    )
