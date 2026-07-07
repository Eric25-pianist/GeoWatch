"""Analytics workflow orchestration for GeoWatch Phase 4."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from geowatch.analytics.change import detect_change_suite
from geowatch.analytics.classification import (
    ANALYTICS_CLASS_NAMES,
    assess_accuracy,
    build_transition_result,
    classify_lulc,
)
from geowatch.analytics.errors import AnalyticsError
from geowatch.analytics.indices import (
    ANALYTICS_INDEX_NAMES,
    compute_scene_indices,
    compute_spectral_indices,
)
from geowatch.analytics.models import (
    AccuracyAssessment,
    AnalyticsReport,
    ChangeDetectionResult,
    ClassificationResult,
    SignedChangeResult,
    SpectralIndexResult,
    ThresholdResult,
    TransitionResult,
)
from geowatch.analytics.thresholding import apply_threshold
from geowatch.processing.io import write_raster
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.utils.paths import ensure_parent


def run_analytics_pipeline(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    *,
    output_root: Path = Path("outputs"),
    classification_method: str = "kmeans",
    training_labels_t1: NDArray[np.int_] | NDArray[np.str_] | None = None,
    training_labels_t2: NDArray[np.int_] | NDArray[np.str_] | None = None,
    reference_labels_t1: NDArray[np.int_] | NDArray[np.str_] | None = None,
    reference_labels_t2: NDArray[np.int_] | NDArray[np.str_] | None = None,
    threshold_method: str = "otsu",
    threshold_kwargs: dict[str, object] | None = None,
    index_names: Sequence[str] | None = None,
    change_methods: Sequence[str] | None = None,
) -> AnalyticsReport:
    """Run the Phase 4 analytics workflow and write outputs."""
    _validate_pair(scene_t1, scene_t2)
    indices_dir = output_root / "indices"
    change_dir = output_root / "change"
    classification_dir = output_root / "classification"
    statistics_dir = output_root / "statistics"
    reports_dir = output_root / "reports"
    for directory in (
        indices_dir,
        change_dir,
        classification_dir,
        statistics_dir,
        reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    index_results = compute_spectral_indices(
        scene_t1,
        scene_t2,
        index_names=index_names or ANALYTICS_INDEX_NAMES,
    )
    change_results = detect_change_suite(
        scene_t1,
        scene_t2,
        index_results=index_results,
        methods=change_methods,
        threshold_method=threshold_method,
        threshold_kwargs=threshold_kwargs,
    )
    classification_t1: ClassificationResult | None = None
    classification_t2: ClassificationResult | None = None
    if classification_method != "none":
        scene_indices_t1 = compute_scene_indices(scene_t1)
        scene_indices_t2 = compute_scene_indices(scene_t2)
        classification_t1 = classify_lulc(
            scene_t1,
            method=classification_method,
            training_labels=training_labels_t1,
            index_maps=scene_indices_t1,
        )
        classification_t2 = classify_lulc(
            scene_t2,
            method=classification_method,
            training_labels=training_labels_t2,
            index_maps=scene_indices_t2,
        )
        transition_result = build_transition_result(
            classification_t1,
            classification_t2,
        )
    else:
        transition_result = _empty_transition_result()
    signed_change = _build_signed_ndvi_change(
        index_results["ndvi"], threshold_method=threshold_method
    )
    accuracy: dict[str, AccuracyAssessment] = {}
    if reference_labels_t1 is not None and classification_t1 is not None:
        accuracy["lulc_t1"] = assess_accuracy(
            reference_labels_t1,
            classification_t1.labels,
            class_names=ANALYTICS_CLASS_NAMES,
        )
    if reference_labels_t2 is not None and classification_t2 is not None:
        accuracy["lulc_t2"] = assess_accuracy(
            reference_labels_t2,
            classification_t2.labels,
            class_names=ANALYTICS_CLASS_NAMES,
        )

    artifacts = _write_outputs(
        index_results=index_results,
        change_results=change_results,
        classification_t1=classification_t1,
        classification_t2=classification_t2,
        transition_result=transition_result,
        accuracy=accuracy,
        indices_dir=indices_dir,
        change_dir=change_dir,
        classification_dir=classification_dir,
        statistics_dir=statistics_dir,
        reports_dir=reports_dir,
        grid=scene_t1.grid,
        signed_change=signed_change,
    )
    messages = (
        f"Computed {len(index_results)} spectral indices.",
        f"Computed {len(change_results)} change detection products.",
        (
            "Skipped LULC classification by configuration."
            if classification_method == "none"
            else f"Generated LULC outputs using {classification_method}."
        ),
        (
            "Skipped transition and change matrices because LULC was disabled."
            if classification_method == "none"
            else "Generated transition and change matrices."
        ),
        f"Created {len(accuracy)} accuracy assessments.",
    )
    report = AnalyticsReport(
        phase=4,
        messages=messages,
        index_results=index_results,
        change_results=change_results,
        classification_results=_classification_results(
            classification_t1,
            classification_t2,
        ),
        transition_result=transition_result,
        accuracy=accuracy,
        artifacts=artifacts,
        signed_change=signed_change,
    )
    report_path = reports_dir / "analytics_report.md"
    report_path.write_text(render_analytics_report(report), encoding="utf-8")
    logger.info("Wrote analytics report to {}", report_path)
    report.artifacts.setdefault("analytics_report", report_path)
    return report


def render_analytics_report(report: AnalyticsReport) -> str:
    """Render a markdown analytics report."""
    lines = [
        "# GeoWatch Phase 4 Report",
        "",
        "- Phase: 4 - Remote Sensing Analytics",
        "- Status: PASS",
        "",
        "## Completed Scope",
        "",
    ]
    lines.extend(f"- {message}" for message in report.messages)
    lines.extend(
        [
            "",
            "## Spectral Indices",
            "",
        ]
    )
    for name, index_result in report.index_results.items():
        lines.append(
            f"- {name.upper()}: T1 mean={index_result.statistics.t1.mean:.4f}, "
            f"T2 mean={index_result.statistics.t2.mean:.4f}, "
            f"Difference mean={index_result.statistics.difference.mean:.4f}"
        )
    lines.extend(["", "## Change Detection", ""])
    for name, change_result in report.change_results.items():
        threshold_fraction = (
            change_result.threshold.change_fraction
            if change_result.threshold is not None
            else 0.0
        )
        lines.append(
            f"- {name}: score mean={change_result.statistics.mean:.4f}, "
            f"changed={threshold_fraction:.2%}"
        )
    lines.extend(["", "## Classification", ""])
    if report.classification_results:
        for name, classification_result in report.classification_results.items():
            counts = ", ".join(
                f"{class_name}={classification_result.counts[class_name]}"
                for class_name in classification_result.class_names
            )
            lines.append(f"- {name}: {counts}")
        lines.extend(["", "## Transition Matrix", ""])
        lines.extend(
            _render_matrix(
                report.transition_result.transition_matrix,
                report.transition_result.class_names,
            )
        )
        lines.extend(["", "## Change Matrix", ""])
        lines.extend(
            _render_matrix(
                report.transition_result.change_matrix,
                report.transition_result.class_names,
            )
        )
    else:
        lines.append("- LULC classification was disabled for this run.")
    if report.accuracy:
        lines.extend(["", "## Accuracy Assessment", ""])
        for name, assessment in report.accuracy.items():
            lines.append(
                f"- {name}: overall={assessment.overall_accuracy:.2%}, "
                f"kappa={assessment.kappa:.4f}"
            )
            for class_name, value in assessment.per_class_accuracy.items():
                lines.append(f"  - {class_name}: {value:.2%}")
    lines.extend(["", "## Artifacts", ""])
    for label, path in report.artifacts.items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def _write_outputs(
    *,
    index_results: dict[str, SpectralIndexResult],
    change_results: dict[str, ChangeDetectionResult],
    classification_t1: ClassificationResult | None,
    classification_t2: ClassificationResult | None,
    transition_result: TransitionResult,
    accuracy: dict[str, AccuracyAssessment],
    indices_dir: Path,
    change_dir: Path,
    classification_dir: Path,
    statistics_dir: Path,
    reports_dir: Path,
    grid: RasterGrid,
    signed_change: SignedChangeResult,
) -> dict[str, Path]:
    """Write analytics artifacts to disk."""
    artifacts: dict[str, Path] = {}

    index_npz_path = indices_dir / "spectral_indices.npz"
    index_payload: dict[str, NDArray[np.float32]] = {}
    for name, index_result in index_results.items():
        index_payload[f"{name}_t1"] = index_result.t1.astype(np.float32, copy=False)
        index_payload[f"{name}_t2"] = index_result.t2.astype(np.float32, copy=False)
        index_payload[f"{name}_difference"] = index_result.difference.astype(
            np.float32,
            copy=False,
        )
    _write_npz(index_npz_path, index_payload)
    artifacts["indices_npz"] = index_npz_path
    for name, index_result in index_results.items():
        for period, values in (
            ("t1", index_result.t1),
            ("t2", index_result.t2),
            ("difference", index_result.difference),
        ):
            artifacts[f"{name}_{period}_cog"] = _write_spatial_array(
                values,
                grid,
                indices_dir / name / f"{name}_{period}.tif",
                band_name=f"{name}_{period}",
            )
    artifacts["index_statistics"] = _write_json(
        statistics_dir / "index_statistics.json",
        [
            {
                "name": name,
                "statistics": asdict(result.statistics),
            }
            for name, result in index_results.items()
        ],
    )

    for name, change_result in change_results.items():
        change_npz = change_dir / f"{name}.npz"
        threshold = change_result.threshold
        mask = (
            threshold.mask
            if threshold is not None
            else np.zeros_like(change_result.score, dtype=bool)
        )
        threshold_value = (
            np.asarray(threshold.threshold, dtype=np.float32)
            if threshold is not None
            else np.asarray(np.nan, dtype=np.float32)
        )
        change_payload = {
            "score": change_result.score.astype(np.float32, copy=False),
            "mask": mask,
            "threshold": threshold_value,
        }
        _write_npz(change_npz, change_payload)
        artifacts[f"change_{name}"] = change_npz
        artifacts[f"change_{name}_score_cog"] = _write_spatial_array(
            change_result.score,
            grid,
            change_dir / name / f"{name}_score.tif",
            band_name=f"{name}_score",
        )
        categorical_mask = np.where(
            np.isfinite(change_result.score), mask.astype(np.uint8), 255
        ).astype(np.uint8)
        artifacts[f"change_{name}_mask_cog"] = _write_spatial_array(
            categorical_mask,
            grid,
            change_dir / name / f"{name}_mask.tif",
            band_name=f"{name}_changed_mask",
            nodata=255,
        )
    artifacts["change_statistics"] = _write_json(
        statistics_dir / "change_statistics.json",
        [
            {
                "method": name,
                "statistics": asdict(change_result.statistics),
                "threshold": _serialize_threshold(change_result.threshold),
                "metadata": change_result.metadata,
            }
            for name, change_result in change_results.items()
        ],
    )

    signed_change_cog = _write_spatial_array(
        signed_change.labels,
        grid,
        change_dir / "ndvi_gain_loss.tif",
        band_name="ndvi_loss_no_change_gain",
        nodata=255,
    )
    artifacts["ndvi_gain_loss_cog"] = signed_change_cog
    if classification_t1 is not None and classification_t2 is not None:
        t1_path = classification_dir / "lulc_t1.npy"
        t2_path = classification_dir / "lulc_t2.npy"
        transition_matrix_path = classification_dir / "transition_matrix.npy"
        change_matrix_path = classification_dir / "change_matrix.npy"
        np.save(t1_path, classification_t1.labels)
        np.save(t2_path, classification_t2.labels)
        np.save(transition_matrix_path, transition_result.transition_matrix)
        np.save(change_matrix_path, transition_result.change_matrix)
        lulc_t1_cog = _write_spatial_array(
            _categorical_labels(classification_t1.labels),
            grid,
            classification_dir / "lulc_t1.tif",
            band_name="lulc_t1",
            nodata=255,
        )
        lulc_t2_cog = _write_spatial_array(
            _categorical_labels(classification_t2.labels),
            grid,
            classification_dir / "lulc_t2.tif",
            band_name="lulc_t2",
            nodata=255,
        )
        transition_raster = _transition_labels(
            classification_t1.labels,
            classification_t2.labels,
            len(transition_result.class_names),
        )
        transition_cog = _write_spatial_array(
            transition_raster,
            grid,
            classification_dir / "transition_codes.tif",
            band_name="lulc_transition_code",
            nodata=255,
        )
        artifacts.update(
            {
                "lulc_t1": t1_path,
                "lulc_t2": t2_path,
                "transition_matrix": transition_matrix_path,
                "change_matrix": change_matrix_path,
                "lulc_t1_cog": lulc_t1_cog,
                "lulc_t2_cog": lulc_t2_cog,
                "transition_codes_cog": transition_cog,
            }
        )
        artifacts["classification_statistics"] = _write_json(
            statistics_dir / "classification_statistics.json",
            {
                "lulc_t1": _classification_payload(classification_t1),
                "lulc_t2": _classification_payload(classification_t2),
            },
        )
    if accuracy:
        artifacts["accuracy"] = _write_json(
            statistics_dir / "accuracy.json",
            {
                name: {
                    "overall_accuracy": assessment.overall_accuracy,
                    "kappa": assessment.kappa,
                    "confusion_matrix": assessment.confusion_matrix.tolist(),
                    "per_class_accuracy": assessment.per_class_accuracy,
                }
                for name, assessment in accuracy.items()
            },
        )

    pixel_area_m2 = _pixel_area_square_metres(grid)
    area_payload: dict[str, object] = {
        "pixel_area_m2": pixel_area_m2,
        "ndvi_change": _count_areas(signed_change.counts, pixel_area_m2),
    }
    if classification_t1 is not None and classification_t2 is not None:
        transition_path = _write_json(
            statistics_dir / "transition.json",
            {
                "class_names": transition_result.class_names,
                "transition_matrix": transition_result.transition_matrix.tolist(),
                "change_matrix": transition_result.change_matrix.tolist(),
                "changed_pixels": transition_result.changed_pixels,
            },
        )
        artifacts["transition_json"] = transition_path
        area_payload["lulc_t1"] = _count_areas(
            classification_t1.counts, pixel_area_m2
        )
        area_payload["lulc_t2"] = _count_areas(
            classification_t2.counts, pixel_area_m2
        )
    artifacts["area_statistics"] = _write_json(
        statistics_dir / "area_statistics.json",
        area_payload,
    )
    report_path = reports_dir / "analytics_report.md"
    artifacts["analytics_report"] = report_path
    return artifacts


def _classification_results(
    classification_t1: ClassificationResult | None,
    classification_t2: ClassificationResult | None,
) -> dict[str, ClassificationResult]:
    """Return the optional classification result mapping."""
    if classification_t1 is None or classification_t2 is None:
        return {}
    return {"lulc_t1": classification_t1, "lulc_t2": classification_t2}


def _empty_transition_result() -> TransitionResult:
    """Return an empty transition result when LULC is disabled."""
    empty = np.zeros((0, 0), dtype=np.int64)
    return TransitionResult(
        class_names=(),
        transition_matrix=empty,
        change_matrix=empty.copy(),
        changed_pixels=0,
    )


def _classification_payload(result: ClassificationResult) -> dict[str, object]:
    """Serialize a classification result for statistics JSON."""
    return {
        "method": result.method,
        "model_name": result.model_name,
        "counts": result.counts,
        "feature_names": result.feature_names,
        "metadata": result.metadata,
    }


def _build_signed_ndvi_change(
    result: SpectralIndexResult,
    *,
    threshold_method: str,
) -> SignedChangeResult:
    difference = np.asarray(result.difference, dtype=np.float32)
    threshold_result = apply_threshold(np.abs(difference), method=threshold_method)
    threshold_value = threshold_result.threshold
    if not isinstance(threshold_value, float):
        finite_thresholds = threshold_value[np.isfinite(threshold_value)]
        scalar_threshold = float(np.nanmedian(finite_thresholds))
        negative = difference < -threshold_value
        positive = difference > threshold_value
    else:
        scalar_threshold = threshold_value
        negative = difference < -threshold_value
        positive = difference > threshold_value
    labels = np.full(difference.shape, 255, dtype=np.uint8)
    valid = np.isfinite(difference)
    labels[valid] = 1
    labels[valid & negative] = 0
    labels[valid & positive] = 2
    names = ("Loss", "No change", "Gain")
    counts = {name: int((labels == index).sum()) for index, name in enumerate(names)}
    return SignedChangeResult(
        name="ndvi_gain_loss",
        labels=labels,
        class_names=names,
        threshold=scalar_threshold,
        counts=counts,
    )


def _write_spatial_array(
    values: NDArray[np.generic],
    grid: RasterGrid,
    path: Path,
    *,
    band_name: str,
    nodata: float | int | None = np.nan,
) -> Path:
    array = np.asarray(values)
    spatial_grid = replace(grid, band_names=(band_name,), nodata=nodata)
    layer = RasterLayer(
        name=band_name,
        data=array[np.newaxis, :, :],
        grid=spatial_grid,
    )
    return write_raster(layer, path, driver="COG")


def _categorical_labels(labels: NDArray[np.int64]) -> NDArray[np.uint8]:
    return np.where(labels >= 0, labels, 255).astype(np.uint8)


def _transition_labels(
    t1: NDArray[np.int64],
    t2: NDArray[np.int64],
    class_count: int,
) -> NDArray[np.uint8]:
    valid = (t1 >= 0) & (t2 >= 0)
    encoded = np.full(t1.shape, 255, dtype=np.uint8)
    encoded[valid] = (t1[valid] * class_count + t2[valid]).astype(np.uint8)
    return encoded


def _pixel_area_square_metres(grid: RasterGrid) -> float:
    a, b, _, d, e, _ = grid.transform
    return float(abs((a * e) - (b * d)))


def _count_areas(
    counts: Mapping[str, int], pixel_area_m2: float
) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "pixels": count,
            "hectares": (count * pixel_area_m2) / 10_000,
            "square_kilometres": (count * pixel_area_m2) / 1_000_000,
        }
        for name, count in counts.items()
    }


def _validate_pair(scene_t1: RasterLayer, scene_t2: RasterLayer) -> None:
    """Validate that two scenes can be compared analytically."""
    if (
        scene_t1.grid.height != scene_t2.grid.height
        or scene_t1.grid.width != scene_t2.grid.width
    ):
        message = "Scenes must have matching dimensions."
        logger.error(message)
        raise AnalyticsError(message)
    if scene_t1.data.shape != scene_t2.data.shape:
        message = "Scenes must have matching band stacks."
        logger.error(message)
        raise AnalyticsError(message)
    if scene_t1.grid.crs != scene_t2.grid.crs:
        message = "Scenes must share a common CRS."
        logger.error(message)
        raise AnalyticsError(message)


def _write_json(path: Path, payload: object) -> Path:
    """Write a JSON artifact and return its path."""
    ensure_parent(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_npz(path: Path, payload: Mapping[str, object]) -> Path:
    """Write a compressed NumPy archive and return its path."""
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]
    return path


def _serialize_threshold(threshold: ThresholdResult | None) -> dict[str, object] | None:
    """Serialize a threshold result for JSON output."""
    if threshold is None:
        return None
    threshold_value: object
    if isinstance(threshold.threshold, np.ndarray):
        threshold_value = threshold.threshold.tolist()
    else:
        threshold_value = float(threshold.threshold)
    return {
        "method": threshold.method,
        "threshold": threshold_value,
        "changed_pixels": threshold.changed_pixels,
        "change_fraction": threshold.change_fraction,
        "metadata": threshold.metadata,
    }


def _render_matrix(matrix: np.ndarray, class_names: tuple[str, ...]) -> list[str]:
    """Render a matrix as a markdown table."""
    header = "| Class | " + " | ".join(class_names) + " |"
    separator = "|" + "---|" * (len(class_names) + 1)
    lines = [header, separator]
    for row_name, row in zip(class_names, matrix, strict=False):
        values = " | ".join(str(int(value)) for value in row)
        lines.append(f"| {row_name} | {values} |")
    return lines
