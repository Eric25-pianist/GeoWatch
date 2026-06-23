"""Thresholding helpers for remote sensing change scores."""

from __future__ import annotations

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from geowatch.analytics.errors import ThresholdingError
from geowatch.analytics.models import ThresholdResult, summarize_array

_EPSILON = np.finfo(np.float32).eps


def apply_threshold(
    score: NDArray[np.float32] | NDArray[np.float64],
    *,
    method: str,
    percentile: float = 90.0,
    manual_threshold: float | None = None,
    window_size: int = 15,
    offset: float = 0.0,
) -> ThresholdResult:
    """Apply a thresholding method to a score map."""
    array = np.asarray(score, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        message = "Thresholding requires at least one finite score value."
        logger.error(message)
        raise ThresholdingError(message)

    method_name = method.lower()
    if method_name == "otsu":
        threshold_value = _otsu_threshold(finite)
        mask = array > threshold_value
        threshold: float | NDArray[np.float32] = float(threshold_value)
    elif method_name == "percentile":
        threshold_value = float(np.nanpercentile(finite, percentile))
        mask = array > threshold_value
        threshold = threshold_value
    elif method_name == "manual":
        if manual_threshold is None:
            message = "manual_threshold must be provided for manual thresholding."
            logger.error(message)
            raise ThresholdingError(message)
        threshold_value = float(manual_threshold)
        mask = array > threshold_value
        threshold = threshold_value
    elif method_name == "adaptive":
        threshold_map = _adaptive_threshold_map(
            array,
            window_size=window_size,
            offset=offset,
        )
        mask = array > threshold_map
        threshold = threshold_map
    else:
        message = f"Unsupported threshold method: {method}"
        logger.error(message)
        raise ThresholdingError(message)

    changed_pixels = int(mask.sum())
    threshold_result = ThresholdResult(
        method=method_name,
        threshold=threshold,
        mask=mask,
        score_statistics=summarize_array(f"{method_name}_score", array),
        changed_pixels=changed_pixels,
        change_fraction=float(changed_pixels / array.size) if array.size else 0.0,
        metadata={
            "percentile": percentile,
            "manual_threshold": manual_threshold,
            "window_size": window_size,
            "offset": offset,
        },
    )
    logger.info(
        "Applied {} thresholding to {} pixels; changed fraction {:.2%}",
        method_name,
        array.size,
        threshold_result.change_fraction,
    )
    return threshold_result


def _otsu_threshold(values: NDArray[np.float32], *, bins: int = 256) -> float:
    """Compute an Otsu threshold from finite score values."""
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return min_value
    histogram, bin_edges = np.histogram(values, bins=bins, range=(min_value, max_value))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total <= _EPSILON:
        raise ThresholdingError("Histogram is empty; cannot compute Otsu threshold.")
    probability = histogram / total
    cumulative_probability = np.cumsum(probability)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cumulative_mean = np.cumsum(probability * bin_centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    numerator = (global_mean * cumulative_probability - cumulative_mean) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        between_class_variance = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    index = int(np.argmax(between_class_variance))
    threshold = float(bin_centers[index])
    logger.debug("Computed Otsu threshold {:.6f}", threshold)
    return threshold


def _adaptive_threshold_map(
    score: NDArray[np.float32],
    *,
    window_size: int,
    offset: float,
) -> NDArray[np.float32]:
    """Compute a local adaptive threshold map."""
    if window_size < 3:
        raise ThresholdingError("adaptive thresholding requires window_size >= 3.")
    if window_size % 2 == 0:
        window_size += 1
    array = np.asarray(score, dtype=np.float32)
    valid = np.isfinite(array)
    filled = np.where(valid, array, 0.0)
    weight = valid.astype(np.float32)
    local_sum = uniform_filter(filled, size=window_size, mode="nearest")
    local_count = uniform_filter(weight, size=window_size, mode="nearest")
    local_mean = np.divide(
        local_sum,
        np.maximum(local_count, _EPSILON),
        out=np.zeros_like(local_sum),
        where=local_count > 0,
    )
    local_square_sum = uniform_filter(filled**2, size=window_size, mode="nearest")
    local_variance = (
        np.divide(
            local_square_sum,
            np.maximum(local_count, _EPSILON),
            out=np.zeros_like(local_square_sum),
            where=local_count > 0,
        )
        - local_mean**2
    )
    local_std = np.sqrt(np.maximum(local_variance, 0.0))
    threshold_map = local_mean + (offset * local_std)
    threshold_map = np.where(valid, threshold_map, np.nan)
    logger.debug(
        "Computed adaptive threshold map with window size {} and offset {}",
        window_size,
        offset,
    )
    return threshold_map.astype(np.float32, copy=False)
