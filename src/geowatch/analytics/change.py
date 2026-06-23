"""Change detection algorithms for GeoWatch Phase 4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from geowatch.analytics.errors import ChangeDetectionError
from geowatch.analytics.indices import (
    ANALYTICS_INDEX_NAMES,
    compute_spectral_indices,
    extract_canonical_bands,
)
from geowatch.analytics.models import (
    ChangeDetectionResult,
    SpectralIndexResult,
    summarize_array,
)
from geowatch.analytics.thresholding import apply_threshold
from geowatch.processing.models import RasterLayer

_EPSILON = np.finfo(np.float32).eps

DEFAULT_CHANGE_METHODS: tuple[str, ...] = (
    "index_difference",
    "cva",
    "pca",
    "mad",
    "irmad",
    "ratio",
    "magnitude",
)


def detect_change_suite(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    *,
    index_results: dict[str, SpectralIndexResult] | None = None,
    methods: Sequence[str] | None = None,
    threshold_method: str = "otsu",
    threshold_kwargs: dict[str, object] | None = None,
) -> dict[str, ChangeDetectionResult]:
    """Run the supported change detection algorithms over two scenes."""
    selected_methods = tuple(methods or DEFAULT_CHANGE_METHODS)
    threshold_options: dict[str, object] = dict(threshold_kwargs or {})
    index_bundle = index_results or compute_spectral_indices(
        scene_t1,
        scene_t2,
        index_names=ANALYTICS_INDEX_NAMES,
    )
    bands_t1 = _scene_band_stack(scene_t1)
    bands_t2 = _scene_band_stack(scene_t2)
    valid_mask = np.all(np.isfinite(bands_t1), axis=0) & np.all(
        np.isfinite(bands_t2),
        axis=0,
    )
    if not np.any(valid_mask):
        message = (
            "Change detection requires at least one finite pixel shared by both scenes."
        )
        logger.error(message)
        raise ChangeDetectionError(message)
    results: dict[str, ChangeDetectionResult] = {}
    for method in selected_methods:
        requested_name = method.lower()
        name = {
            "index_differencing": "index_difference",
            "image_ratioing": "ratio",
        }.get(requested_name, requested_name)
        metadata: dict[str, object]
        if name == "index_difference":
            score = _index_difference_score(index_bundle)
            metadata = {"index_names": tuple(index_bundle)}
        elif name == "cva":
            score = _cva_score(bands_t1, bands_t2, valid_mask=valid_mask)
            metadata = {"band_count": bands_t1.shape[0]}
        elif name == "pca":
            score = _pca_score(bands_t1, bands_t2, valid_mask=valid_mask)
            metadata = {"band_count": bands_t1.shape[0], "components": 3}
        elif name == "mad":
            score, mad_metadata = _mad_score(bands_t1, bands_t2, valid_mask=valid_mask)
            metadata = mad_metadata
        elif name == "irmad":
            score, mad_metadata = _irmad_score(
                bands_t1,
                bands_t2,
                valid_mask=valid_mask,
            )
            metadata = mad_metadata
        elif name == "ratio":
            score = _ratio_score(bands_t1, bands_t2, valid_mask=valid_mask)
            metadata = {"band_count": bands_t1.shape[0]}
        elif name == "magnitude":
            score = _magnitude_score(bands_t1, bands_t2, valid_mask=valid_mask)
            metadata = {"band_count": bands_t1.shape[0]}
        else:
            message = f"Unsupported change method: {method}"
            logger.error(message)
            raise ChangeDetectionError(message)

        threshold = apply_threshold(
            score,
            method=threshold_method,
            percentile=cast(float, threshold_options.get("percentile", 90.0)),
            manual_threshold=cast(
                float | None,
                threshold_options.get("manual_threshold"),
            ),
            window_size=cast(int, threshold_options.get("window_size", 15)),
            offset=cast(float, threshold_options.get("offset", 0.0)),
        )
        score_float32 = cast(NDArray[np.float32], np.asarray(score, dtype=np.float32))
        results[name] = ChangeDetectionResult(
            method=name,
            score=score_float32,
            statistics=summarize_array(f"{name}_score", score_float32),
            threshold=threshold,
            metadata=metadata | {"threshold_method": threshold_method},
        )
    logger.info(
        "Computed {} change detection products between {} and {}",
        len(results),
        scene_t1.name,
        scene_t2.name,
    )
    return results


def _scene_band_stack(scene: RasterLayer) -> NDArray[np.float32]:
    """Return a band-first float stack for change detection."""
    bands = extract_canonical_bands(scene)
    stack = np.stack(
        [
            bands["blue"],
            bands["green"],
            bands["red"],
            bands["nir"],
            bands["swir1"],
            bands["swir2"],
        ],
        axis=0,
    )
    return stack.astype(np.float32, copy=False)


def _index_difference_score(
    index_results: dict[str, SpectralIndexResult],
) -> NDArray[np.float32]:
    """Aggregate index differences into a single change score."""
    if not index_results:
        raise ChangeDetectionError("Index differencing requires spectral index maps.")
    stack = np.stack([result.difference for result in index_results.values()], axis=0)
    score = np.nanmean(np.abs(stack), axis=0)
    return cast(NDArray[np.float32], np.asarray(score, dtype=np.float32))


def _cva_score(
    bands_t1: NDArray[np.float32],
    bands_t2: NDArray[np.float32],
    *,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute change vector magnitude."""
    diff = bands_t2 - bands_t1
    score = np.sqrt(np.sum(diff[:, valid_mask] ** 2, axis=0))
    return _pack_score(score, valid_mask)


def _magnitude_score(
    bands_t1: NDArray[np.float32],
    bands_t2: NDArray[np.float32],
    *,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute mean absolute spectral change across bands."""
    score = np.mean(np.abs(bands_t2[:, valid_mask] - bands_t1[:, valid_mask]), axis=0)
    return _pack_score(score, valid_mask)


def _ratio_score(
    bands_t1: NDArray[np.float32],
    bands_t2: NDArray[np.float32],
    *,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute a log-ratio change score."""
    ratio = np.log(
        np.divide(
            bands_t2[:, valid_mask] + _EPSILON,
            bands_t1[:, valid_mask] + _EPSILON,
        )
    )
    score = np.mean(np.abs(ratio), axis=0)
    return _pack_score(score, valid_mask)


def _pca_score(
    bands_t1: NDArray[np.float32],
    bands_t2: NDArray[np.float32],
    *,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute a PCA-based change score from spectral differences."""
    diff = (bands_t2 - bands_t1)[:, valid_mask].T
    scaler = StandardScaler()
    diff_scaled = scaler.fit_transform(diff)
    n_components = min(3, diff_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(diff_scaled)
    score = np.linalg.norm(transformed, axis=1)
    return _pack_score(score, valid_mask)


def _mad_score(
    bands_t1: NDArray[np.float32],
    bands_t2: NDArray[np.float32],
    *,
    valid_mask: NDArray[np.bool_],
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    """Compute Multivariate Alteration Detection scores."""
    x = bands_t1[:, valid_mask].T.astype(np.float64)
    y = bands_t2[:, valid_mask].T.astype(np.float64)
    if weights is None:
        weights_arr = np.ones(x.shape[0], dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = np.clip(weights_arr, 0.0, None)
    if not np.any(weights_arr > 0):
        weights_arr = np.ones_like(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()

    x_centered = x - np.average(x, axis=0, weights=weights_arr)
    y_centered = y - np.average(y, axis=0, weights=weights_arr)
    sxx = _weighted_covariance(x_centered, x_centered, weights_arr)
    syy = _weighted_covariance(y_centered, y_centered, weights_arr)
    sxy = _weighted_covariance(x_centered, y_centered, weights_arr)
    sxx += np.eye(sxx.shape[0]) * 1e-6
    syy += np.eye(syy.shape[0]) * 1e-6

    inv_sxx = np.linalg.pinv(sxx)
    inv_syy = np.linalg.pinv(syy)
    matrix = inv_sxx @ sxy @ inv_syy @ sxy.T
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    canonical_correlations = np.sqrt(eigenvalues)
    b_vectors = inv_syy @ sxy.T @ eigenvectors
    b_vectors = np.divide(
        b_vectors,
        np.where(canonical_correlations > _EPSILON, canonical_correlations, 1.0),
    )
    u = x_centered @ eigenvectors
    v = y_centered @ b_vectors
    mad = u - v
    std = np.sqrt(np.maximum(np.average(mad**2, axis=0, weights=weights_arr), _EPSILON))
    standardized = mad / std
    score = np.sum(standardized**2, axis=1)
    metadata: dict[str, object] = {
        "canonical_correlations": tuple(
            float(value) for value in canonical_correlations
        ),
    }
    return _pack_score(score, valid_mask), metadata


def _irmad_score(
    bands_t1: NDArray[np.float32],
    bands_t2: NDArray[np.float32],
    *,
    valid_mask: NDArray[np.bool_],
    max_iterations: int = 5,
    tolerance: float = 1e-3,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    """Compute iteratively reweighted MAD scores."""
    weights = np.ones(int(np.count_nonzero(valid_mask)), dtype=np.float64)
    metadata: dict[str, object] = {"iterations": 0, "delta": float("inf")}
    score = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    for iteration in range(1, max_iterations + 1):
        score, mad_metadata = _mad_score(
            bands_t1,
            bands_t2,
            valid_mask=valid_mask,
            weights=weights,
        )
        flat_score = np.asarray(score[valid_mask], dtype=np.float64)
        degrees_of_freedom = max(1, bands_t1.shape[0])
        new_weights = chi2.sf(flat_score, df=degrees_of_freedom)
        delta = float(np.mean(np.abs(new_weights - weights)))
        weights = new_weights
        metadata = {
            "iterations": iteration,
            "delta": delta,
            "canonical_correlations": mad_metadata["canonical_correlations"],
        }
        if delta <= tolerance:
            break
    return cast(NDArray[np.float32], score), metadata


def _pack_score(
    score: NDArray[np.float32] | NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Pack a 1D score vector back into the full grid with NaNs outside the AOI."""
    packed = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    packed[valid_mask] = np.asarray(score, dtype=np.float32)
    return packed


def _weighted_covariance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute a weighted covariance matrix."""
    weighted = x.T @ (y * weights[:, None])
    return weighted
