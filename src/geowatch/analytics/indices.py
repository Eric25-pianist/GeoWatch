"""Spectral index calculations for GeoWatch Phase 4."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from geowatch.analytics.errors import IndexValidationError
from geowatch.analytics.models import (
    ANALYTICS_INDEX_NAMES as _ANALYTICS_INDEX_NAMES,
)
from geowatch.analytics.models import (
    IndexStatistics,
    SpectralIndexResult,
    summarize_array,
)
from geowatch.processing.models import RasterLayer

_EPSILON = np.finfo(np.float32).eps

ANALYTICS_INDEX_NAMES = _ANALYTICS_INDEX_NAMES
_INDEX_NAMES: tuple[str, ...] = ANALYTICS_INDEX_NAMES


def compute_scene_indices(
    scene: RasterLayer,
    *,
    index_names: Sequence[str] | None = None,
) -> dict[str, NDArray[np.float32]]:
    """Compute all supported indices for a single raster scene."""
    bands = extract_canonical_bands(scene)
    selected = tuple(index_names or _INDEX_NAMES)
    results: dict[str, NDArray[np.float32]] = {}
    for index_name in selected:
        results[index_name] = _compute_index(index_name, bands)
    logger.info("Computed {} spectral indices for {}", len(results), scene.name)
    return results


def compute_spectral_indices(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    *,
    index_names: Sequence[str] | None = None,
) -> dict[str, SpectralIndexResult]:
    """Compute T1, T2, and difference maps for supported spectral indices."""
    selected = tuple(index_names or _INDEX_NAMES)
    t1_indices = compute_scene_indices(scene_t1, index_names=selected)
    t2_indices = compute_scene_indices(scene_t2, index_names=selected)
    results: dict[str, SpectralIndexResult] = {}
    for index_name in selected:
        t1 = t1_indices[index_name]
        t2 = t2_indices[index_name]
        difference = t2 - t1
        results[index_name] = SpectralIndexResult(
            name=index_name,
            t1=t1,
            t2=t2,
            difference=difference,
            statistics=IndexStatistics(
                t1=summarize_array(f"{index_name}_t1", t1),
                t2=summarize_array(f"{index_name}_t2", t2),
                difference=summarize_array(f"{index_name}_difference", difference),
            ),
        )
    logger.info(
        "Computed {} spectral index time series between {} and {}",
        len(results),
        scene_t1.name,
        scene_t2.name,
    )
    return results


def extract_canonical_bands(scene: RasterLayer) -> Mapping[str, NDArray[np.float32]]:
    """Return canonical spectral bands from a scene."""
    canonical: dict[str, NDArray[np.float32]] = {}
    band_names = scene.grid.band_names or tuple(
        scene.band_name(i) for i in range(scene.band_count)
    )
    for index, band_name in enumerate(band_names):
        canonical_name = _canonical_band_name(band_name)
        if canonical_name not in canonical:
            canonical[canonical_name] = np.asarray(scene.data[index], dtype=np.float32)
    required = {"blue", "green", "red", "nir", "swir1", "swir2"}
    missing = sorted(required.difference(canonical))
    if missing:
        message = f"Scene {scene.name} is missing required bands: {', '.join(missing)}"
        logger.error(message)
        raise IndexValidationError(message)
    return canonical


def _compute_index(
    index_name: str,
    bands: Mapping[str, NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Compute a supported spectral index from canonical bands."""
    name = index_name.lower()
    if name == "ndvi":
        return _bounded_ratio(bands["nir"] - bands["red"], bands["nir"] + bands["red"])
    if name == "evi":
        numerator = 2.5 * (bands["nir"] - bands["red"])
        denominator = bands["nir"] + (6.0 * bands["red"]) - (7.5 * bands["blue"]) + 1.0
        return _safe_ratio(numerator, denominator)
    if name == "savi":
        l_factor = 0.5
        numerator = (bands["nir"] - bands["red"]) * (1.0 + l_factor)
        denominator = bands["nir"] + bands["red"] + l_factor
        return _safe_ratio(numerator, denominator)
    if name == "ndwi":
        return _bounded_ratio(
            bands["green"] - bands["nir"], bands["green"] + bands["nir"]
        )
    if name == "mndwi":
        return _bounded_ratio(
            bands["green"] - bands["swir1"],
            bands["green"] + bands["swir1"],
        )
    if name == "ndbi":
        return _bounded_ratio(
            bands["swir1"] - bands["nir"],
            bands["swir1"] + bands["nir"],
        )
    if name == "bsi":
        numerator = (bands["swir1"] + bands["red"]) - (bands["nir"] + bands["blue"])
        denominator = (bands["swir1"] + bands["red"]) + (bands["nir"] + bands["blue"])
        return _bounded_ratio(numerator, denominator)
    if name == "ndmi":
        return _bounded_ratio(
            bands["nir"] - bands["swir1"], bands["nir"] + bands["swir1"]
        )
    if name == "gndvi":
        return _bounded_ratio(
            bands["nir"] - bands["green"], bands["nir"] + bands["green"]
        )
    if name == "nbr":
        return _bounded_ratio(
            bands["nir"] - bands["swir2"], bands["nir"] + bands["swir2"]
        )
    message = f"Unsupported spectral index: {index_name}"
    logger.error(message)
    raise IndexValidationError(message)


def _safe_ratio(
    numerator: NDArray[np.float32],
    denominator: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Safely compute an index ratio and preserve NaNs for invalid pixels."""
    result = np.full_like(numerator, np.nan, dtype=np.float32)
    valid = np.abs(denominator) > _EPSILON
    np.divide(numerator, denominator, out=result, where=valid)
    return result.astype(np.float32, copy=False)


def _bounded_ratio(
    numerator: NDArray[np.float32],
    denominator: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute a normalized ratio and enforce its physical [-1, 1] range."""
    return np.clip(_safe_ratio(numerator, denominator), -1.0, 1.0).astype(
        np.float32, copy=False
    )


def _canonical_band_name(name: str) -> str:
    """Normalize a band identifier to a canonical GeoWatch band name."""
    normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
    aliases = {
        "blue": {"blue", "b02", "b2", "band2", "band02"},
        "green": {"green", "b03", "b3", "band3", "band03"},
        "red": {"red", "b04", "b4", "band4", "band04"},
        "nir": {"nir", "b08", "b8", "band8", "band08"},
        "swir1": {"swir1", "b11", "band11"},
        "swir2": {"swir2", "b12", "band12"},
    }
    for canonical_name, alias_set in aliases.items():
        if normalized in alias_set:
            return canonical_name
    if normalized.startswith("band") and normalized[4:].isdigit():
        band_number = int(normalized[4:])
        return {
            2: "blue",
            3: "green",
            4: "red",
            8: "nir",
            11: "swir1",
            12: "swir2",
        }.get(band_number, normalized)
    return normalized
