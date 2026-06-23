"""Sensor selection, analytical bands, scaling, and cloud-mask contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from geowatch.acquisition.models import DatasetName
from geowatch.application.models import SensorPreference
from geowatch.core.errors import ConfigurationError


@dataclass(frozen=True)
class SensorProfile:
    """Processing metadata for one harmonized satellite product family."""

    dataset: DatasetName
    display_name: str
    start_year: int
    end_year: int | None
    resolution_m: float
    band_aliases: dict[str, tuple[str, ...]]
    qa_aliases: tuple[str, ...]
    saturation_aliases: tuple[str, ...]
    scale: float
    offset: float


LANDSAT_5 = SensorProfile(
    dataset="landsat-5-c2-l2",
    display_name="Landsat 5 TM Collection 2 Level 2",
    start_year=1984,
    end_year=2012,
    resolution_m=30.0,
    band_aliases={
        "blue": ("blue", "SR_B1", "B1"),
        "green": ("green", "SR_B2", "B2"),
        "red": ("red", "SR_B3", "B3"),
        "nir": ("nir08", "nir", "SR_B4", "B4"),
        "swir1": ("swir16", "SR_B5", "B5"),
        "swir2": ("swir22", "SR_B7", "B7"),
    },
    qa_aliases=("qa_pixel", "QA_PIXEL"),
    saturation_aliases=("qa_radsat", "QA_RADSAT"),
    scale=0.0000275,
    offset=-0.2,
)
LANDSAT_7 = SensorProfile(
    dataset="landsat-7-c2-l2",
    display_name="Landsat 7 ETM+ Collection 2 Level 2",
    start_year=1999,
    end_year=None,
    resolution_m=30.0,
    band_aliases=LANDSAT_5.band_aliases,
    qa_aliases=LANDSAT_5.qa_aliases,
    saturation_aliases=LANDSAT_5.saturation_aliases,
    scale=LANDSAT_5.scale,
    offset=LANDSAT_5.offset,
)
LANDSAT_8 = SensorProfile(
    dataset="landsat-8-c2-l2",
    display_name="Landsat 8 OLI Collection 2 Level 2",
    start_year=2013,
    end_year=None,
    resolution_m=30.0,
    band_aliases={
        "blue": ("blue", "SR_B2", "B2"),
        "green": ("green", "SR_B3", "B3"),
        "red": ("red", "SR_B4", "B4"),
        "nir": ("nir08", "nir", "SR_B5", "B5"),
        "swir1": ("swir16", "SR_B6", "B6"),
        "swir2": ("swir22", "SR_B7", "B7"),
    },
    qa_aliases=("qa_pixel", "QA_PIXEL"),
    saturation_aliases=("qa_radsat", "QA_RADSAT"),
    scale=0.0000275,
    offset=-0.2,
)
LANDSAT_9 = SensorProfile(
    dataset="landsat-9-c2-l2",
    display_name="Landsat 9 OLI-2 Collection 2 Level 2",
    start_year=2021,
    end_year=None,
    resolution_m=30.0,
    band_aliases=LANDSAT_8.band_aliases,
    qa_aliases=LANDSAT_8.qa_aliases,
    saturation_aliases=LANDSAT_8.saturation_aliases,
    scale=LANDSAT_8.scale,
    offset=LANDSAT_8.offset,
)
SENTINEL_2 = SensorProfile(
    dataset="sentinel-2-l2a",
    display_name="Sentinel-2 Level-2A",
    start_year=2015,
    end_year=None,
    resolution_m=10.0,
    band_aliases={
        "blue": ("B02", "blue"),
        "green": ("B03", "green"),
        "red": ("B04", "red"),
        "nir": ("B08", "nir"),
        "swir1": ("B11", "swir16"),
        "swir2": ("B12", "swir22"),
    },
    qa_aliases=("SCL", "scl", "QA60"),
    saturation_aliases=(),
    scale=0.0001,
    offset=0.0,
)

SENSOR_PROFILES = {
    profile.dataset: profile
    for profile in (LANDSAT_5, LANDSAT_7, LANDSAT_8, LANDSAT_9, SENTINEL_2)
}


def select_common_sensor(
    start_year: int,
    end_year: int,
    preference: SensorPreference = "auto",
) -> SensorProfile:
    """Select one scientifically consistent sensor family for both endpoints."""
    if preference == "sentinel-2":
        if start_year < SENTINEL_2.start_year:
            raise ConfigurationError("Sentinel-2 is unavailable before 2015.")
        return SENTINEL_2
    if preference == "landsat":
        return _landsat_for_period(start_year, end_year)
    if start_year >= SENTINEL_2.start_year:
        logger.info("Selected Sentinel-2 for modern high-resolution comparison.")
        return SENTINEL_2
    return _landsat_for_period(start_year, end_year)


def _landsat_for_period(start_year: int, end_year: int) -> SensorProfile:
    """Choose a Landsat mission spanning the comparison where possible."""
    if end_year <= 2012:
        if start_year >= 1999:
            logger.warning(
                "Selected Landsat 7; post-2003 SLC-off gaps require "
                "multi-scene compositing."
            )
            return LANDSAT_7
        return LANDSAT_5
    if start_year >= 2021:
        return LANDSAT_9
    if start_year >= 2013:
        return LANDSAT_8
    raise ConfigurationError(
        "No single Landsat mission spans this period. Split the analysis or enable "
        "cross-sensor harmonization explicitly."
    )


def required_assets(profile: SensorProfile) -> tuple[str, ...]:
    """Return preferred STAC asset names for all analytical bands and QA."""
    names = [aliases[0] for aliases in profile.band_aliases.values()]
    names.append(profile.qa_aliases[0])
    if profile.saturation_aliases:
        names.append(profile.saturation_aliases[0])
    return tuple(names)


def landsat_cloud_mask(values: object) -> NDArray[np.bool_]:
    """Return the Collection 2 QA_PIXEL invalid-data mask."""
    qa = np.asarray(values, dtype=np.uint16)
    invalid_bits = (0, 1, 2, 3, 4, 5)
    mask = np.zeros(qa.shape, dtype=bool)
    for bit in invalid_bits:
        mask |= (qa & (1 << bit)) != 0
    return mask


def sentinel_cloud_mask(
    values: object, *, asset_name: str = "SCL"
) -> NDArray[np.bool_]:
    """Return a Sentinel-2 cloud/shadow/snow invalid-data mask."""
    qa = np.asarray(values)
    if asset_name.upper() == "QA60":
        qa_bits = qa.astype(np.uint16)
        return np.asarray(
            ((qa_bits & (1 << 10)) != 0) | ((qa_bits & (1 << 11)) != 0),
            dtype=np.bool_,
        )
    return np.isin(qa.astype(np.uint8), (0, 1, 3, 8, 9, 10, 11))
