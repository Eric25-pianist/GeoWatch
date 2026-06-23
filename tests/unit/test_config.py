"""Unit tests for configuration loading and model validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from geowatch.config.loader import load_config, write_config
from geowatch.config.models import AOIConfig, DateRangeConfig, ProjectConfig
from geowatch.core.errors import ConfigurationError


def test_load_default_config(sample_config_path: Path) -> None:
    """The committed default YAML config should load as a ProjectConfig."""
    config = load_config(sample_config_path)

    assert isinstance(config, ProjectConfig)
    assert config.project_name == "geowatch-foundation-sample"
    assert config.aoi.kind == "bbox"


def test_config_round_trip_json(tmp_path: Path, sample_config_path: Path) -> None:
    """ProjectConfig should write and reload from JSON."""
    config = load_config(sample_config_path)
    target = tmp_path / "config.json"

    write_config(config, target)
    loaded = load_config(target)

    assert loaded == config


def test_invalid_date_order_raises() -> None:
    """Date ranges must be ordered."""
    with pytest.raises(ValueError, match="start_date"):
        DateRangeConfig(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 1, 1),
        )


def test_invalid_bbox_raises() -> None:
    """Bounding boxes must have ordered coordinates."""
    with pytest.raises(ValueError, match="ordered"):
        AOIConfig(kind="bbox", bbox=(10.0, 1.0, 9.0, 2.0))


def test_missing_config_raises(tmp_path: Path) -> None:
    """Loading a missing config should raise a typed configuration error."""
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")
