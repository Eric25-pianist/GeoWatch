"""Shared pytest fixtures for GeoWatch Phase 1 tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_config_path() -> Path:
    """Return the repository default configuration path."""
    return Path("configs/default.yaml")
