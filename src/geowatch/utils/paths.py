"""Path creation and validation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from geowatch.core.errors import InitializationError

PROJECT_DIRECTORIES: tuple[str, ...] = (
    "configs/examples",
    "configs/schemas",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/cache",
    "docs",
    "logs",
    "outputs/rasters",
    "outputs/vectors",
    "outputs/maps",
    "outputs/reports",
    "outputs/statistics",
    "outputs/manifests",
    "outputs/exports",
    "scripts",
    "tests/unit",
    "tests/integration",
    "tests/fixtures/sample_data",
    "tests/cli",
    "docker",
    ".github/workflows",
)


def ensure_directories(
    root: Path,
    directories: Iterable[str] = PROJECT_DIRECTORIES,
) -> list[Path]:
    """Create project directories under ``root`` and return the created paths."""
    created: list[Path] = []
    try:
        for directory in directories:
            path = root / directory
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
            logger.debug("Ensured directory exists: {}", path)
    except OSError as exc:
        logger.exception("Failed to create project directories under {}", root)
        msg = f"Could not create project directories in {root}"
        raise InitializationError(msg) from exc
    return created


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for ``path`` and return ``path``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.exception("Failed to create parent directory for {}", path)
        msg = f"Could not create parent directory for {path}"
        raise InitializationError(msg) from exc
    return path
