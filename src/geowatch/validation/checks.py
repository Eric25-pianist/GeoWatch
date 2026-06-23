"""Validation checks used by the GeoWatch CLI and tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from geowatch.config.loader import load_config
from geowatch.config.models import AOIConfig, ProjectConfig
from geowatch.core.errors import ConfigurationError

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ValidationMessage:
    """Single validation result with severity and actionable text."""

    check: str
    severity: Severity
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation messages."""

    messages: tuple[ValidationMessage, ...]

    @property
    def ok(self) -> bool:
        """Return ``True`` when no error-severity checks are present."""
        return all(message.severity != "error" for message in self.messages)

    def format_text(self) -> str:
        """Render the report as readable text."""
        lines = ["GeoWatch validation report"]
        for message in self.messages:
            lines.append(
                f"[{message.severity.upper()}] {message.check}: {message.message}"
            )
        return "\n".join(lines)


def run_validation(
    config_path: Path | None = None,
    *,
    strict_deps: bool = False,
) -> ValidationReport:
    """Run Phase 1 validation checks and return a structured report."""
    messages: list[ValidationMessage] = []
    config: ProjectConfig | None = None

    messages.extend(validate_python_version())
    messages.extend(validate_dependency("typer", required=True))
    messages.extend(validate_dependency("pydantic", required=True))
    messages.extend(validate_dependency("yaml", required=True))
    messages.extend(validate_dependency("loguru", required=True))
    messages.extend(validate_dependency("rasterio", required=strict_deps))
    messages.extend(
        validate_dependency("osgeo.gdal", required=strict_deps, label="GDAL")
    )

    if config_path is not None:
        try:
            config = load_config(config_path)
            messages.append(
                ValidationMessage("configuration", "info", f"Loaded {config_path}")
            )
        except ConfigurationError as exc:
            messages.append(ValidationMessage("configuration", "error", str(exc)))

    if config is not None:
        base_dir = config_path.parent if config_path else Path()
        messages.extend(validate_aoi(config.aoi, base_dir=base_dir))
        messages.extend(validate_output_directories(config))

    report = ValidationReport(tuple(messages))
    logger.info("Validation completed with status ok={}", report.ok)
    return report


def validate_python_version() -> list[ValidationMessage]:
    """Validate the active Python interpreter version."""
    version = _current_python_version()
    if (3, 12) <= version < (3, 14):
        return [
            ValidationMessage(
                "python",
                "info",
                f"Python {sys.version.split()[0]} is supported (3.12-3.13)",
            )
        ]
    if version >= (3, 14):
        return [
            ValidationMessage(
                "python",
                "warning",
                f"Python {sys.version.split()[0]} can run the core package, but "
                "compiled GIS wheels may be unavailable. Use the supplied Python "
                "3.12 Micromamba environment for real imagery processing.",
            )
        ]
    return [
        ValidationMessage(
            "python",
            "error",
            f"Python {sys.version.split()[0]} is unsupported; "
            "install Python 3.12-3.13.",
        )
    ]


def _current_python_version() -> tuple[int, int]:
    """Return the active Python major and minor version for runtime checks."""
    return (sys.version_info.major, sys.version_info.minor)


def validate_dependency(
    module_name: str, *, required: bool, label: str | None = None
) -> list[ValidationMessage]:
    """Validate that a Python module is importable."""
    display = label or module_name
    if _module_is_importable(module_name):
        return [ValidationMessage(display, "info", f"{display} is importable.")]
    severity: Severity = "error" if required else "warning"
    action = "Install it before running strict geospatial processing."
    return [
        ValidationMessage(
            display,
            severity,
            f"{display} is not importable. {action}",
        )
    ]


def _module_is_importable(module_name: str) -> bool:
    """Return whether ``module_name`` can be found without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        logger.debug("Parent package missing for dependency {}", module_name)
        return False


def validate_aoi(aoi: AOIConfig, *, base_dir: Path) -> list[ValidationMessage]:
    """Validate AOI values and vector file readability where relevant."""
    if aoi.kind == "bbox":
        return [ValidationMessage("aoi", "info", f"Validated bbox {aoi.bbox}")]
    if aoi.path is None:
        return [ValidationMessage("aoi", "error", "AOI path is required.")]

    path = aoi.path if aoi.path.is_absolute() else base_dir / aoi.path
    if not path.exists():
        return [ValidationMessage("aoi", "error", f"AOI file does not exist: {path}")]
    if aoi.kind == "geojson":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to read AOI GeoJSON {}", path)
            return [ValidationMessage("aoi", "error", f"Invalid GeoJSON AOI: {exc}")]
        supported_types = {"FeatureCollection", "Feature", "Polygon", "MultiPolygon"}
        if data.get("type") not in supported_types:
            return [
                ValidationMessage(
                    "aoi",
                    "error",
                    "GeoJSON AOI has unsupported type.",
                )
            ]
    return [ValidationMessage("aoi", "info", f"Validated AOI file {path}")]


def validate_output_directories(config: ProjectConfig) -> list[ValidationMessage]:
    """Create and validate configured output directories."""
    messages: list[ValidationMessage] = []
    for directory in config.output_directories():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            messages.append(
                ValidationMessage(
                    "directory",
                    "info",
                    f"Directory is ready: {directory}",
                )
            )
        except OSError as exc:
            logger.exception("Output directory is not writable: {}", directory)
            messages.append(
                ValidationMessage(
                    "directory",
                    "error",
                    f"Directory is not writable: {directory}: {exc}",
                )
            )
    return messages
