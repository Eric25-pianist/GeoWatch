"""Foundation-only pipeline actions for Phase 1 readiness checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from geowatch.config.models import ProjectConfig
from geowatch.core.errors import ValidationFailure
from geowatch.utils.paths import ensure_parent
from geowatch.validation.checks import run_validation


def run_foundation_pipeline(config: ProjectConfig, config_path: Path) -> Path:
    """Validate Phase 1 systems and write a deterministic readiness manifest."""
    report = run_validation(config_path)
    if not report.ok:
        logger.error("Foundation run failed validation.")
        raise ValidationFailure(report.format_text())

    manifest_path = ensure_parent(config.outputs.manifests / "foundation_run.json")
    payload = {
        "project_name": config.project_name,
        "phase": 1,
        "status": "validated",
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "validation_messages": [message.__dict__ for message in report.messages],
    }
    try:
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to write foundation manifest {}", manifest_path)
        msg = f"Could not write foundation manifest: {manifest_path}"
        raise ValidationFailure(msg) from exc
    logger.bind(channel="pipeline").info(
        "Foundation manifest written to {}",
        manifest_path,
    )
    return manifest_path


def write_map_readiness(config: ProjectConfig, config_path: Path) -> Path:
    """Write a Phase 1 map-readiness artifact without generating cartography."""
    report = run_validation(config_path)
    if not report.ok:
        raise ValidationFailure(report.format_text())
    path = ensure_parent(config.outputs.maps / "map_readiness.json")
    payload = {
        "project_name": config.project_name,
        "phase": 1,
        "cartography_status": "output directory and configuration validated",
        "map_generation_phase": 5,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to write map readiness artifact {}", path)
        msg = f"Could not write map readiness artifact: {path}"
        raise ValidationFailure(msg) from exc
    return path
