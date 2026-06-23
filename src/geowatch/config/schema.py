"""JSON schema generation for GeoWatch configuration files."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from geowatch.config.models import ProjectConfig
from geowatch.core.errors import ConfigurationError
from geowatch.utils.paths import ensure_parent


def write_json_schema(path: Path) -> Path:
    """Write the Pydantic-generated configuration schema to ``path``."""
    destination = ensure_parent(path)
    try:
        schema = ProjectConfig.model_json_schema()
        destination.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        logger.info("Wrote configuration schema to {}", destination)
    except OSError as exc:
        logger.exception("Could not write schema to {}", destination)
        raise ConfigurationError(f"Could not write schema to {destination}") from exc
    else:
        return destination
