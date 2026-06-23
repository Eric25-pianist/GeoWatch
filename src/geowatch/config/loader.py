"""Load, validate, and write GeoWatch configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError

from geowatch.config.models import ProjectConfig
from geowatch.core.errors import ConfigurationError
from geowatch.utils.paths import ensure_parent

SUPPORTED_CONFIG_SUFFIXES: tuple[str, ...] = (".yaml", ".yml", ".json")


def load_config(path: Path) -> ProjectConfig:
    """Load a YAML or JSON config file into a validated ``ProjectConfig``."""
    resolved = path.expanduser()
    if not resolved.exists():
        logger.error("Configuration file does not exist: {}", resolved)
        raise ConfigurationError(f"Configuration file does not exist: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_CONFIG_SUFFIXES:
        raise ConfigurationError(
            f"Unsupported config format '{resolved.suffix}'. Use YAML or JSON."
        )

    try:
        raw_text = resolved.read_text(encoding="utf-8")
        data = _parse_config_text(raw_text, resolved.suffix.lower())
        config = ProjectConfig.model_validate(data)
        logger.info("Loaded configuration from {}", resolved)
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        logger.exception("Failed to parse configuration {}", resolved)
        msg = f"Could not parse configuration file: {resolved}"
        raise ConfigurationError(msg) from exc
    except ValidationError as exc:
        logger.exception("Configuration validation failed for {}", resolved)
        raise ConfigurationError(str(exc)) from exc
    else:
        return config


def write_config(config: ProjectConfig, path: Path) -> Path:
    """Write ``config`` to YAML or JSON based on the destination suffix."""
    destination = ensure_parent(path.expanduser())
    data = config.model_dump(mode="json")
    try:
        if destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
        elif destination.suffix.lower() in {".yaml", ".yml"}:
            destination.write_text(
                yaml.safe_dump(data, sort_keys=False),
                encoding="utf-8",
            )
        else:
            raise ConfigurationError(
                f"Unsupported config format '{destination.suffix}'. Use YAML or JSON."
            )
        logger.info("Wrote configuration to {}", destination)
    except OSError as exc:
        logger.exception("Failed to write configuration {}", destination)
        msg = f"Could not write configuration file: {destination}"
        raise ConfigurationError(msg) from exc
    else:
        return destination


def _parse_config_text(raw_text: str, suffix: str) -> dict[str, Any]:
    """Parse configuration text based on its file extension."""
    parsed: Any = (
        json.loads(raw_text) if suffix == ".json" else yaml.safe_load(raw_text)
    )
    if not isinstance(parsed, dict):
        raise ConfigurationError("Configuration root must be a mapping.")
    return parsed
