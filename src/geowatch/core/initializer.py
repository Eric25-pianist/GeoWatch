"""Project initialization workflow for GeoWatch Phase 1."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from loguru import logger

from geowatch.config.loader import write_config
from geowatch.config.models import AOIConfig, DateRangeConfig, ProjectConfig
from geowatch.config.schema import write_json_schema
from geowatch.core.errors import InitializationError
from geowatch.utils.paths import ensure_directories


def initialize_project(project_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Create a Phase 1 project structure with sample config and AOI."""
    root = project_dir.expanduser().resolve()
    created = ensure_directories(root)
    sample_config = ProjectConfig(
        project_name="geowatch-foundation-sample",
        aoi=AOIConfig(kind="bbox", bbox=(74.15, 31.35, 74.55, 31.7), crs="EPSG:4326"),
        dates=DateRangeConfig(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31)),
    )
    created.extend(
        [
            _write_default_config(root, sample_config, overwrite=overwrite),
            _write_example_config(root, sample_config, overwrite=overwrite),
            _write_sample_aoi(root, overwrite=overwrite),
            write_json_schema(root / "configs" / "schemas" / "pipeline.schema.json"),
        ]
    )
    logger.info("Initialized GeoWatch project at {}", root)
    return created


def _write_default_config(
    root: Path,
    config: ProjectConfig,
    *,
    overwrite: bool,
) -> Path:
    """Write the default configuration file."""
    path = root / "configs" / "default.yaml"
    if path.exists() and not overwrite:
        logger.info("Keeping existing config {}", path)
        return path
    return write_config(config, path)


def _write_example_config(
    root: Path,
    config: ProjectConfig,
    *,
    overwrite: bool,
) -> Path:
    """Write an example configuration file."""
    path = root / "configs" / "examples" / "lahore_foundation.yaml"
    if path.exists() and not overwrite:
        return path
    return write_config(config, path)


def _write_sample_aoi(root: Path, *, overwrite: bool) -> Path:
    """Write a valid GeoJSON sample AOI for validation tests and quick starts."""
    path = root / "tests" / "fixtures" / "sample_data" / "sample_aoi.geojson"
    if path.exists() and not overwrite:
        return path
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "sample-aoi"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [74.15, 31.35],
                            [74.55, 31.35],
                            [74.55, 31.70],
                            [74.15, 31.70],
                            [74.15, 31.35],
                        ]
                    ],
                },
            }
        ],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not write sample AOI {}", path)
        raise InitializationError(f"Could not write sample AOI: {path}") from exc
    else:
        return path
