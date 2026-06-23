"""Deterministic output layout and project specification persistence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from geowatch.application.models import RunSpecification
from geowatch.core.errors import ConfigurationError


def location_slug(value: str) -> str:
    """Convert a location name into a portable folder name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    if not slug:
        raise ConfigurationError("Location name does not contain usable characters.")
    return slug


@dataclass(frozen=True)
class ProjectLayout:
    """All stable paths used by a professional GeoWatch run."""

    root: Path

    @classmethod
    def from_spec(cls, spec: RunSpecification) -> ProjectLayout:
        """Build a project path from the output root and location name."""
        return cls(spec.outputs.root / location_slug(spec.location.name))

    @property
    def specification(self) -> Path:
        """Return the persisted project specification path."""
        return self.root / "project.yaml"

    @property
    def manifest(self) -> Path:
        """Return the resumable stage manifest path."""
        return self.root / "run_manifest.json"

    def directories(self, years: tuple[int, ...]) -> tuple[Path, ...]:
        """Return every directory required for the selected years."""
        common = (
            self.root / "boundary" / "source",
            self.root / "boundary" / "validated",
            self.root / "boundary" / "preview",
            self.root / "processed" / "composites",
            self.root / "indices" / "change",
            self.root / "classification" / "transitions",
            self.root / "change",
            self.root / "statistics",
            self.root / "maps" / "comparisons",
            self.root / "maps" / "indices",
            self.root / "maps" / "lulc",
            self.root / "maps" / "change",
            self.root / "reports",
            self.root / "exports",
            self.root / "validation",
            self.root / "logs",
            self.root / "cache",
            self.root / "configs",
        )
        yearly = tuple(
            path
            for year in years
            for path in (
                self.root / "raw" / str(year),
                self.root / "processed" / str(year),
            )
        )
        return common + yearly

    def create(self, years: tuple[int, ...]) -> None:
        """Create the complete project directory tree."""
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.directories(years):
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("Prepared GeoWatch project layout at {}", self.root)


def write_run_specification(spec: RunSpecification, layout: ProjectLayout) -> Path:
    """Persist a validated run specification as readable YAML."""
    layout.create(spec.temporal.years())
    try:
        layout.specification.write_text(
            yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.exception(
            "Could not write project specification {}", layout.specification
        )
        raise ConfigurationError("Could not write project specification") from exc
    return layout.specification


def load_run_specification(path: Path) -> RunSpecification:
    """Load and validate a professional project specification."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return RunSpecification.model_validate(payload)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        logger.exception("Could not load run specification {}", path)
        raise ConfigurationError(f"Invalid project specification: {path}") from exc
