"""Atomic resumable stage tracking for professional GeoWatch runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from geowatch.core.errors import GeoWatchError

StageStatus = Literal["pending", "running", "completed", "failed"]


class StageRecord(BaseModel):
    """Execution state for one idempotent pipeline stage."""

    status: StageStatus = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    message: str = ""
    artifacts: tuple[Path, ...] = ()


class RunManifest(BaseModel):
    """Versioned state document used by process, resume, and status commands."""

    schema_version: str = "1.0"
    project_file: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stages: dict[str, StageRecord] = Field(default_factory=dict)

    def start(self, stage: str) -> None:
        """Mark a stage running."""
        self.stages[stage] = StageRecord(status="running", started_at=datetime.now(UTC))
        self.updated_at = datetime.now(UTC)

    def complete(self, stage: str, *artifacts: Path, message: str = "") -> None:
        """Mark a stage complete with its material artifacts."""
        previous = self.stages.get(stage, StageRecord())
        self.stages[stage] = StageRecord(
            status="completed",
            started_at=previous.started_at,
            completed_at=datetime.now(UTC),
            message=message,
            artifacts=tuple(artifacts),
        )
        self.updated_at = datetime.now(UTC)

    def fail(self, stage: str, message: str) -> None:
        """Record a recoverable stage failure."""
        previous = self.stages.get(stage, StageRecord())
        self.stages[stage] = StageRecord(
            status="failed",
            started_at=previous.started_at,
            completed_at=datetime.now(UTC),
            message=message,
        )
        self.updated_at = datetime.now(UTC)

    def is_complete(self, stage: str) -> bool:
        """Return whether a stage and all recorded artifacts remain available."""
        record = self.stages.get(stage)
        return bool(
            record
            and record.status == "completed"
            and all(path.exists() for path in record.artifacts)
        )


def load_or_create_manifest(path: Path, project_file: Path) -> RunManifest:
    """Load an existing manifest or initialize a new one."""
    if not path.exists():
        return RunManifest(project_file=project_file)
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.exception("Invalid run manifest {}", path)
        raise GeoWatchError(f"Invalid run manifest: {path}") from exc


def save_manifest(manifest: RunManifest, path: Path) -> Path:
    """Atomically save a run manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        logger.exception("Could not save run manifest {}", path)
        raise GeoWatchError(f"Could not save run manifest: {path}") from exc
    return path
