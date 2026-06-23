"""Loguru-based logger configuration for application and pipeline logs."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from geowatch.core.errors import LoggingSetupError


class LoggerManager:
    """Configure and expose GeoWatch log sinks."""

    LOG_FILES: tuple[str, ...] = (
        "application.log",
        "pipeline.log",
        "error.log",
        "debug.log",
    )

    def __init__(self, log_dir: Path, level: str = "INFO") -> None:
        self.log_dir = log_dir
        self.level = level

    def configure(self) -> None:
        """Configure console and file sinks for Loguru."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.remove()
            logger.add(sys.stderr, level=self.level, enqueue=False)
            logger.add(
                self.log_dir / "application.log",
                level=self.level,
                rotation="10 MB",
            )
            logger.add(
                self.log_dir / "pipeline.log",
                level="INFO",
                rotation="10 MB",
                filter=lambda record: record["extra"].get("channel") == "pipeline",
            )
            logger.add(self.log_dir / "error.log", level="ERROR", rotation="10 MB")
            logger.add(self.log_dir / "debug.log", level="DEBUG", rotation="10 MB")
            logger.debug("Configured GeoWatch logging in {}", self.log_dir)
        except OSError as exc:
            msg = f"Could not configure logging in {self.log_dir}"
            raise LoggingSetupError(msg) from exc

    def log_files(self) -> tuple[Path, ...]:
        """Return the log file paths managed by this logger."""
        return tuple(self.log_dir / name for name in self.LOG_FILES)
