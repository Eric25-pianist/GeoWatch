"""Unit tests for Loguru logger setup."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from geowatch.logging.manager import LoggerManager


def test_logger_manager_creates_logs(tmp_path: Path) -> None:
    """LoggerManager should configure the four required log files."""
    manager = LoggerManager(tmp_path, "DEBUG")
    manager.configure()

    logger.info("application message")
    logger.bind(channel="pipeline").info("pipeline message")
    logger.error("error message")
    logger.debug("debug message")
    logger.complete()

    for log_file in manager.log_files():
        assert log_file.exists()
