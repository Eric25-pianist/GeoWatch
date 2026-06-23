"""Generate the Phase 1 completion report."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from geowatch.core.errors import ValidationFailure
from geowatch.utils.paths import ensure_parent
from geowatch.validation.checks import ValidationReport


def generate_phase_report(path: Path, validation_report: ValidationReport) -> Path:
    """Write ``PHASE_REPORT.md`` with completed checks and validation status."""
    status = "PASS" if validation_report.ok else "FAIL"
    lines = [
        "# GeoWatch Phase 1 Report",
        "",
        "- Phase: 1 - Foundation",
        f"- Status: {status}",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Completed Scope",
        "",
        "- Project structure created.",
        "- Packaging, linting, typing, and test configuration created.",
        "- Pydantic V2 configuration system created for YAML and JSON.",
        "- Loguru logging manager created with all required log files.",
        "- Typer CLI created with required Phase 1 commands.",
        "- Validation framework created for config, AOI, directories,",
        "  and dependencies.",
        "- Project initializer created sample config, sample AOI, outputs, and logs.",
        "- Unit, integration, and CLI tests created.",
        "- Installation and run utilities created.",
        "",
        "## Validation Results",
        "",
    ]
    lines.extend(
        f"- `{message.severity.upper()}` {message.check}: {message.message}"
        for message in validation_report.messages
    )
    try:
        destination = ensure_parent(path)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Generated phase report at {}", destination)
    except OSError as exc:
        logger.exception("Failed to generate phase report {}", path)
        raise ValidationFailure(f"Could not write phase report: {path}") from exc
    else:
        return destination
