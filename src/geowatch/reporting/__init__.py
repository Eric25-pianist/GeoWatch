"""Reporting helpers for GeoWatch publication workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geowatch.reporting.cartography import analyze_hotspots, render_cartography_suite
from geowatch.reporting.dashboard import render_dashboard, write_dashboard
from geowatch.reporting.demo import DemoPublicationInputs, build_demo_publication_inputs
from geowatch.reporting.exports import export_publication_tables
from geowatch.reporting.interpretation import (
    InterpretationReport,
    InterpretationSection,
    generate_interpretation,
    render_interpretation_html,
    write_interpretation,
)
from geowatch.reporting.lahore_qc import LahoreQcResult, run_lahore_qc
from geowatch.reporting.models import HotspotAnalysis, MapArtifact, PublicationBundle
from geowatch.reporting.phase import generate_phase_report

__all__ = [
    "DemoPublicationInputs",
    "HotspotAnalysis",
    "InterpretationReport",
    "InterpretationSection",
    "LahoreQcResult",
    "MapArtifact",
    "PublicationBundle",
    "analyze_hotspots",
    "build_demo_publication_inputs",
    "build_phase5_publication",
    "export_portfolio_package",
    "export_publication_tables",
    "generate_interpretation",
    "generate_phase_report",
    "render_cartography_suite",
    "render_dashboard",
    "render_interpretation_html",
    "run_lahore_qc",
    "write_build_report",
    "write_dashboard",
    "write_interpretation",
    "write_phase_report",
]


def build_phase5_publication(config: Any) -> PublicationBundle:
    """Lazily import the Phase 5 builder to avoid circular imports."""
    from geowatch.reporting.phase5 import build_phase5_publication as _impl

    return _impl(config)


def write_build_report(
    bundle: PublicationBundle,
    validation_summary: Mapping[str, object],
    path: Path,
) -> Path:
    """Lazily import the build report writer to avoid circular imports."""
    from geowatch.reporting.phase5 import write_build_report as _impl

    return _impl(bundle, validation_summary, path)


def write_phase_report(
    bundle: PublicationBundle,
    validation_summary: Mapping[str, object],
    path: Path,
) -> Path:
    """Lazily import the phase report writer to avoid circular imports."""
    from geowatch.reporting.phase5 import write_phase_report as _impl

    return _impl(bundle, validation_summary, path)


def export_portfolio_package(*args: Any, **kwargs: Any) -> dict[str, Path]:
    """Lazily import the portfolio exporter to avoid circular imports."""
    from geowatch.portfolio.exporter import export_portfolio_package as _impl

    return _impl(*args, **kwargs)
