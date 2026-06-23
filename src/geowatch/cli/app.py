"""Typer CLI for GeoWatch Phase 1 foundation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from geowatch import __version__
from geowatch.acquisition.pipeline import run_acquisition
from geowatch.application.project import load_run_specification
from geowatch.application.wizard import run_interactive_wizard
from geowatch.application.workflow import (
    preflight_project,
    process_project,
    project_status,
)
from geowatch.cartography.themes import MapThemeName
from geowatch.config.loader import load_config
from geowatch.config.models import ProjectConfig
from geowatch.core.errors import GeoWatchError
from geowatch.core.initializer import initialize_project
from geowatch.logging.manager import LoggerManager
from geowatch.pipelines.foundation import run_foundation_pipeline, write_map_readiness
from geowatch.reporting.lahore_qc import run_lahore_qc
from geowatch.reporting.phase import generate_phase_report
from geowatch.reporting.phase5 import build_phase5_publication
from geowatch.validation.checks import run_validation
from geowatch.validation.doctor import format_doctor, run_doctor
from geowatch.validation.quality_score import load_quality_report

app = typer.Typer(
    name="geowatch",
    help="Professional terminal GIS and satellite change-detection application.",
    no_args_is_help=True,
)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_REPORT_PATH = Path("PHASE_REPORT.md")
ProjectDirOption = Annotated[Path, typer.Option("--project-dir", help="Project root.")]
ConfigArgument = Annotated[Path, typer.Argument(help="YAML or JSON config.")]
ConfigOption = Annotated[Path, typer.Option("--config", help="Config path.")]
ReportOutputOption = Annotated[Path, typer.Option("--output", help="Report path.")]
OverwriteOption = Annotated[
    bool,
    typer.Option("--overwrite", help="Overwrite generated files."),
]
StrictDepsOption = Annotated[
    bool,
    typer.Option("--strict-deps", help="Treat missing GDAL or Rasterio as errors."),
]
OutputRootOption = Annotated[
    Path | None,
    typer.Option("--output-root", help="Override the output root."),
]
ProjectFileArgument = Annotated[
    Path,
    typer.Argument(help="Professional project.yaml created by the wizard."),
]
MapThemeOption = Annotated[
    MapThemeName | None,
    typer.Option(
        "--map-theme",
        help="Override the professional map theme for this run.",
    ),
]
LahoreOutputOption = Annotated[
    Path | None,
    typer.Option("--output-root", help="Override the Lahore QC output root."),
]


@app.command("init")
def init_command(
    project_dir: ProjectDirOption = Path(),
    overwrite: OverwriteOption = False,
) -> None:
    """Create Phase 1 folders, config, schema, sample AOI, and outputs."""
    try:
        created = initialize_project(project_dir, overwrite=overwrite)
    except GeoWatchError as exc:
        typer.echo(f"Initialization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Initialized GeoWatch project at {project_dir.resolve()}")
    typer.echo(f"Verified {len(created)} project paths.")


@app.command("validate")
def validate_command(
    config_path: ConfigArgument = DEFAULT_CONFIG_PATH,
    strict_deps: StrictDepsOption = False,
) -> None:
    """Validate config, AOI, directories, Python, and dependency availability."""
    report = run_validation(config_path, strict_deps=strict_deps)
    typer.echo(report.format_text())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("run")
def run_command(
    config_path: ConfigArgument = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the Phase 1 foundation readiness pipeline."""
    try:
        config = load_config(config_path)
        LoggerManager(config.logging.directory, config.logging.level).configure()
        manifest = run_foundation_pipeline(config, config_path)
    except GeoWatchError as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Foundation readiness manifest: {manifest}")


@app.command("acquire")
def acquire_command(
    config_path: ConfigArgument = DEFAULT_CONFIG_PATH,
    download: Annotated[
        bool,
        typer.Option("--download", help="Download selected scene assets."),
    ] = False,
) -> None:
    """Search imagery metadata and optionally download selected assets."""
    try:
        config = load_config(config_path)
        config.acquisition.download = download or config.acquisition.download
        LoggerManager(config.logging.directory, config.logging.level).configure()
        result = run_acquisition(config, base_dir=config_path.parent)
    except GeoWatchError as exc:
        typer.echo(f"Acquisition failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Acquired {len(result.scenes)} scenes from {result.provider}; "
        f"catalog: {result.catalog_path}"
    )


@app.command("map")
def map_command(
    config_path: ConfigArgument = DEFAULT_CONFIG_PATH,
) -> None:
    """Validate map output readiness for Phase 1 without generating cartography."""
    try:
        config = load_config(config_path)
        artifact = write_map_readiness(config, config_path)
    except GeoWatchError as exc:
        typer.echo(f"Map readiness failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Map readiness artifact: {artifact}")


@app.command("report")
def report_command(
    config_path: ConfigOption = DEFAULT_CONFIG_PATH,
    output: ReportOutputOption = DEFAULT_REPORT_PATH,
) -> None:
    """Generate the Phase 1 report from current validation results."""
    report = run_validation(config_path)
    try:
        path = generate_phase_report(output, report)
    except GeoWatchError as exc:
        typer.echo(f"Report generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Phase report: {path}")
    if not report.ok:
        logger.error("Phase report generated with validation errors.")
        raise typer.Exit(code=1)


@app.command("publish")
def publish_command(
    config_path: ConfigArgument = DEFAULT_CONFIG_PATH,
    output_root: OutputRootOption = None,
) -> None:
    """Generate Phase 5 publication outputs."""
    try:
        config = load_config(config_path)
        if output_root is not None:
            _override_output_root(config, output_root)
        LoggerManager(config.logging.directory, config.logging.level).configure()
        bundle = build_phase5_publication(config)
    except GeoWatchError as exc:
        typer.echo(f"Publish failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"HTML report: {bundle.html_report}")
    typer.echo(f"PDF report: {bundle.pdf_report}")
    typer.echo(f"Dashboard: {bundle.dashboard}")
    typer.echo(f"Portfolio exports: {bundle.example_outputs['portfolio_directory']}")
    _echo_quality_summary(output_root or config.outputs.root)
    typer.echo(f"Map directory: {bundle.example_outputs['map_directory']}")
    typer.echo(f"Export directory: {bundle.example_outputs['export_directory']}")


@app.command("lahore-qc")
def lahore_qc_command(
    config_2018: Annotated[
        Path,
        typer.Option(
            "--config-2018",
            help="2018 Lahore comparison config.",
        ),
    ] = Path("configs/examples/lahore_2018_summer.yaml"),
    config_2020: Annotated[
        Path,
        typer.Option(
            "--config-2020",
            help="2020 Lahore comparison config.",
        ),
    ] = Path("configs/examples/lahore_2020_summer.yaml"),
    output_root: LahoreOutputOption = None,
) -> None:
    """Run the Lahore QC and repair workflow."""
    try:
        result = run_lahore_qc(config_2018, config_2020, output_root=output_root)
    except GeoWatchError as exc:
        typer.echo(f"Lahore QC failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Validation report: {result.reports['validation_report']}")
    typer.echo(f"Scientific report: {result.reports['scientific_report']}")
    typer.echo(f"HTML report: {result.reports['html_report']}")
    typer.echo(f"PDF report: {result.reports['pdf_report']}")
    typer.echo(f"Exports: {result.outputs['export_directory']}")


@app.command("version")
def version_command() -> None:
    """Print the GeoWatch package version."""
    typer.echo(__version__)


@app.command("doctor")
def doctor_command(
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit nonzero when any production check fails."),
    ] = False,
) -> None:
    """Validate the active Python and production GIS dependencies."""
    checks = run_doctor()
    typer.echo(format_doctor(checks))
    if strict and not all(check.ok for check in checks):
        raise typer.Exit(code=1)


@app.command("wizard")
def wizard_command(
    output_root: Annotated[
        Path, typer.Option("--output-root", help="Parent folder for location projects.")
    ] = Path("outputs"),
    setup_only: Annotated[
        bool,
        typer.Option(
            "--setup-only", help="Create the project without downloading imagery."
        ),
    ] = False,
) -> None:
    """Create and optionally run a professional GIS project interactively."""
    try:
        spec, layout = run_interactive_wizard(output_root=output_root)
        typer.echo(f"Project specification: {layout.specification}")
        if not setup_only:
            try:
                availability = preflight_project(layout.specification)
            except (GeoWatchError, ValueError) as exc:
                raise GeoWatchError(
                    _friendly_provider_failure(spec.imagery.provider, exc)
                ) from exc
            typer.echo(availability.summary())
            _require_fallback_approval(availability.used_fallback)
            try:
                result = process_project(layout.specification)
            except (GeoWatchError, ValueError) as exc:
                raise GeoWatchError(
                    _friendly_provider_failure(spec.imagery.provider, exc)
                ) from exc
            typer.echo(f"Completed project: {result}")
            _echo_quality_summary(result)
    except (GeoWatchError, ValueError) as exc:
        typer.echo(f"Wizard stopped: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("process")
def process_command(
    project_file: ProjectFileArgument,
    map_theme: MapThemeOption = None,
) -> None:
    """Process a professional project from its saved specification."""
    try:
        result = process_project(project_file, resume=False, map_theme=map_theme)
    except (GeoWatchError, ValueError) as exc:
        typer.echo(f"Processing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Completed project: {result}")
    _echo_quality_summary(result)


@app.command("resume")
def resume_command(
    project_file: ProjectFileArgument,
    map_theme: MapThemeOption = None,
) -> None:
    """Resume a project without repeating completed verified stages."""
    try:
        result = process_project(project_file, resume=True, map_theme=map_theme)
    except (GeoWatchError, ValueError) as exc:
        typer.echo(f"Resume failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Completed project: {result}")
    _echo_quality_summary(result)


@app.command("status")
def status_command(project_file: ProjectFileArgument) -> None:
    """Show acquisition, processing, analytics, and publication stage state."""
    try:
        load_run_specification(project_file)
        typer.echo(project_status(project_file))
    except (GeoWatchError, ValueError) as exc:
        typer.echo(f"Status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("quality")
def quality_command(project_file: ProjectFileArgument) -> None:
    """Show the exported GeoWatch quality summary for a completed project."""
    try:
        load_run_specification(project_file)
        report = load_quality_report(
            project_file.parent / "validation" / "quality_score.json"
        )
    except (GeoWatchError, RuntimeError, ValueError) as exc:
        typer.echo(f"Quality summary failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.format_terminal())


def main() -> None:
    """Run the Typer application."""
    app()


def _override_output_root(config: ProjectConfig, output_root: Path) -> None:
    """Rewrite the configured output directories under a new root."""
    outputs = config.outputs
    outputs.root = output_root
    outputs.rasters = output_root / "rasters"
    outputs.vectors = output_root / "vectors"
    outputs.maps = output_root / "maps"
    outputs.reports = output_root / "reports"
    outputs.statistics = output_root / "statistics"
    outputs.manifests = output_root / "manifests"
    outputs.exports = output_root / "exports"


def _require_fallback_approval(used_fallback: bool) -> None:
    """Require interactive approval before material fallback downloads."""
    if used_fallback and not typer.confirm(
        "Use this common fallback policy for all years?", default=True
    ):
        raise GeoWatchError("Imagery fallback policy was not approved.")


def _echo_quality_summary(output_root: Path) -> None:
    """Print the exported quality summary when it exists."""
    quality_path = output_root / "validation" / "quality_score.json"
    if not quality_path.exists():
        return
    typer.echo(load_quality_report(quality_path).format_terminal())


def _friendly_provider_failure(provider: str, exc: Exception) -> str:
    """Add beginner-friendly guidance for common provider/network failures."""
    message = str(exc)
    lowered = message.casefold()
    if provider == "usgs" and (
        "timed out" in lowered
        or "m2m.cr.usgs.gov" in lowered
        or "operation failed after" in lowered
    ):
        return (
            "USGS imagery search did not respond in time. Your project setup is "
            "fine, but the provider or network timed out. Try the wizard again "
            "with Provider set to auto or planetary-computer, or retry USGS later."
        )
    return message
