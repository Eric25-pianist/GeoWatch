"""Interactive beginner-friendly terminal wizard for professional GeoWatch runs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import cast

import typer
from loguru import logger

from geowatch.application.boundaries import (
    BoundaryCandidate,
    candidate_from_file,
    render_boundary_preview,
    save_boundary_candidate,
    search_boundaries,
    validate_candidate,
)
from geowatch.application.models import (
    AnalysisSpec,
    ImagerySpec,
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.project import ProjectLayout, write_run_specification
from geowatch.application.sensors import select_common_sensor
from geowatch.cartography.themes import MAP_THEME_CHOICES, MapThemeName
from geowatch.core.errors import GeoWatchError

SEASONS: dict[str, tuple[int, int]] = {
    "winter": (12, 12),
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "monsoon": (7, 9),
    "custom": (1, 12),
}


def run_interactive_wizard(
    *,
    output_root: Path = Path("outputs"),
) -> tuple[RunSpecification, ProjectLayout]:
    """Ask the minimum questions, confirm a boundary, and persist the project."""
    typer.echo("\nGeoWatch Professional GIS Project Wizard")
    typer.echo("Press Ctrl+C at any time to stop safely.\n")
    location = typer.prompt("Location name").strip()
    country = typer.prompt("Country").strip()
    region_text = typer.prompt("State/province (optional)", default="").strip()
    region = region_text or None
    start_year, end_year = _prompt_years()
    season = (
        typer.prompt(
            "Season [winter/spring/summer/autumn/monsoon/custom]",
            default="summer",
        )
        .strip()
        .lower()
    )
    if season not in SEASONS:
        raise GeoWatchError(f"Unknown season: {season}")
    start_month, end_month = SEASONS[season]
    if season == "custom":
        start_month = typer.prompt("Start month (1-12)", type=int)
        end_month = typer.prompt("End month (1-12)", type=int)

    local_path_text = typer.prompt(
        "Local boundary file (leave blank to search online)", default=""
    ).strip()
    if local_path_text:
        candidate = candidate_from_file(Path(local_path_text), name=location)
    else:
        candidates = search_boundaries(location, country, region)
        candidate = _choose_candidate(candidates)

    provisional = RunSpecification(
        location=LocationSpec(name=location, country=country, region=region),
        temporal=TemporalSpec(
            start_year=start_year,
            end_year=end_year,
            start_month=start_month,
            end_month=end_month,
        ),
        outputs=OutputSpec(root=output_root),
    )
    layout = ProjectLayout.from_spec(provisional)
    layout.create(provisional.temporal.years())
    preview = render_boundary_preview(
        candidate, layout.root / "boundary" / "preview" / "boundary_preview.png"
    )
    findings = validate_candidate(candidate)
    _show_boundary(candidate, preview, findings)
    if not typer.confirm("Use this administrative boundary?", default=True):
        raise GeoWatchError(
            "Boundary was not approved. Run the wizard again or supply a local file."
        )

    advanced = typer.confirm("Open advanced settings?", default=False)
    imagery = ImagerySpec()
    analysis = AnalysisSpec()
    temporal = provisional.temporal
    outputs = provisional.outputs
    if advanced:
        imagery, analysis, temporal, outputs = _advanced_settings(
            imagery, analysis, temporal, outputs
        )
    theme_name = _prompt_map_theme(default=outputs.map_theme)
    outputs = outputs.model_copy(update={"map_theme": theme_name})

    source, validated, metadata = save_boundary_candidate(
        candidate,
        source_path=layout.root / "boundary" / "source" / "boundary.geojson",
        validated_path=layout.root / "boundary" / "validated" / "boundary.geojson",
        metadata_path=layout.root / "boundary" / "validated" / "provenance.json",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name=location,
            country=country,
            region=region,
            administrative_level=candidate.administrative_level,
            boundary_path=validated.resolve(),
            boundary_source=candidate.source,
            boundary_source_url=candidate.source_url,
            boundary_license=candidate.license,
        ),
        temporal=temporal,
        imagery=imagery,
        analysis=analysis,
        outputs=outputs,
    )
    profile = select_common_sensor(start_year, end_year, imagery.sensor)
    typer.echo("\nProject summary")
    typer.echo(f"  Location: {_terminal_text(candidate.display_name)}")
    typer.echo(f"  Years: {spec.temporal.years()}")
    typer.echo(f"  Months: {start_month}-{end_month}")
    typer.echo(f"  Sensor: {profile.display_name}")
    typer.echo(f"  Provider: {imagery.provider}")
    typer.echo(f"  Cloud limit: {imagery.max_cloud_cover:.0f}%")
    typer.echo(f"  LULC: {analysis.classification} (exploratory when unlabeled)")
    typer.echo(f"  Map theme: {_theme_label(outputs.map_theme)}")
    typer.echo(f"  Project folder: {layout.root.resolve()}")
    if not typer.confirm("Create this project?", default=True):
        raise GeoWatchError("Project creation was cancelled.")
    write_run_specification(spec, layout)
    logger.info("Wizard created project using boundary source {}", source)
    logger.debug("Boundary provenance stored at {}", metadata)
    return spec, layout


def _choose_candidate(candidates: tuple[BoundaryCandidate, ...]) -> BoundaryCandidate:
    typer.echo("\nBoundary candidates:")
    for index, candidate in enumerate(candidates, start=1):
        typer.echo(
            f"  {index}. {_terminal_text(candidate.display_name)} | "
            f"admin={candidate.administrative_level or 'n/a'} "
            f"| area={candidate.area_sq_km:,.1f} km2"
        )
    selection = cast(int, typer.prompt("Choose boundary number", default=1, type=int))
    if not 1 <= selection <= len(candidates):
        raise GeoWatchError("Boundary selection is outside the candidate list.")
    return candidates[selection - 1]


def _show_boundary(
    candidate: BoundaryCandidate,
    preview: Path,
    findings: tuple[str, ...],
) -> None:
    typer.echo("\nProposed administrative boundary")
    typer.echo(f"  Name: {_terminal_text(candidate.display_name)}")
    typer.echo(f"  Source: {candidate.source}")
    typer.echo(f"  Admin level: {candidate.administrative_level or 'unknown'}")
    typer.echo(f"  Area: {candidate.area_sq_km:,.2f} km2")
    typer.echo(f"  Bounds: {candidate.bounds}")
    typer.echo(f"  Preview: {preview.resolve()}")
    for finding in findings:
        typer.echo(f"  Check: {finding}")


def _advanced_settings(
    imagery: ImagerySpec,
    analysis: AnalysisSpec,
    temporal: TemporalSpec,
    outputs: OutputSpec,
) -> tuple[ImagerySpec, AnalysisSpec, TemporalSpec, OutputSpec]:
    mode = typer.prompt("Time mode [endpoints/annual/interval]", default=temporal.mode)
    interval = temporal.interval_years
    if mode == "interval":
        interval = typer.prompt("Interval in years", default=2, type=int)
    sensor = typer.prompt("Sensor [auto/landsat/sentinel-2]", default=imagery.sensor)
    provider = typer.prompt(
        "Provider [auto/planetary-computer/usgs/copernicus]",
        default=imagery.provider,
    )
    cloud = typer.prompt(
        "Maximum cloud cover percent", default=imagery.max_cloud_cover, type=float
    )
    max_scenes = typer.prompt(
        "Maximum scenes/tiles per year",
        default=imagery.max_scenes_per_year,
        type=int,
    )
    classification = typer.prompt(
        "LULC [kmeans/isodata/random_forest/xgboost/svm]",
        default=analysis.classification,
    )
    training_text = ""
    if classification in {"random_forest", "xgboost", "svm"}:
        typer.echo(
            "  Supervised LULC methods need a labeled raster where pixel values "
            "represent known classes."
        )
        typer.echo(
            "  If you do not already have that file, choose kmeans or isodata "
            "instead."
        )
        training_text = _prompt_optional_text("Aligned labeled training raster path")
        if not training_text:
            raise GeoWatchError(
                f"{classification} requires a labeled training raster. "
                "Choose kmeans or isodata if you do not have one yet."
            )
        training_path = Path(training_text)
        if not training_path.exists():
            raise GeoWatchError(f"Training raster file does not exist: {training_path}")
    workers = typer.prompt(
        "Maximum worker processes", default=outputs.max_workers, type=int
    )
    return (
        imagery.model_copy(
            update={
                "sensor": sensor,
                "provider": provider,
                "max_cloud_cover": cloud,
                "max_scenes_per_year": max_scenes,
            }
        ),
        analysis.model_copy(
            update={
                "classification": classification,
                "training_data": Path(training_text) if training_text else None,
            }
        ),
        temporal.model_copy(update={"mode": mode, "interval_years": interval}),
        outputs.model_copy(update={"max_workers": workers}),
    )


def _prompt_map_theme(*, default: MapThemeName) -> MapThemeName:
    """Prompt for one of the supported professional map themes."""
    typer.echo("\nChoose map design theme:")
    for index, (_name, label) in enumerate(MAP_THEME_CHOICES, start=1):
        typer.echo(f"{index}. {label}")
    choice = typer.prompt("Theme number", default=_theme_index(default), type=int)
    if not 1 <= choice <= len(MAP_THEME_CHOICES):
        raise GeoWatchError("Theme selection is outside the supported list.")
    return cast(MapThemeName, MAP_THEME_CHOICES[choice - 1][0])


def _theme_index(name: MapThemeName) -> int:
    """Return the 1-based menu index for a theme."""
    for index, (theme_name, _) in enumerate(MAP_THEME_CHOICES, start=1):
        if theme_name == name:
            return index
    return 1


def _theme_label(name: MapThemeName) -> str:
    """Return the display label for a theme name."""
    for theme_name, label in MAP_THEME_CHOICES:
        if theme_name == name:
            return label
    return name.replace("_", " ").title()


def _terminal_text(value: str) -> str:
    """Render multilingual names safely on legacy Windows console encodings."""
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _prompt_years() -> tuple[int, int]:
    """Accept either one start year or a compact start-end range."""
    while True:
        raw = typer.prompt("Start year or range (example: 2018-2020)").strip()
        try:
            start_year, embedded_end = parse_year_range(raw)
        except ValueError as exc:
            typer.echo(f"Invalid year entry: {exc}")
            continue
        end_year = (
            embedded_end
            if embedded_end is not None
            else cast(int, typer.prompt("End year", type=int))
        )
        return start_year, end_year


def _prompt_optional_text(label: str) -> str:
    """Read one optional free-text value without forcing a non-empty response."""
    typer.echo(f"{label}: ", nl=False)
    try:
        return input().strip()
    except EOFError as exc:
        raise GeoWatchError(f"{label} was not provided.") from exc


def parse_year_range(value: str) -> tuple[int, int | None]:
    """Parse `YYYY` or `YYYY-YYYY` for the beginner wizard."""
    match = re.fullmatch(r"\s*(\d{4})(?:\s*-\s*(\d{4}))?\s*", value)
    if match is None:
        raise ValueError("enter a four-digit year or range such as 2018-2020")
    start_year = int(match.group(1))
    end_year = int(match.group(2)) if match.group(2) else None
    if end_year is not None and start_year >= end_year:
        raise ValueError("the end year must be later than the start year")
    return start_year, end_year
