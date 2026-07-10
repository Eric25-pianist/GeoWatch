"""Interactive wizard setup test with an offline boundary candidate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shapely.geometry import Polygon
from typer.testing import CliRunner

from geowatch.application.boundaries import BoundaryCandidate
from geowatch.application.models import (
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.project import load_run_specification
from geowatch.cli.app import _friendly_provider_failure, app
from geowatch.core.errors import GeoWatchError


def test_wizard_creates_confirmed_project(tmp_path: Path, monkeypatch: Any) -> None:
    """A beginner can create a reusable project without editing YAML."""
    candidate = BoundaryCandidate(
        name="Test City",
        display_name="Test City, Test Country",
        country_code="tc",
        administrative_level="6",
        source="OpenStreetMap Nominatim",
        source_url="https://example.test/osm",
        license="ODbL",
        geometry=Polygon(((74.0, 31.0), (75.0, 31.0), (75.0, 32.0), (74.0, 32.0))),
        centroid=(74.5, 31.5),
        area_sq_km=10_000.0,
    )

    def fake_search(
        location: str,
        country: str,
        region: str | None,
        **_: object,
    ) -> tuple[BoundaryCandidate, ...]:
        del location, country, region
        return (candidate,)

    monkeypatch.setattr("geowatch.application.wizard.search_boundaries", fake_search)
    runner = CliRunner()
    user_input = "\n".join(
        (
            "Test City",
            "Test Country",
            "",
            "2018",
            "2020",
            "summer",
            "",
            "1",
            "1",
            "y",
            "n",
            "",
            "y",
            "",
        )
    )
    result = runner.invoke(
        app,
        ["wizard", "--setup-only", "--output-root", str(tmp_path)],
        input=user_input,
    )

    assert result.exit_code == 0, result.output
    project = tmp_path / "Test_City"
    assert (project / "project.yaml").exists()
    assert (project / "boundary" / "validated" / "boundary.geojson").exists()
    assert "Project specification" in result.output


def test_wizard_requires_training_raster_for_supervised_lulc(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Supervised classifiers should stop with a plain-language explanation."""
    candidate = BoundaryCandidate(
        name="Test City",
        display_name="Test City, Test Country",
        country_code="tc",
        administrative_level="6",
        source="OpenStreetMap Nominatim",
        source_url="https://example.test/osm",
        license="ODbL",
        geometry=Polygon(((74.0, 31.0), (75.0, 31.0), (75.0, 32.0), (74.0, 32.0))),
        centroid=(74.5, 31.5),
        area_sq_km=10_000.0,
    )

    monkeypatch.setattr(
        "geowatch.application.wizard.search_boundaries",
        lambda *_, **__: (candidate,),
    )
    runner = CliRunner()
    user_input = (
        "\n".join(
            (
                "Test City",
                "Test Country",
                "",
                "2018-2020",
                "summer",
                "",
                "1",
                "1",
                "y",
                "y",
                "",
                "",
                "",
                "",
                "",
                "xgboost",
                "",
            )
        )
        + "\n"
    )
    result = runner.invoke(
        app,
        ["wizard", "--setup-only", "--output-root", str(tmp_path)],
        input=user_input,
    )

    assert result.exit_code == 1
    assert "requires a labeled training raster" in result.output


def test_wizard_normalizes_advanced_choices_before_writing_yaml(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Case-insensitive advanced choices should save a valid project file."""
    candidate = BoundaryCandidate(
        name="Lahore",
        display_name="Lahore District, Punjab, Pakistan",
        country_code="pk",
        administrative_level="6",
        source="OpenStreetMap Nominatim",
        source_url="https://example.test/osm-lahore",
        license="ODbL",
        geometry=Polygon(((74.0, 31.0), (75.0, 31.0), (75.0, 32.0), (74.0, 32.0))),
        centroid=(74.5, 31.5),
        area_sq_km=1_800.0,
    )

    monkeypatch.setattr(
        "geowatch.application.wizard.search_boundaries",
        lambda *_, **__: (candidate,),
    )
    runner = CliRunner()
    user_input = "\n".join(
        (
            "Lahore",
            "Pakistan",
            "Punjab",
            "2015-2017",
            "",
            "",
            "1",
            "1",
            "y",
            "y",
            "Annual",
            "LANDSAT",
            "USGS",
            "",
            "",
            "Isodata",
            "",
            "5",
            "y",
            "",
        )
    )

    result = runner.invoke(
        app,
        ["wizard", "--setup-only", "--output-root", str(tmp_path)],
        input=user_input,
    )

    assert result.exit_code == 0, result.output
    spec = load_run_specification(tmp_path / "Lahore" / "project.yaml")
    assert spec.temporal.mode == "annual"
    assert spec.temporal.years() == (2015, 2016, 2017)
    assert spec.imagery.sensor == "landsat"
    assert spec.imagery.provider == "usgs"
    assert spec.analysis.classification == "isodata"
    assert spec.outputs.map_theme == "dark"


def test_wizard_reports_friendly_usgs_timeout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """USGS timeouts should suggest a practical provider fallback."""
    spec = RunSpecification(
        location=LocationSpec(name="Tokyo", country="Japan"),
        temporal=TemporalSpec(start_year=2020, end_year=2021),
        outputs=OutputSpec(root=tmp_path / "outputs"),
    )
    project_file = tmp_path / "outputs" / "Tokyo" / "project.yaml"
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text("schema_version: '1.0'\n", encoding="utf-8")
    layout = SimpleNamespace(specification=project_file)
    spec = spec.model_copy(
        update={"imagery": spec.imagery.model_copy(update={"provider": "usgs"})}
    )

    monkeypatch.setattr(
        "geowatch.cli.app.run_interactive_wizard",
        lambda **_: (spec, layout),
    )
    monkeypatch.setattr(
        "geowatch.cli.app.preflight_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GeoWatchError(
                "Operation failed after 3 attempts: HTTP request timed out for "
                "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search."
            )
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["wizard"])

    assert result.exit_code == 1
    assert "Provider set to auto or planetary-computer" in result.output


def test_plain_geowatch_launches_welcome_and_wizard(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Typing only geowatch should open the guided application."""
    spec = RunSpecification(
        location=LocationSpec(name="Sample City", country="Sample Country"),
        temporal=TemporalSpec(start_year=2018, end_year=2020),
        outputs=OutputSpec(root=tmp_path / "outputs"),
    )
    project_file = tmp_path / "outputs" / "Sample_City" / "project.yaml"
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text("schema_version: '1.0'\n", encoding="utf-8")
    layout = SimpleNamespace(specification=project_file)

    monkeypatch.setattr(
        "geowatch.cli.app.run_interactive_wizard",
        lambda **_: (spec, layout),
    )
    monkeypatch.setattr(
        "geowatch.cli.app.preflight_project",
        lambda *_args, **_kwargs: SimpleNamespace(
            used_fallback=False,
            summary=lambda: "GeoWatch imagery availability plan\n- ok",
        ),
    )
    monkeypatch.setattr(
        "geowatch.cli.app.process_project",
        lambda *_args, **_kwargs: project_file.parent,
    )

    runner = CliRunner()
    result = runner.invoke(app, [])

    assert result.exit_code == 0, result.output
    assert "GEOWATCH" in result.output
    assert "Project specification" in result.output
    assert "Completed project" in result.output


def test_provider_dns_failure_gets_resume_guidance() -> None:
    """DNS failures should not look like an imagery-policy problem."""
    message = _friendly_provider_failure(
        "auto",
        GeoWatchError(
            "Operation failed after 3 attempts: Could not resolve the imagery "
            "provider host for https://planetarycomputer.microsoft.com/api/stac/v1/search"
        ),
    )

    assert "DNS or internet access failed" in message
    assert "geowatch resume" in message
