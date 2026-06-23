"""Run specification and project layout tests."""

from __future__ import annotations

from pathlib import Path

from geowatch.application.models import (
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.project import (
    ProjectLayout,
    load_run_specification,
    location_slug,
    write_run_specification,
)


def test_endpoint_specification_and_project_layout(tmp_path: Path) -> None:
    """Endpoint projects should create every professional output folder."""
    spec = RunSpecification(
        location=LocationSpec(name="New York City", country="United States"),
        temporal=TemporalSpec(start_year=2018, end_year=2020),
        outputs=OutputSpec(root=tmp_path),
    )
    layout = ProjectLayout.from_spec(spec)
    path = write_run_specification(spec, layout)

    assert location_slug("New York City") == "New_York_City"
    assert path == tmp_path / "New_York_City" / "project.yaml"
    assert (layout.root / "raw" / "2018").is_dir()
    assert (layout.root / "classification" / "transitions").is_dir()
    assert (layout.root / "maps" / "comparisons").is_dir()
    assert load_run_specification(path) == spec


def test_annual_and_interval_years_include_end_year() -> None:
    """Temporal strategies should always include the requested end year."""
    annual = TemporalSpec(start_year=2010, end_year=2013, mode="annual")
    interval = TemporalSpec(
        start_year=2010,
        end_year=2015,
        mode="interval",
        interval_years=2,
    )

    assert annual.years() == (2010, 2011, 2012, 2013)
    assert interval.years() == (2010, 2012, 2014, 2015)
