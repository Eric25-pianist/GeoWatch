"""CLI tests for required Phase 1 commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from geowatch.cli.app import app

runner = CliRunner()


def test_version_command() -> None:
    """The version command should execute."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output.strip()


def test_professional_commands_are_registered() -> None:
    """The beginner and resumable project commands should be discoverable."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("wizard", "process", "resume", "status", "quality"):
        assert command in result.output


def test_process_help_includes_map_theme_override() -> None:
    """The process command should advertise the professional theme selector."""
    result = runner.invoke(app, ["process", "--help"])

    assert result.exit_code == 0
    assert "--map-theme" in result.output


def test_validate_command(sample_config_path: Path) -> None:
    """The validate command should execute successfully."""
    result = runner.invoke(app, ["validate", str(sample_config_path)])

    assert result.exit_code == 0
    assert "GeoWatch validation report" in result.output


def test_init_command(tmp_path: Path) -> None:
    """The init command should create project assets."""
    result = runner.invoke(
        app,
        ["init", "--project-dir", str(tmp_path), "--overwrite"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "configs" / "default.yaml").exists()


def test_run_map_report_commands(tmp_path: Path, sample_config_path: Path) -> None:
    """Run, map, and report commands should execute with the sample config."""
    run_result = runner.invoke(app, ["run", str(sample_config_path)])
    map_result = runner.invoke(app, ["map", str(sample_config_path)])
    report_result = runner.invoke(
        app,
        [
            "report",
            "--config",
            str(sample_config_path),
            "--output",
            str(tmp_path / "PHASE_REPORT.md"),
        ],
    )

    assert run_result.exit_code == 0
    assert map_result.exit_code == 0
    assert report_result.exit_code == 0
    assert (tmp_path / "PHASE_REPORT.md").exists()


def test_quality_command_reads_exported_quality_report(tmp_path: Path) -> None:
    """The quality command should print the saved score summary."""
    project_dir = tmp_path / "Sample_City"
    project_dir.mkdir(parents=True)
    (project_dir / "validation").mkdir()
    (project_dir / "project.yaml").write_text(
        "schema_version: '1.0'\n"
        "location:\n"
        "  name: Sample City\n"
        "  country: Sample Country\n"
        "temporal:\n"
        "  start_year: 2018\n"
        "  end_year: 2020\n"
        "  start_month: 6\n"
        "  end_month: 8\n"
        "  mode: endpoints\n"
        "  interval_years: 1\n"
        "imagery:\n"
        "  sensor: auto\n"
        "  provider: auto\n"
        "  max_cloud_cover: 20.0\n"
        "  max_scenes_per_year: 3\n"
        "  composite_method: median\n"
        "analysis:\n"
        "  indices: [ndvi]\n"
        "  change_methods: [index_differencing]\n"
        "  classification: kmeans\n"
        "outputs:\n"
        f"  root: {str(tmp_path / 'outputs').replace('\\', '/')}\n"
        "  formats: [png, jpeg, pdf]\n"
        "  dpi: [300]\n"
        "  target_crs: auto\n"
        "  max_workers: 2\n",
        encoding="utf-8",
    )
    (project_dir / "validation" / "quality_score.json").write_text(
        "{\n"
        '  "generated_at": "2026-06-20T00:00:00+00:00",\n'
        '  "total_score": 87.0,\n'
        '  "max_score": 100,\n'
        '  "rounded_score": 87,\n'
        '  "overall_status": "High",\n'
        '  "classification_confidence": "Exploratory",\n'
        '  "components": [\n'
        '    {"key":"boundary","title":"Boundary confidence",'
        '"weight":20,"score":18.0,"status":"High","summary":"ok",'
        '"reasons":["ok"],"warnings":[]}\n'
        "  ],\n"
        '  "warnings": ["Example warning"],\n'
        '  "accuracy_metrics": {},\n'
        '  "metadata": {}\n'
        "}\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["quality", str(project_dir / "project.yaml")])

    assert result.exit_code == 0
    assert "GeoWatch Quality Score: 87/100" in result.output
    assert "Warning: Example warning" in result.output
