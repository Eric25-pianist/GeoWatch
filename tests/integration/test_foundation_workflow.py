"""Integration tests for the Phase 1 foundation workflow."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from geowatch.cli.app import app


def test_init_validate_run_workflow(tmp_path: Path) -> None:
    """A freshly initialized project should validate and run."""
    runner = CliRunner()
    init_result = runner.invoke(
        app,
        ["init", "--project-dir", str(tmp_path), "--overwrite"],
    )
    config_path = tmp_path / "configs" / "default.yaml"
    validate_result = runner.invoke(app, ["validate", str(config_path)])
    run_result = runner.invoke(app, ["run", str(config_path)])

    assert init_result.exit_code == 0
    assert validate_result.exit_code == 0
    assert run_result.exit_code == 0
    assert (Path("outputs") / "manifests" / "foundation_run.json").exists() or (
        tmp_path / "outputs" / "manifests" / "foundation_run.json"
    ).exists()
