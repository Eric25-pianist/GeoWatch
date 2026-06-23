"""Integration tests for the Phase 5 publish CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from geowatch.cli.app import app

runner = CliRunner()


def test_publish_command_generates_outputs(
    tmp_path: Path,
    sample_config_path: Path,
) -> None:
    """The publish command should generate the Phase 5 outputs."""
    output_root = tmp_path / "outputs"
    result = runner.invoke(
        app,
        [
            "publish",
            str(sample_config_path),
            "--output-root",
            str(output_root),
        ],
    )

    assert result.exit_code == 0
    assert (output_root / "reports" / "report.html").exists()
    assert (output_root / "reports" / "dashboard.html").exists()
    assert (output_root / "reports" / "report.pdf").exists()
    interpretation = output_root / "reports" / "interpretation.md"
    assert interpretation.exists()
    assert "Data quality and uncertainty" in interpretation.read_text(encoding="utf-8")
    assert (output_root / "validation" / "quality_score.json").exists()
    assert (output_root / "validation" / "quality_score.md").exists()
    assert (output_root / "maps" / "phase5").exists()
    assert (output_root / "exports" / "phase5" / "summary.csv").exists()
    assert (output_root / "portfolio_exports" / "01_summary_infographic.png").exists()
    assert (
        output_root / "portfolio_exports" / "07_short_portfolio_report.pdf"
    ).exists()
