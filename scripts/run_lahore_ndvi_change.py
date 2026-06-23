"""Run the Lahore NDVI QC and change workflow."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from geowatch.reporting.lahore_qc import run_lahore_qc


def main() -> None:
    """Run the Lahore QC workflow from the project root."""
    project_root = Path.cwd()
    result = run_lahore_qc(
        project_root / "configs/examples/lahore_2018_summer.yaml",
        project_root / "configs/examples/lahore_2020_summer.yaml",
        output_root=project_root / "outputs/lahore_ndvi_qc",
    )
    logger.info("Lahore QC completed: {}", result.output_root)
    print("Lahore QC and NDVI change workflow completed.")
    print(f"Validation report: {result.reports['validation_report']}")
    print(f"Scientific report: {result.reports['scientific_report']}")
    print(f"HTML report: {result.reports['html_report']}")
    print(f"PDF report: {result.reports['pdf_report']}")
    print(f"Exports: {result.outputs['export_directory']}")


if __name__ == "__main__":
    main()
