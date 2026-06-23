"""Validate spatial integrity and provenance of a completed GeoWatch project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from loguru import logger
from rasterio.features import geometry_mask


def validate_project(project_file: Path) -> dict[str, Any]:
    """Run strict spatial checks and return a serializable report."""
    project = yaml.safe_load(project_file.read_text(encoding="utf-8"))
    if not isinstance(project, dict):
        raise TypeError("Project specification must be a mapping.")
    location = project.get("location", {})
    imagery = project.get("imagery", {})
    boundary_path = Path(str(location["boundary_path"]))
    boundary = gpd.read_file(boundary_path)
    if boundary.empty or not bool(boundary.geometry.is_valid.all()):
        raise ValueError("Boundary is empty or invalid.")

    root = project_file.parent
    processed = sorted((root / "processed").glob("*/surface_reflectance.tif"))
    if len(processed) < 2:
        raise ValueError("At least two processed composites are required.")

    expected_grid: tuple[object, ...] | None = None
    raster_rows: list[dict[str, Any]] = []
    for path in processed:
        row, grid = _validate_raster(path, boundary)
        if expected_grid is None:
            expected_grid = grid
        elif grid != expected_grid:
            raise ValueError(f"Processed grid mismatch: {path}")
        raster_rows.append(row)

    analytical = sorted(
        path
        for path in root.rglob("*.tif")
        if "raw" not in path.parts
        and "_pre_fix_backup" not in str(path)
        and "_coverage_retry" not in str(path)
        and "_visual_qa_backup" not in str(path)
        and path not in processed
    )
    for path in analytical:
        row, grid = _validate_raster(path, boundary)
        if grid != expected_grid:
            raise ValueError(f"Analytical grid mismatch: {path}")
        raster_rows.append(row)

    catalogs = sorted((root / "raw").glob("*/acquisition_catalog.json"))
    datasets = _catalog_datasets(catalogs)
    expected_mission = _expected_dataset(str(imagery.get("sensor", "auto")), processed)
    if datasets != {expected_mission}:
        raise ValueError(
            f"Mission contamination: expected {expected_mission}, "
            f"found {sorted(datasets)}"
        )

    report: dict[str, Any] = {
        "project": str(project_file),
        "boundary_valid": True,
        "boundary_feature_count": len(boundary),
        "dataset": expected_mission,
        "processed_grid_identical": True,
        "raster_count": len(raster_rows),
        "rasters": raster_rows,
        "all_cog": all(bool(row["is_cog"]) for row in raster_rows),
        "all_outside_valid_pixels_zero": all(
            int(row["valid_pixels_outside_aoi"]) == 0 for row in raster_rows
        ),
    }
    if not report["all_cog"] or not report["all_outside_valid_pixels_zero"]:
        raise ValueError("One or more raster integrity checks failed.")
    return report


def _validate_raster(
    path: Path, boundary: gpd.GeoDataFrame
) -> tuple[dict[str, Any], tuple[object, ...]]:
    """Validate one COG against the approved boundary and return its grid."""
    with rasterio.open(path) as dataset:
        local_boundary = boundary.to_crs(dataset.crs)
        inside = geometry_mask(
            local_boundary.geometry,
            out_shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            invert=True,
        )
        values = dataset.read(1)
        valid = np.isfinite(values)
        if dataset.nodata is not None:
            if np.isnan(dataset.nodata):
                valid &= ~np.isnan(values)
            else:
                valid &= values != dataset.nodata
        outside = int((valid & ~inside).sum())
        layout = dataset.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        row = {
            "path": str(path),
            "crs": str(dataset.crs),
            "width": dataset.width,
            "height": dataset.height,
            "nodata": dataset.nodata,
            "valid_pixels": int(valid.sum()),
            "valid_pixels_outside_aoi": outside,
            "is_cog": layout == "COG",
            "minimum": float(values[valid].min()) if valid.any() else None,
            "maximum": float(values[valid].max()) if valid.any() else None,
        }
        grid: tuple[object, ...] = (
            str(dataset.crs),
            dataset.width,
            dataset.height,
            tuple(dataset.transform),
        )
    logger.info("Validated raster {}", path)
    return row, grid


def _catalog_datasets(catalogs: list[Path]) -> set[str]:
    """Collect normalized datasets from all acquisition catalogs."""
    datasets: set[str] = set()
    for path in catalogs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenes = payload.get("scenes", []) if isinstance(payload, dict) else []
        if isinstance(scenes, list):
            datasets.update(
                str(scene["dataset"])
                for scene in scenes
                if isinstance(scene, dict) and "dataset" in scene
            )
    return datasets


def _expected_dataset(sensor: str, processed: list[Path]) -> str:
    """Read the exact dataset recorded in the first processed composite."""
    del sensor
    with rasterio.open(processed[0]) as dataset:
        raw = dataset.tags().get("dataset")
    if raw is None:
        raise ValueError("Processed composite lacks dataset provenance.")
    parsed = json.loads(raw)
    return str(parsed)


def _write_reports(project_file: Path, report: dict[str, Any]) -> None:
    """Persist JSON and Markdown acceptance evidence."""
    reports = project_file.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "acceptance_validation.json"
    markdown_path = reports / "acceptance_validation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# GeoWatch Acceptance Validation",
        "",
        f"- Dataset: {report['dataset']}",
        f"- Raster count: {report['raster_count']}",
        f"- Identical processed grid: {report['processed_grid_identical']}",
        f"- All rasters are COG: {report['all_cog']}",
        (f"- Zero valid pixels outside AOI: {report['all_outside_valid_pixels_zero']}"),
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run command-line validation with a nonzero failure status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("project_file", type=Path)
    args = parser.parse_args()
    try:
        report = validate_project(args.project_file)
        _write_reports(args.project_file, report)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("Acceptance validation failed: {}", exc)
        return 1
    logger.info("Acceptance validation passed for {}", args.project_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
