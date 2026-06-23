"""Integration test for the Lahore QC workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geowatch.config.loader import load_config, write_config
from geowatch.processing.io import write_raster
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.reporting.lahore_qc import run_lahore_qc
from geowatch.utils.geometry import load_vector_geometry


def test_lahore_qc_workflow_runs(tmp_path: Path) -> None:
    """The Lahore QC workflow should run end-to-end on the example data."""
    boundary = Path("configs/examples/lahore_boundary.geojson").resolve()
    config_2018 = _write_qc_fixture(tmp_path, boundary, year=2018, offset=0.00)
    config_2020 = _write_qc_fixture(tmp_path, boundary, year=2020, offset=0.03)

    result = run_lahore_qc(
        config_2018,
        config_2020,
        output_root=tmp_path / "lahore_qc",
    )

    assert result.reports["validation_report"].exists()
    assert result.reports["scientific_report"].exists()
    assert result.reports["html_report"].exists()
    assert result.reports["pdf_report"].exists()
    assert result.outputs["export_directory"].exists()
    assert result.map_artifacts["ndvi_change"].files


def _write_qc_fixture(
    root: Path,
    boundary_path: Path,
    *,
    year: int,
    offset: float,
) -> Path:
    """Create a complete small Sentinel-style catalog for one QC endpoint."""
    boundary = load_vector_geometry(boundary_path)
    west, south, east, north = boundary.geometry.bounds
    width = 32
    height = 32
    grid = RasterGrid(
        crs="EPSG:4326",
        transform=(
            (east - west) / width,
            0.0,
            west,
            0.0,
            -((north - south) / height),
            north,
        ),
        width=width,
        height=height,
        nodata=np.nan,
    )
    rows, columns = np.mgrid[0:height, 0:width]
    pattern = ((rows + columns) / float(width + height)).astype(np.float32)
    values = {
        "B02": 0.12 + (pattern * 0.08),
        "B03": 0.16 + (pattern * 0.10),
        "B04": 0.20 + (pattern * 0.12),
        "B08": 0.45 + (pattern * 0.20) + offset,
        "B11": 0.24 + (pattern * 0.12),
        "B12": 0.18 + (pattern * 0.10),
    }
    scene_id = f"synthetic-sentinel-{year}"
    downloads: list[dict[str, object]] = []
    for band_name, data in values.items():
        path = root / "raw" / str(year) / f"{band_name}.tif"
        layer = RasterLayer(
            name=band_name,
            data=data[np.newaxis, ...].astype(np.float32),
            grid=grid,
        )
        write_raster(layer, path)
        downloads.append(
            {
                "scene_id": scene_id,
                "asset_name": band_name,
                "path": str(path.resolve()),
                "bytes_written": path.stat().st_size,
                "verified": True,
            }
        )

    catalog = root / "catalogs" / str(year) / "acquisition_catalog.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        json.dumps(
            {
                "provider": "synthetic-test-fixture",
                "scenes": [{"scene_id": scene_id}],
                "downloads": downloads,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    template = load_config(Path(f"configs/examples/lahore_{year}_summer.yaml"))
    config = template.model_copy(
        update={
            "aoi": template.aoi.model_copy(update={"path": boundary_path}),
            "acquisition": template.acquisition.model_copy(
                update={"metadata_catalog": catalog.resolve()}
            ),
        }
    )
    config_path = root / "configs" / f"lahore_{year}.yaml"
    write_config(config, config_path)
    return config_path
