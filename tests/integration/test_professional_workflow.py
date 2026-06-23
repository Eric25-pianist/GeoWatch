"""Integration tests for the resumable professional project orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from geowatch.acquisition.models import DownloadResult, SceneMetadata
from geowatch.application.availability import AvailabilityPlan, YearAvailability
from geowatch.application.models import (
    LocationSpec,
    OutputSpec,
    RunSpecification,
    TemporalSpec,
)
from geowatch.application.project import ProjectLayout, write_run_specification
from geowatch.application.workflow import process_project, project_status
from geowatch.processing.models import RasterGrid, RasterLayer


def test_professional_workflow_runs_and_resumes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A completed project should resume without repeating material stages."""
    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"Polygon","coordinates":[[[74,31],[75,31],[75,32],[74,32],[74,31]]]}',
        encoding="utf-8",
    )
    spec = RunSpecification(
        location=LocationSpec(
            name="Test City",
            country="Test Country",
            boundary_path=boundary,
        ),
        temporal=TemporalSpec(start_year=2018, end_year=2020),
        outputs=OutputSpec(root=tmp_path / "outputs"),
    )
    layout = ProjectLayout.from_spec(spec)
    project_file = write_run_specification(spec, layout)
    calls = {"acquire": 0, "process": 0, "publish": 0}
    availability = AvailabilityPlan(
        dataset="sentinel-2-l2a",
        requested_start_month=6,
        requested_end_month=9,
        requested_cloud_cover=20.0,
        effective_start_month=6,
        effective_end_month=9,
        effective_cloud_cover=20.0,
        minimum_scenes_per_year=1,
        years={
            year: YearAvailability(
                year=year,
                scene_ids=(f"scene-{year}",),
                scene_count=1,
                cloud_cover=(5.0,),
                acquired_dates=(f"{year}-07-01",),
            )
            for year in (2018, 2020)
        },
    )

    def fake_acquisition(config: Any, **_: Any) -> Any:
        calls["acquire"] += 1
        config.acquisition.metadata_catalog.parent.mkdir(parents=True, exist_ok=True)
        config.acquisition.metadata_catalog.write_text(
            '{"scenes": [], "downloads": []}', encoding="utf-8"
        )
        config.acquisition.acquisition_report.write_text("report", encoding="utf-8")
        download_path = config.acquisition.download_directory / "asset.tif"
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"data")
        download = DownloadResult(
            scene_id="scene",
            asset_name="blue",
            path=download_path,
            bytes_written=4,
            verified=True,
        )
        return SimpleNamespace(
            downloads=(download,),
            catalog_path=config.acquisition.metadata_catalog,
            report_path=config.acquisition.acquisition_report,
            provider="planetary-computer",
        )

    grid = RasterGrid(
        crs="EPSG:32643",
        transform=(30.0, 0.0, 300000.0, 0.0, -30.0, 3500000.0),
        width=4,
        height=4,
        band_names=("blue", "green", "red", "nir", "swir1", "swir2"),
        nodata=np.nan,
    )
    layer = RasterLayer(
        name="scene",
        data=np.ones((6, 4, 4), dtype=np.float32),
        grid=grid,
        cloud_mask=np.zeros((4, 4), dtype=bool),
    )

    def fake_composite(*_: Any, output_path: Path, **__: Any) -> RasterLayer:
        calls["process"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"raster")
        result = layer.with_data(layer.data.copy())
        result.metadata["output_path"] = str(output_path)
        return result

    def fake_outputs(*_: Any, **__: Any) -> dict[str, Path]:
        calls["publish"] += 1
        reports = layout.root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        outputs = {
            "html_report": reports / "report.html",
            "pdf_report": reports / "report.pdf",
            "validation_report": reports / "validation_report.md",
            "dashboard": reports / "dashboard.html",
        }
        for path in outputs.values():
            path.write_text("output", encoding="utf-8")
        return outputs

    def fake_analytics(*_: Any, **__: Any) -> Any:
        return SimpleNamespace()

    def fake_maps(*_: Any, **__: Any) -> dict[str, object]:
        return {}

    def fake_availability(*_: Any, **__: Any) -> AvailabilityPlan:
        return availability

    monkeypatch.setattr(
        "geowatch.application.workflow.run_acquisition", fake_acquisition
    )
    monkeypatch.setattr(
        "geowatch.application.workflow._prepare_availability",
        fake_availability,
    )
    monkeypatch.setattr(
        "geowatch.application.workflow.build_year_composite", fake_composite
    )
    monkeypatch.setattr(
        "geowatch.application.workflow.load_processed_composite", lambda _: layer
    )
    monkeypatch.setattr(
        "geowatch.application.workflow.run_analytics_pipeline",
        fake_analytics,
    )
    monkeypatch.setattr(
        "geowatch.application.workflow.render_cartography_suite",
        fake_maps,
    )
    monkeypatch.setattr(
        "geowatch.application.workflow.write_professional_outputs", fake_outputs
    )
    monkeypatch.setattr(
        "geowatch.application.workflow._read_sources",
        lambda _: (
            SceneMetadata(
                scene_id="scene",
                provider="planetary-computer",
                dataset="sentinel-2-l2a",
                acquired_at=datetime(2020, 7, 1, tzinfo=UTC),
                assets=(),
            ),
        ),
    )

    result = process_project(project_file)
    assert result == layout.root
    assert calls == {"acquire": 2, "process": 2, "publish": 1}
    assert "publication:2018-2020: completed" in project_status(project_file)

    process_project(project_file, resume=True)
    assert calls == {"acquire": 2, "process": 2, "publish": 1}
