"""Unit tests for the Phase 2 acquisition pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from geowatch.acquisition.http import AcquisitionError
from geowatch.acquisition.models import AssetMetadata, SceneMetadata
from geowatch.acquisition.pipeline import run_acquisition
from geowatch.config.models import AOIConfig, DateRangeConfig, ProjectConfig


class StubProvider:
    """Provider stub used to exercise orchestration."""

    name = "copernicus"

    def __init__(self, scenes: tuple[SceneMetadata, ...]) -> None:
        self._scenes = scenes

    def search(self, _query: object) -> tuple[SceneMetadata, ...]:
        return self._scenes

    def metadata(self, _scene_id: str) -> SceneMetadata:
        return self._scenes[0]


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        project_name="phase-two",
        aoi=AOIConfig(kind="bbox", bbox=(1, 2, 3, 4)),
        dates=DateRangeConfig(
            start_date=datetime(2024, 1, 1).date(),
            end_date=datetime(2024, 1, 31).date(),
        ),
    )


def test_run_acquisition_with_stub_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The acquisition pipeline should orchestrate search and write outputs."""
    config = _config(tmp_path)
    config.acquisition.download = False
    config.acquisition.metadata_catalog = tmp_path / "catalog.json"
    config.acquisition.acquisition_report = tmp_path / "report.md"

    asset = AssetMetadata(
        name="B04",
        href="https://example.com/b04.tif",
        roles=("data",),
    )
    scene = SceneMetadata(
        scene_id="scene-5",
        provider="copernicus",
        dataset="sentinel-2-l2a",
        acquired_at=datetime.now(UTC),
        assets=(asset,),
    )

    monkeypatch.setattr(
        "geowatch.acquisition.pipeline.build_provider",
        lambda *_args, **_kwargs: StubProvider((scene,)),
    )

    result = run_acquisition(config)

    assert result.scenes[0].scene_id == "scene-5"
    assert result.catalog_path.exists()


def test_run_acquisition_disabled_raises(tmp_path: Path) -> None:
    """Disabled acquisition should fail fast."""
    config = _config(tmp_path)
    config.acquisition.enabled = False

    with pytest.raises(AcquisitionError, match="disabled"):
        run_acquisition(config)


def test_run_acquisition_with_geojson_aoi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The acquisition pipeline should resolve GeoJSON AOI extents."""
    config = _config(tmp_path)
    config.aoi = AOIConfig(
        kind="geojson",
        path=Path("sample_aoi.geojson"),
        crs="EPSG:4326",
    )
    config.acquisition.download = False
    config.acquisition.metadata_catalog = tmp_path / "catalog.json"
    config.acquisition.acquisition_report = tmp_path / "report.md"

    captured: dict[str, object] = {}

    class GeoJSONProvider(StubProvider):
        def search(self, query: object) -> tuple[SceneMetadata, ...]:
            captured["query"] = query
            return super().search(query)

    asset = AssetMetadata(
        name="B04",
        href="https://example.com/b04.tif",
        roles=("data",),
    )
    scene = SceneMetadata(
        scene_id="scene-geojson",
        provider="copernicus",
        dataset="sentinel-2-l2a",
        acquired_at=datetime.now(UTC),
        assets=(asset,),
    )

    monkeypatch.setattr(
        "geowatch.acquisition.pipeline.build_provider",
        lambda *_args, **_kwargs: GeoJSONProvider((scene,)),
    )

    result = run_acquisition(config, base_dir=Path("tests/fixtures/sample_data"))

    assert result.scenes[0].scene_id == "scene-geojson"
    assert "query" in captured
