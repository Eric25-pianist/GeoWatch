"""Unit tests for the Phase 2 acquisition pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from geowatch.acquisition.http import AcquisitionError
from geowatch.acquisition.models import AssetMetadata, DownloadResult, SceneMetadata
from geowatch.acquisition.pipeline import _download_if_requested, run_acquisition
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


def test_selected_scene_ids_pin_auto_provider_to_planetary_computer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selected STAC scene IDs should be fetched from their planning provider."""
    config = _config(tmp_path)
    config.acquisition.download = False
    config.acquisition.provider = "auto"
    config.acquisition.selected_scene_ids = ("scene-selected",)
    config.acquisition.metadata_catalog = tmp_path / "catalog.json"
    config.acquisition.acquisition_report = tmp_path / "report.md"
    providers: list[str] = []

    asset = AssetMetadata(
        name="B04",
        href="https://example.com/b04.tif",
        roles=("data",),
    )
    scene = SceneMetadata(
        scene_id="scene-selected",
        provider="planetary-computer",
        dataset="sentinel-2-l2a",
        acquired_at=datetime.now(UTC),
        assets=(asset,),
    )

    def provider_factory(
        provider_name: str, *_args: object, **_kwargs: object
    ) -> object:
        providers.append(provider_name)

        class SelectedProvider(StubProvider):
            pass

        SelectedProvider.name = provider_name

        return SelectedProvider((scene,))

    monkeypatch.setattr(
        "geowatch.acquisition.pipeline.build_provider",
        provider_factory,
    )

    result = run_acquisition(config)

    assert result.provider == "planetary-computer"
    assert providers == ["planetary-computer"]


def test_download_if_requested_skips_incomplete_scene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One failed scene should not discard other complete scenes."""
    config = _config(tmp_path)
    config.acquisition.download = True
    config.acquisition.download_directory = tmp_path / "assets"
    config.acquisition.max_downloads = 4
    config.acquisition.retry_attempts = 5
    config.acquisition.retry_backoff_seconds = 2.0
    captured_policy: dict[str, float | int] = {}

    def scene(scene_id: str) -> SceneMetadata:
        return SceneMetadata(
            scene_id=scene_id,
            provider="planetary-computer",
            dataset="sentinel-2-l2a",
            acquired_at=datetime.now(UTC),
            assets=(
                AssetMetadata(
                    name="B02",
                    href=f"https://example.com/{scene_id}/B02.tif",
                    roles=("data",),
                ),
                AssetMetadata(
                    name="B03",
                    href=f"https://example.com/{scene_id}/B03.tif",
                    roles=("data",),
                ),
            ),
        )

    class SceneAwareManager:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            retry_policy = kwargs["retry_policy"]
            captured_policy["attempts"] = retry_policy.attempts
            captured_policy["backoff"] = retry_policy.backoff_seconds

        def download(self, request: Any) -> DownloadResult:
            scene_id = request.scene.scene_id
            asset_name = request.asset.name
            path = tmp_path / f"{scene_id}_{asset_name}.tif"
            if scene_id == "bad" and asset_name == "B03":
                raise AcquisitionError("temporary DNS failure")
            path.write_bytes(b"ok")
            return DownloadResult(
                scene_id=scene_id,
                asset_name=asset_name,
                path=path,
                bytes_written=path.stat().st_size,
                verified=True,
            )

    monkeypatch.setattr(
        "geowatch.acquisition.pipeline.DownloadManager",
        SceneAwareManager,
    )

    downloads = _download_if_requested(
        config,
        (scene("good"), scene("bad")),
        http_client=None,
    )

    assert {download.scene_id for download in downloads} == {"good"}
    assert captured_policy == {"attempts": 5, "backoff": 2.0}
