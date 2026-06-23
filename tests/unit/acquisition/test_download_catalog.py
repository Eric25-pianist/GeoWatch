"""Unit tests for download manager and catalog writers."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from geowatch.acquisition.catalog import (
    write_acquisition_report,
    write_metadata_catalog,
)
from geowatch.acquisition.download import (
    DownloadManager,
    build_download_requests,
    verify_download,
)
from geowatch.acquisition.http import HTTPResponse
from geowatch.acquisition.models import AssetMetadata, DownloadRequest, SceneMetadata
from geowatch.acquisition.retry import RetryPolicy


class FakeDownloadHTTPClient:
    """Return bytes for download URLs."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
        timeout: float = 30.0,
    ) -> HTTPResponse:
        _ = (method, url, headers, json_body, timeout)
        return HTTPResponse(200, {}, self.content)


def _scene_and_asset(content: bytes, tmp_path: Path) -> DownloadRequest:
    checksum = hashlib.sha256(content).hexdigest()
    asset = AssetMetadata(
        name="B04",
        href="https://example.com/b04.tif",
        roles=("data",),
        checksum=checksum,
        checksum_algorithm="sha256",
        size=len(content),
    )
    scene = SceneMetadata(
        scene_id="scene-4",
        provider="copernicus",
        dataset="sentinel-2-l2a",
        acquired_at=datetime.now(UTC),
        assets=(asset,),
    )
    return DownloadRequest(scene=scene, asset=asset, destination=tmp_path / "b04.tif")


def test_download_manager_downloads_and_verifies(tmp_path: Path) -> None:
    """Downloads should be written and checksum-verified."""
    content = b"hello world"
    request = _scene_and_asset(content, tmp_path)
    manager = DownloadManager(
        http_client=FakeDownloadHTTPClient(content),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )

    result = manager.download(request)

    assert result.verified
    assert result.bytes_written == len(content)


def test_streaming_download_resumes_and_reuses_verified_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default downloads should resume partial files and later reuse them."""
    content = b"hello world"
    request = _scene_and_asset(content, tmp_path)
    partial = request.destination.with_suffix(".tif.part")
    partial.write_bytes(b"hello ")

    class FakeResponse:
        status = 206

        def __init__(self) -> None:
            self.stream = io.BytesIO(b"world")
            self.headers = {"Content-Length": "5"}

        def getcode(self) -> int:
            return self.status

        def read(self, size: int = -1) -> bytes:
            return self.stream.read(size)

        def close(self) -> None:
            self.stream.close()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    monkeypatch.setattr(
        "geowatch.acquisition.download.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    manager = DownloadManager(retry_policy=RetryPolicy(attempts=1, backoff_seconds=0))
    result = manager.download(request)

    assert result.verified
    assert request.destination.read_bytes() == content
    assert request.destination.with_suffix(".tif.verified.json").exists()

    def unexpected_request(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("verified asset should not be downloaded again")

    monkeypatch.setattr("geowatch.acquisition.download.urlopen", unexpected_request)
    reused = manager.download(request)
    assert reused.verified


def test_build_download_requests_limits_results(tmp_path: Path) -> None:
    """Download request builder should respect the max downloads setting."""
    content = b"hello world"
    request = _scene_and_asset(content, tmp_path)
    requests = build_download_requests(
        (request.scene,),
        download_directory=tmp_path,
        preferred_roles=("data",),
        max_downloads=1,
        max_bytes=100,
    )

    assert len(requests) == 1


def test_verify_download_rejects_missing_file(tmp_path: Path) -> None:
    """Verification should fail on missing files."""
    request = _scene_and_asset(b"hello", tmp_path)

    path = request.destination
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"hello")
    verified = verify_download(request.scene, request.asset, path, b"hello")

    assert verified.verified


def test_catalog_writers_create_files(tmp_path: Path) -> None:
    """Catalog and report writers should emit JSON and Markdown outputs."""
    request = _scene_and_asset(b"hello", tmp_path)
    request.destination.parent.mkdir(parents=True, exist_ok=True)
    request.destination.write_bytes(b"hello")
    download_result = verify_download(
        request.scene,
        request.asset,
        request.destination,
        b"hello",
    )
    catalog = write_metadata_catalog(
        (request.scene,),
        (download_result,),
        tmp_path / "catalog.json",
        provider="copernicus",
    )
    report = write_acquisition_report(
        (request.scene,),
        (download_result,),
        tmp_path / "report.md",
        provider="copernicus",
    )

    assert json.loads(catalog.read_text(encoding="utf-8"))["phase"] == 2
    assert "GeoWatch Acquisition Report" in report.read_text(encoding="utf-8")
