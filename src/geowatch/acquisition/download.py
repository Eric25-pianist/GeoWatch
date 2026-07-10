"""Download manager with checksum and completeness verification."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from loguru import logger

try:
    from planetary_computer import sign_url as planetary_sign_url
except ImportError:  # pragma: no cover - validated by production doctor
    planetary_sign_url = None

from geowatch.acquisition.http import AcquisitionError, HTTPClient
from geowatch.acquisition.models import (
    AssetMetadata,
    DownloadRequest,
    DownloadResult,
    SceneMetadata,
)
from geowatch.acquisition.retry import RetryPolicy, retry_call
from geowatch.utils.paths import ensure_parent


class DownloadManager:
    """Download and verify scene assets."""

    def __init__(
        self,
        *,
        http_client: HTTPClient | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.http_client = http_client
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout = timeout

    def download(self, request: DownloadRequest) -> DownloadResult:
        """Download one asset to disk and verify completeness."""
        destination = ensure_parent(request.destination)

        if self.http_client is None:
            return retry_call(
                lambda: self._download_streaming(request, destination),
                self.retry_policy,
            )
        return self._download_buffered(request, destination)

    def _download_buffered(
        self,
        request: DownloadRequest,
        destination: Path,
    ) -> DownloadResult:
        """Download through an injected HTTP client for deterministic tests."""
        if self.http_client is None:  # pragma: no cover - guarded by caller
            raise AcquisitionError("Buffered download requires an HTTP client.")
        client = self.http_client

        def action() -> DownloadResult:
            download_url = _download_url(request)
            response = client.request(
                "GET",
                download_url,
                timeout=self.timeout,
            )
            if not response.ok:
                msg = (
                    f"Download failed for {_safe_url(download_url)}: "
                    f"HTTP {response.status_code}"
                )
                raise AcquisitionError(msg)
            if len(response.content) > request.max_bytes:
                msg = (
                    "Download exceeds configured size limit: "
                    f"{len(response.content)} bytes"
                )
                raise AcquisitionError(msg)
            destination.write_bytes(response.content)
            result = verify_download(
                request.scene,
                request.asset,
                destination,
                response.content,
            )
            logger.info("Downloaded {} to {}", request.asset.name, destination)
            return result

        return retry_call(action, self.retry_policy)

    def _download_streaming(
        self,
        request: DownloadRequest,
        destination: Path,
    ) -> DownloadResult:
        """Stream an asset to an atomic partial file and resume when supported."""
        reusable = _reuse_verified_download(request, destination)
        if reusable is not None:
            logger.info("Reused verified {}", destination)
            return reusable

        partial = destination.with_suffix(destination.suffix + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        download_url = _download_url(request)
        http_request = Request(download_url, headers=headers)
        try:
            response = urlopen(http_request, timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            message = _friendly_streaming_error(download_url, exc)
            raise AcquisitionError(message) from exc

        status = int(getattr(response, "status", response.getcode()))
        append = existing > 0 and status == 206
        if not append:
            existing = 0
        content_length = response.headers.get("Content-Length")
        expected_remaining = int(content_length) if content_length else None
        expected_total = (
            existing + expected_remaining if expected_remaining is not None else None
        )
        if expected_total is not None and expected_total > request.max_bytes:
            response.close()
            raise AcquisitionError(
                f"Download exceeds configured size limit: {expected_total} bytes"
            )

        mode = "ab" if append else "wb"
        bytes_written = existing
        try:
            with response as source, partial.open(mode) as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > request.max_bytes:
                        raise AcquisitionError(
                            "Download exceeds configured size limit while streaming: "
                            f"{bytes_written} bytes"
                        )
                    target.write(chunk)
        except OSError as exc:
            raise AcquisitionError(
                f"Could not write partial download: {partial}"
            ) from exc

        if expected_total is not None and bytes_written != expected_total:
            raise AcquisitionError(
                f"Incomplete streamed download: {bytes_written}/{expected_total} bytes"
            )
        checksum = _verify_path(request, partial)
        partial.replace(destination)
        _write_verification_sidecar(request, destination, checksum)
        logger.info("Downloaded {} to {}", request.asset.name, destination)
        return DownloadResult(
            scene_id=request.scene.scene_id,
            asset_name=request.asset.name,
            path=destination,
            bytes_written=destination.stat().st_size,
            checksum=checksum,
            verified=True,
        )


def build_download_requests(
    scenes: tuple[SceneMetadata, ...],
    *,
    download_directory: Path,
    preferred_roles: tuple[str, ...],
    max_downloads: int,
    max_bytes: int,
) -> tuple[DownloadRequest, ...]:
    """Create bounded download requests for preferred scene assets."""
    requests: list[DownloadRequest] = []
    for scene in scenes:
        for asset in _rank_assets_for_download(scene, preferred_roles):
            safe_name = _safe_asset_name(scene.scene_id, asset)
            requests.append(
                DownloadRequest(
                    scene=scene,
                    asset=asset,
                    destination=download_directory / scene.dataset / safe_name,
                    max_bytes=max_bytes,
                )
            )
            if len(requests) >= max_downloads:
                return tuple(requests)
    return tuple(requests)


def _rank_assets_for_download(
    scene: SceneMetadata,
    preferred_roles: tuple[str, ...],
) -> tuple[AssetMetadata, ...]:
    """Prioritize assets that are useful for NDVI and general analysis."""
    preferred_names: tuple[str, ...]
    if scene.dataset == "sentinel-2-l2a":
        preferred_names = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")
    elif scene.dataset in {"landsat-5-c2-l2", "landsat-7-c2-l2"}:
        preferred_names = (
            "blue",
            "green",
            "red",
            "nir08",
            "swir16",
            "swir22",
            "qa_pixel",
            "qa_radsat",
            "SR_B1",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B7",
            "QA_PIXEL",
            "QA_RADSAT",
        )
    elif scene.dataset in {"landsat-8-c2-l2", "landsat-9-c2-l2"}:
        preferred_names = (
            "blue",
            "green",
            "red",
            "nir08",
            "swir16",
            "swir22",
            "qa_pixel",
            "qa_radsat",
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
            "QA_PIXEL",
            "QA_RADSAT",
        )
    else:
        preferred_names = ()

    ranked: list[AssetMetadata] = []
    seen: set[tuple[str, str]] = set()
    for name in preferred_names:
        for asset in scene.assets:
            key = (asset.name, str(asset.href))
            if key in seen:
                continue
            if asset.name == name:
                ranked.append(asset)
                seen.add(key)
    if ranked:
        return tuple(ranked)
    for asset in scene.preferred_assets(preferred_roles):
        key = (asset.name, str(asset.href))
        if key not in seen:
            ranked.append(asset)
            seen.add(key)
    return tuple(ranked)


def _safe_url(value: str) -> str:
    """Return a URL without query parameters or fragments for safe logs."""
    parsed = urlparse(value)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _download_url(request: DownloadRequest) -> str:
    """Return a fresh download URL, re-signing Planetary Computer assets if needed."""
    href = str(request.asset.href)
    if request.scene.provider != "planetary-computer":
        return href
    if planetary_sign_url is None:
        raise AcquisitionError(
            "Planetary Computer downloads require the 'planetary-computer' package. "
            "Run setup-micromamba.ps1 and try again."
        )
    return str(planetary_sign_url(_safe_url(href)))


def _friendly_streaming_error(url: str, exc: BaseException) -> str:
    """Explain streaming failures without leaking signed provider URLs."""
    safe_url = _safe_url(url)
    reason = exc.reason if isinstance(exc, URLError) else exc
    if _is_dns_error(reason):
        host = urlparse(url).netloc or "the imagery provider host"
        return (
            f"Could not resolve imagery asset host '{host}' while downloading "
            f"{safe_url}. Check internet/DNS/VPN/firewall settings, then run "
            "`geowatch resume <project.yaml>`; verified files will be reused."
        )
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
        return (
            f"Streaming download timed out for {safe_url}. Retry with "
            "`geowatch resume <project.yaml>`; verified files will be reused."
        )
    return f"Streaming download failed for {safe_url}: {reason}"


def _is_dns_error(reason: object) -> bool:
    """Return True for Windows and cross-platform DNS resolution failures."""
    return isinstance(reason, socket.gaierror) or getattr(reason, "winerror", None) in {
        11001,
        11002,
        11004,
    }


def verify_download(
    scene: SceneMetadata,
    asset: AssetMetadata,
    path: Path,
    content: bytes,
) -> DownloadResult:
    """Verify file existence, non-empty content, and optional checksum."""
    if not path.exists():
        raise AcquisitionError(f"Downloaded file does not exist: {path}")
    bytes_written = path.stat().st_size
    if bytes_written == 0:
        raise AcquisitionError(f"Downloaded file is empty: {path}")
    if bytes_written != len(content):
        raise AcquisitionError(f"Downloaded file size mismatch: {path}")

    checksum = None
    if asset.checksum and asset.checksum_algorithm:
        checksum = _digest(content, asset.checksum_algorithm)
        if checksum.lower() != asset.checksum.lower():
            raise AcquisitionError(f"Checksum mismatch for {path}")
    return DownloadResult(
        scene_id=scene.scene_id,
        asset_name=asset.name,
        path=path,
        bytes_written=bytes_written,
        checksum=checksum,
        verified=True,
    )


def _reuse_verified_download(
    request: DownloadRequest,
    destination: Path,
) -> DownloadResult | None:
    """Reuse a complete asset when size/checksum or its sidecar verifies it."""
    if not destination.exists() or destination.stat().st_size == 0:
        return None
    sidecar = _verification_sidecar(destination)
    sidecar_valid = False
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_valid = (
                payload.get("scene_id") == request.scene.scene_id
                and payload.get("asset_name") == request.asset.name
                and payload.get("bytes_written") == destination.stat().st_size
            )
        except (OSError, json.JSONDecodeError):
            sidecar_valid = False
    size_valid = request.asset.size is not None and (
        destination.stat().st_size == request.asset.size
    )
    raster_valid = _legacy_raster_is_valid(request, destination)
    if not sidecar_valid and not size_valid and not raster_valid:
        return None
    checksum = _verify_path(request, destination)
    if not sidecar_valid:
        _write_verification_sidecar(request, destination, checksum)
    return DownloadResult(
        scene_id=request.scene.scene_id,
        asset_name=request.asset.name,
        path=destination,
        bytes_written=destination.stat().st_size,
        checksum=checksum,
        verified=True,
    )


def _legacy_raster_is_valid(
    request: DownloadRequest,
    destination: Path,
) -> bool:
    """Validate a pre-sidecar GeoTIFF before adopting it into resume state."""
    if destination.suffix.casefold() not in {".tif", ".tiff"}:
        return False
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(destination) as dataset:
            if dataset.count < 1 or dataset.crs is None:
                return False
            if dataset.width < 1 or dataset.height < 1:
                return False
            bounds = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False
    if request.scene.bbox is None:
        return False
    west, south, east, north = request.scene.bbox
    overlaps = not (
        bounds[2] <= west
        or bounds[0] >= east
        or bounds[3] <= south
        or bounds[1] >= north
    )
    if overlaps:
        logger.info(
            "Adopting readable legacy raster into verified resume state: {}",
            destination,
        )
    return overlaps


def _verify_path(request: DownloadRequest, path: Path) -> str:
    """Verify a streamed file and return a deterministic checksum."""
    if not path.exists() or path.stat().st_size == 0:
        raise AcquisitionError(f"Downloaded file is empty or missing: {path}")
    if request.asset.size is not None and path.stat().st_size != request.asset.size:
        raise AcquisitionError(
            f"Downloaded file size mismatch for {path}: "
            f"{path.stat().st_size}/{request.asset.size}"
        )
    algorithm = request.asset.checksum_algorithm or "sha256"
    checksum = _digest_path(path, algorithm)
    if request.asset.checksum and checksum.lower() != request.asset.checksum.lower():
        raise AcquisitionError(f"Checksum mismatch for {path}")
    return checksum


def _write_verification_sidecar(
    request: DownloadRequest,
    destination: Path,
    checksum: str,
) -> None:
    payload = {
        "scene_id": request.scene.scene_id,
        "asset_name": request.asset.name,
        "bytes_written": destination.stat().st_size,
        "checksum": checksum,
        "checksum_algorithm": request.asset.checksum_algorithm or "sha256",
    }
    _verification_sidecar(destination).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _verification_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".verified.json")


def _digest_path(path: Path, algorithm: str) -> str:
    hasher = (
        hashlib.sha256()
        if algorithm == "sha256"
        else hashlib.md5(usedforsecurity=False)
    )
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest(content: bytes, algorithm: str) -> str:
    if algorithm == "sha256":
        return hashlib.sha256(content).hexdigest()
    if algorithm == "md5":
        return hashlib.md5(content, usedforsecurity=False).hexdigest()
    raise AcquisitionError(f"Unsupported checksum algorithm: {algorithm}")


def _safe_asset_name(scene_id: str, asset: AssetMetadata) -> str:
    parsed = urlparse(str(asset.href))
    suffix = Path(parsed.path).suffix or ".bin"
    safe_scene = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in scene_id
    )
    safe_asset = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in asset.name
    )
    return f"{safe_scene}_{safe_asset}{suffix}"
