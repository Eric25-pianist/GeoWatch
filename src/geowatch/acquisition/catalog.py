"""Acquisition catalog and report writers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from geowatch.acquisition.models import DownloadResult, ProviderName, SceneMetadata
from geowatch.core.errors import GeoWatchError
from geowatch.utils.paths import ensure_parent


class CatalogError(GeoWatchError):
    """Raised when acquisition catalog or report writing fails."""


def write_metadata_catalog(
    scenes: tuple[SceneMetadata, ...],
    downloads: tuple[DownloadResult, ...],
    path: Path,
    *,
    provider: ProviderName,
) -> Path:
    """Write normalized scene metadata and download records to JSON."""
    payload = {
        "phase": 2,
        "provider": provider,
        "generated_at": datetime.now(UTC).isoformat(),
        "scene_count": len(scenes),
        "download_count": len(downloads),
        "scenes": [scene.model_dump(mode="json") for scene in scenes],
        "downloads": [download.model_dump(mode="json") for download in downloads],
    }
    destination = ensure_parent(path)
    try:
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to write acquisition catalog {}", destination)
        msg = f"Could not write acquisition catalog: {destination}"
        raise CatalogError(msg) from exc
    return destination


def write_acquisition_report(
    scenes: tuple[SceneMetadata, ...],
    downloads: tuple[DownloadResult, ...],
    path: Path,
    *,
    provider: ProviderName,
) -> Path:
    """Write a concise acquisition report."""
    lines = [
        "# GeoWatch Acquisition Report",
        "",
        "- Phase: 2 - Data Acquisition",
        f"- Provider: {provider}",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Scenes: {len(scenes)}",
        f"- Downloads: {len(downloads)}",
        "",
        "## Scenes",
        "",
    ]
    for scene in scenes:
        lines.append(
            f"- `{scene.scene_id}` {scene.dataset} cloud={scene.cloud_cover} "
            f"assets={len(scene.assets)}"
        )
    if downloads:
        lines.extend(["", "## Downloads", ""])
        for download in downloads:
            lines.append(
                f"- `{download.scene_id}` `{download.asset_name}` "
                f"{download.bytes_written} bytes verified={download.verified}"
            )
    destination = ensure_parent(path)
    try:
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to write acquisition report {}", destination)
        msg = f"Could not write acquisition report: {destination}"
        raise CatalogError(msg) from exc
    return destination
