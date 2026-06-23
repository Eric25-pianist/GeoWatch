"""Unit tests for STAC search normalization."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from geowatch.acquisition.http import HTTPClient, HTTPResponse
from geowatch.acquisition.models import SearchQuery
from geowatch.acquisition.stac import STACClient, normalize_stac_item


class FakeHTTPClient(HTTPClient):
    """Deterministic HTTP client for acquisition tests."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

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
        self.calls.append((method, url))
        return HTTPResponse(
            status_code=200,
            headers={},
            content=json.dumps(self.payload).encode("utf-8"),
        )


def test_normalize_stac_item() -> None:
    """STAC items should normalize into scene metadata."""
    item: dict[str, Any] = {
        "id": "scene-1",
        "collection": "sentinel-2-l2a",
        "bbox": [1, 2, 3, 4],
        "properties": {
            "datetime": "2024-01-02T03:04:05Z",
            "eo:cloud_cover": 7.5,
        },
        "assets": {
            "B04": {
                "href": "https://example.com/b04.tif",
                "roles": ["data"],
                "type": "image/tiff",
            }
        },
        "links": [{"rel": "self", "href": "https://example.com/item"}],
    }

    scene = normalize_stac_item(
        item,
        provider="copernicus",
        dataset_map={"sentinel-2-l2a": "sentinel-2-l2a"},
    )

    assert scene.scene_id == "scene-1"
    assert scene.cloud_cover == 7.5
    assert scene.acquired_at is not None
    assert scene.assets[0].name == "B04"


def test_stac_client_search_returns_scenes() -> None:
    """The STAC client should search and normalize returned features."""
    payload: dict[str, Any] = {
        "features": [
            {
                "id": "scene-2",
                "collection": "sentinel-2-l2a",
                "bbox": [1, 2, 3, 4],
                "properties": {"datetime": "2024-01-02T03:04:05Z"},
                "assets": {},
                "links": [],
            }
        ]
    }
    client = STACClient(
        "https://example.test",
        provider="copernicus",
        http_client=FakeHTTPClient(payload),
    )
    query = SearchQuery(
        bbox=(1, 2, 3, 4),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        datasets=("sentinel-2-l2a",),
    )

    scenes = client.search(
        query,
        collections=("sentinel-2-l2a",),
        dataset_map={"sentinel-2-l2a": "sentinel-2-l2a"},
    )

    assert scenes[0].scene_id == "scene-2"
