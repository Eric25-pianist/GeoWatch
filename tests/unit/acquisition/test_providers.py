"""Unit tests for acquisition provider connectors."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from geowatch.acquisition.http import HTTPResponse
from geowatch.acquisition.models import AcquisitionConfig, SearchQuery
from geowatch.acquisition.providers.stac_provider import STACProvider
from geowatch.acquisition.providers.usgs import USGSProvider
from geowatch.acquisition.selector import (
    build_provider,
    choose_datasets,
    rank_providers,
)


class FakeHTTPClient:
    """Minimal fake HTTP client."""

    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.responses = responses
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
        _ = (method, headers, json_body, timeout)
        self.calls.append((method, url))
        payload = self.responses[url]
        return HTTPResponse(200, {}, json.dumps(payload).encode("utf-8"))


def test_selector_chooses_sentinel_after_2015() -> None:
    """Recent timelines should prefer Sentinel-2."""
    datasets = choose_datasets(date(2024, 1, 1), ("sentinel-2-l2a", "landsat-8-c2-l2"))

    assert datasets == ("sentinel-2-l2a",)
    assert rank_providers(datasets)[0] == "planetary-computer"


def test_selector_builds_copernicus_provider() -> None:
    """Provider factory should return a concrete connector."""
    provider = build_provider(
        "copernicus",
        AcquisitionConfig(),
    )

    assert isinstance(provider, STACProvider)


def test_usgs_provider_metadata_and_search() -> None:
    """USGS provider should normalize search results."""
    payload: dict[str, Any] = {
        "data": {
            "results": [
                {
                    "entityId": "scene-3",
                    "acquisitionDate": "2024-01-02",
                    "cloudCover": 3.5,
                    "spatialBounds": {"minX": 1, "minY": 2, "maxX": 3, "maxY": 4},
                }
            ]
        }
    }
    provider = USGSProvider(
        http_client=FakeHTTPClient(
            {"https://m2m.cr.usgs.gov/api/api/json/stable/scene-search": payload}
        )
    )
    query = SearchQuery(
        bbox=(1, 2, 3, 4),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        datasets=("landsat-8-c2-l2",),
    )

    scenes = provider.search(query)

    assert scenes[0].scene_id == "scene-3"
    assert provider.metadata("scene-3").scene_id == "scene-3"
