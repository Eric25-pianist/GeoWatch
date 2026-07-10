"""Small HTTP abstraction used by acquisition clients and tests."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from geowatch.core.errors import GeoWatchError


class AcquisitionError(GeoWatchError):
    """Raised when acquisition operations fail."""


class NonRetryableAcquisitionError(AcquisitionError):
    """Raised when retrying the same acquisition request cannot help."""


class HTTPClient(Protocol):
    """Protocol implemented by HTTP clients used in acquisition."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
        timeout: float = 30.0,
    ) -> HTTPResponse:
        """Execute an HTTP request and return a response."""


@dataclass(frozen=True)
class HTTPResponse:
    """Minimal HTTP response."""

    status_code: int
    headers: dict[str, str]
    content: bytes

    def json(self) -> dict[str, object]:
        """Decode the response body as a JSON object."""
        data = json.loads(self.content.decode("utf-8"))
        if not isinstance(data, dict):
            raise AcquisitionError("Expected a JSON object response.")
        return data

    @property
    def ok(self) -> bool:
        """Return whether the response status code is successful."""
        return 200 <= self.status_code < 300


class UrllibHTTPClient:
    """HTTP client implemented with the Python standard library."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
        timeout: float = 30.0,
    ) -> HTTPResponse:
        """Execute an HTTP request with JSON support."""
        body = None
        request_headers = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request_headers.setdefault("User-Agent", "GeoWatch/0.1")
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()
                logger.debug("HTTP {} {} -> {}", method, url, response.status)
                return HTTPResponse(
                    status_code=int(response.status),
                    headers=dict(response.headers.items()),
                    content=content,
                )
        except urllib.error.HTTPError as exc:
            content = exc.read()
            logger.warning("HTTP {} {} failed with {}", method, url, exc.code)
            return HTTPResponse(
                status_code=int(exc.code),
                headers=dict(exc.headers.items()),
                content=content,
            )
        except urllib.error.URLError as exc:
            message = _friendly_url_error(url, exc)
            logger.error(message)
            raise AcquisitionError(message) from exc


def _friendly_url_error(url: str, exc: urllib.error.URLError) -> str:
    """Translate low-level URL errors into terminal-friendly acquisition messages."""
    reason = exc.reason
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return (
            f"HTTP request timed out for {url}. Check your internet connection or "
            "try a different imagery provider."
        )
    if getattr(reason, "winerror", None) == 10060:
        return (
            f"HTTP request timed out for {url}. The remote provider did not respond "
            "in time."
        )
    if (
        isinstance(reason, socket.gaierror)
        or getattr(reason, "winerror", None) == 11002
    ):
        return (
            f"Could not resolve the imagery provider host for {url}. Check that the "
            "PC is connected to the internet, DNS is working, and firewall/VPN "
            "settings allow access to the provider."
        )
    if reason:
        return f"HTTP request failed for {url}: {reason}"
    return f"HTTP request failed for {url}: {exc}"
