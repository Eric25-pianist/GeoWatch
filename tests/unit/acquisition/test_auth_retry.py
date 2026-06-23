"""Unit tests for acquisition auth and retry helpers."""

from __future__ import annotations

import pytest

from geowatch.acquisition.auth import AuthManager
from geowatch.acquisition.http import AcquisitionError
from geowatch.acquisition.retry import RetryPolicy, retry_call


def test_auth_manager_reads_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider tokens should come from environment variables."""
    monkeypatch.setenv("GEOWATCH_COPERNICUS_TOKEN", "secret-token")

    credentials = AuthManager().credentials_for("copernicus")

    assert credentials.token == "secret-token"
    assert AuthManager().authorization_headers("copernicus") == {
        "Authorization": "Bearer secret-token"
    }


def test_retry_call_retries_and_raises() -> None:
    """Retry policy should call the action again and eventually fail."""
    attempts: list[int] = []

    def action() -> str:
        attempts.append(1)
        raise AcquisitionError("boom")

    with pytest.raises(AcquisitionError, match="Operation failed after 2 attempts"):
        retry_call(action, RetryPolicy(attempts=2, backoff_seconds=0))

    assert len(attempts) == 2
