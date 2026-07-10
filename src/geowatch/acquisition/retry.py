"""Retry and exponential backoff utilities for acquisition requests."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from geowatch.acquisition.http import AcquisitionError, NonRetryableAcquisitionError


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy with bounded exponential backoff."""

    attempts: int = 3
    backoff_seconds: float = 0.5
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)


def retry_call[T](action: Callable[[], T], policy: RetryPolicy) -> T:
    """Run ``action`` with retries for acquisition errors."""
    last_error: Exception | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return action()
        except NonRetryableAcquisitionError:
            raise
        except AcquisitionError as exc:
            last_error = exc
            logger.warning(
                "Acquisition attempt {}/{} failed: {}",
                attempt,
                policy.attempts,
                exc,
            )
            if attempt < policy.attempts and policy.backoff_seconds > 0:
                time.sleep(policy.backoff_seconds * (2 ** (attempt - 1)))
    detail = f": {last_error}" if last_error is not None else "."
    msg = f"Operation failed after {policy.attempts} attempts{detail}"
    raise AcquisitionError(msg) from last_error
