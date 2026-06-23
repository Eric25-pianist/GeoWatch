"""Beginner-friendly wizard input parsing tests."""

from __future__ import annotations

import pytest

from geowatch.application.wizard import parse_year_range


def test_parse_compact_year_range() -> None:
    """The first year prompt should accept a complete comparison range."""
    assert parse_year_range("2018-2019") == (2018, 2019)
    assert parse_year_range("2018") == (2018, None)


def test_parse_year_range_rejects_reversed_period() -> None:
    """A reversed or equal range should provide an actionable validation error."""
    with pytest.raises(ValueError, match="later"):
        parse_year_range("2020-2018")
