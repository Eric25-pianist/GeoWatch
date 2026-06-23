"""Fresh-process import-order regression tests."""

from __future__ import annotations

import subprocess
import sys


def test_geometry_imports_before_processing_engine() -> None:
    """Geometry helpers must import cleanly before any processing modules."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from geowatch.utils.geometry import load_vector_geometry",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
