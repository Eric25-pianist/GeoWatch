"""Write the GeoWatch sample AOI GeoJSON file."""

from __future__ import annotations

from pathlib import Path

from geowatch.core.initializer import initialize_project


def main() -> None:
    """Create sample AOI assets in the current project."""
    initialize_project(Path(), overwrite=False)


if __name__ == "__main__":
    main()
