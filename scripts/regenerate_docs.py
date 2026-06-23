"""Regenerate generated foundation documentation artifacts."""

from __future__ import annotations

from pathlib import Path

from geowatch.config.schema import write_json_schema


def main() -> None:
    """Regenerate the configuration JSON schema."""
    write_json_schema(Path("configs/schemas/pipeline.schema.json"))


if __name__ == "__main__":
    main()
