"""Environment diagnostics for real GeoWatch processing."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class DoctorCheck:
    """One environment diagnostic result."""

    name: str
    ok: bool
    detail: str


def run_doctor() -> tuple[DoctorCheck, ...]:
    """Inspect the active interpreter and production GIS dependencies."""
    checks = [
        DoctorCheck(
            "python",
            sys.version_info[:2] == (3, 12),
            f"{sys.version.split()[0]} at {Path(sys.executable).resolve()}",
        ),
        _pip_check(),
    ]
    modules = (
        ("rasterio", "rasterio"),
        ("GDAL", "osgeo.gdal"),
        ("GeoPandas", "geopandas"),
        ("Shapely", "shapely"),
        ("PyProj", "pyproj"),
        ("GeoWatch", "geowatch"),
    )
    for label, module_name in modules:
        checks.append(_module_check(label, module_name))
    logger.info(
        "GeoWatch doctor completed with ok={}", all(check.ok for check in checks)
    )
    return tuple(checks)


def _pip_check() -> DoctorCheck:
    """Verify that pip belongs to the active GeoWatch interpreter."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning("Doctor pip check failed: {}", exc)
        return DoctorCheck("pip", False, f"not runnable: {exc}")
    detail = result.stdout.strip()
    expected_prefix = str(Path(sys.prefix).resolve()).lower()
    ok = expected_prefix in detail.lower()
    return DoctorCheck("pip", ok, detail)


def format_doctor(checks: tuple[DoctorCheck, ...]) -> str:
    """Render diagnostics for a terminal."""
    lines = ["GeoWatch environment doctor"]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def _module_check(label: str, module_name: str) -> DoctorCheck:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.warning("Doctor import failed for {}: {}", module_name, exc)
        return DoctorCheck(label, False, f"not importable: {exc}")
    version = getattr(module, "__version__", None)
    if module_name == "osgeo.gdal":
        version = module.VersionInfo()
    detail = f"version {version}" if version else "importable"
    return DoctorCheck(label, True, detail)
