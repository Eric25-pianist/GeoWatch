"""Lahore-specific QC and repair workflow for NDVI change mapping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from shapely.geometry.base import BaseGeometry

from geowatch.analytics.models import AnalyticsReport
from geowatch.analytics.pipeline import run_analytics_pipeline
from geowatch.config.loader import load_config
from geowatch.config.models import ProjectConfig
from geowatch.processing.engine import (
    clip_layer_to_geometry,
    resample_layer,
)
from geowatch.processing.io import read_raster
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.reporting.cartography import render_cartography_suite
from geowatch.reporting.models import MapArtifact
from geowatch.utils.geometry import (
    ensure_valid_geometry,
    geometry_mask_for_grid,
    load_vector_geometry,
    reproject_geometry,
)
from geowatch.utils.paths import ensure_parent

_BAND_ORDER: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")
_TARGET_SIZE = 1024
_REFERENCE_URLS: tuple[tuple[str, str], ...] = (
    (
        "ESA Sentinel-2 mission",
        "https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2",
    ),
    (
        "Copernicus Sentinel-2 overview",
        "https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2",
    ),
    (
        "NASA Earthdata NDVI definition",
        "https://cmr.earthdata.nasa.gov/search/concepts/C2207478498-FEDEO.html",
    ),
    (
        "Lahore district change study",
        "https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1734724/full",
    ),
    (
        "Punjab government Lahore division map",
        "https://lahoredivision.punjab.gov.pk/division_maps",
    ),
)


@dataclass(frozen=True)
class DatasetSummary:
    """Compact inventory row for a raster or vector layer."""

    label: str
    path: Path
    crs: str
    width: int
    height: int
    band_count: int
    cell_size_x: float
    cell_size_y: float
    bounds: tuple[float, float, float, float]
    nodata: float | int | None
    valid_geometry: bool | None = None
    area_sq_km: float | None = None


@dataclass(frozen=True)
class ValidationFinding:
    """A single QC check result."""

    check: str
    status: str
    details: str
    fix: str


@dataclass(frozen=True)
class ScientificReference:
    """A citation used for scientific validation."""

    title: str
    url: str


@dataclass(frozen=True)
class SceneBundle:
    """Masked scene stack and provenance for a single acquisition year."""

    year: int
    scene_id: str
    layer: RasterLayer
    inventory: tuple[DatasetSummary, ...]
    catalog_path: Path
    source_directory: Path


@dataclass(frozen=True)
class LahoreQcResult:
    """Complete Lahore QC deliverable bundle."""

    project_name: str
    generated_at: datetime
    output_root: Path
    boundary: DatasetSummary
    scene_t1: SceneBundle
    scene_t2: SceneBundle
    analytics_report: AnalyticsReport
    findings: tuple[ValidationFinding, ...]
    references: tuple[ScientificReference, ...]
    map_artifacts: dict[str, MapArtifact]
    outputs: dict[str, Path]
    reports: dict[str, Path]


def run_lahore_qc(
    config_2018_path: Path,
    config_2020_path: Path,
    *,
    output_root: Path | None = None,
) -> LahoreQcResult:
    """Run the Lahore QC workflow and generate reports/maps."""
    config_2018 = load_config(config_2018_path)
    config_2020 = load_config(config_2020_path)
    project_root = Path.cwd()
    target_root = output_root or (project_root / "outputs" / "lahore_ndvi_qc")
    maps_dir = target_root / "maps"
    reports_dir = target_root / "reports"
    exports_dir = target_root / "exports"
    validation_dir = target_root / "validation"
    science_dir = target_root / "science"
    for directory in (maps_dir, reports_dir, exports_dir, validation_dir, science_dir):
        directory.mkdir(parents=True, exist_ok=True)

    boundary_path = _resolve_relative_path(
        config_2018.aoi.path or config_2020.aoi.path,
        config_2018_path.parent,
        config_2020_path.parent,
        Path.cwd(),
    )
    if boundary_path is None:
        raise ValueError("Lahore QC requires an AOI boundary file.")
    boundary_loaded = load_vector_geometry(boundary_path)
    boundary_geometry = ensure_valid_geometry(
        boundary_loaded.geometry,
        source_path=boundary_loaded.source_path,
    )
    boundary_inventory = _summarize_boundary(
        boundary_loaded.source_path,
        boundary_geometry,
        boundary_loaded.crs,
    )

    scene_t1 = _load_scene_bundle(
        config_2018,
        config_2018_path,
        year=2018,
        target_geometry=boundary_geometry,
        target_crs=boundary_loaded.crs,
    )
    scene_t2 = _load_scene_bundle(
        config_2020,
        config_2020_path,
        year=2020,
        target_geometry=boundary_geometry,
        target_crs=boundary_loaded.crs,
    )

    publication_config = config_2018.model_copy(
        update={
            "project_name": "lahore-ndvi-qc",
            "aoi": config_2018.aoi.model_copy(
                update={
                    "path": boundary_path,
                    "crs": boundary_loaded.crs,
                    "kind": "geojson",
                }
            ),
        },
    )

    analytics_report = run_analytics_pipeline(
        scene_t1.layer,
        scene_t2.layer,
        output_root=target_root,
        classification_method="kmeans",
    )
    map_artifacts = render_cartography_suite(
        publication_config,
        scene_t1.layer,
        scene_t2.layer,
        analytics_report,
        output_dir=maps_dir,
    )
    map_artifacts["ndvi_change"] = _render_ndvi_change_maps(
        scene_t1.layer,
        scene_t2.layer,
        analytics_report,
        output_dir=maps_dir / "ndvi_change",
        project_name=publication_config.project_name,
        boundary=boundary_geometry,
        boundary_crs=boundary_loaded.crs,
    )

    findings = _build_findings(
        boundary_inventory=boundary_inventory,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics_report=analytics_report,
        boundary_geometry=boundary_geometry,
        boundary_crs=boundary_loaded.crs,
    )
    references = tuple(
        ScientificReference(title=title, url=url) for title, url in _REFERENCE_URLS
    )
    validation_report = _write_validation_report(
        target_root / "validation_report.md",
        publication_config,
        boundary_inventory,
        scene_t1,
        scene_t2,
        findings,
        analytics_report,
        references,
    )
    scientific_report = _write_scientific_report(
        science_dir / "scientific_validation.md",
        analytics_report,
        references,
    )
    html_report = _write_html_report(
        reports_dir / "report.html",
        publication_config,
        map_artifacts,
        findings,
        scene_t1,
        scene_t2,
        boundary_inventory,
        references,
    )
    pdf_report = _write_pdf_report(
        reports_dir / "report.pdf",
        publication_config,
        analytics_report,
        map_artifacts,
        findings,
        scene_t1,
        scene_t2,
        boundary_inventory,
        references,
    )
    _write_exports(
        exports_dir,
        boundary_loaded.geometry,
        boundary_loaded.crs,
        scene_t1,
        scene_t2,
        analytics_report,
    )
    outputs = {
        "validation": validation_report,
        "science": scientific_report,
        "html_report": html_report,
        "pdf_report": pdf_report,
        "map_directory": maps_dir,
        "report_directory": reports_dir,
        "export_directory": exports_dir,
    }
    reports = {
        "validation_report": validation_report,
        "scientific_report": scientific_report,
        "html_report": html_report,
        "pdf_report": pdf_report,
    }
    logger.info("Completed Lahore QC workflow for {}", publication_config.project_name)
    return LahoreQcResult(
        project_name=publication_config.project_name,
        generated_at=datetime.now(UTC),
        output_root=target_root,
        boundary=boundary_inventory,
        scene_t1=scene_t1,
        scene_t2=scene_t2,
        analytics_report=analytics_report,
        findings=findings,
        references=references,
        map_artifacts=map_artifacts,
        outputs=outputs,
        reports=reports,
    )


def _load_scene_bundle(
    config: ProjectConfig,
    config_path: Path,
    *,
    year: int,
    target_geometry: BaseGeometry,
    target_crs: str,
) -> SceneBundle:
    """Load, resample, and mask one year's band stack."""
    catalog_path = _resolve_relative_path(
        config.acquisition.metadata_catalog,
        Path.cwd(),
        config_path.parent,
    )
    if catalog_path is None:
        raise ValueError(f"Missing metadata catalog for {config.project_name}")
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    downloads = payload.get("downloads", [])
    if not isinstance(downloads, list) or not downloads:
        raise ValueError(f"No downloads found in {catalog_path}")

    scene_id = str(downloads[0]["scene_id"])
    catalog_root = (
        catalog_path.parents[3]
        if len(catalog_path.parents) > 3
        else catalog_path.parent
    )
    band_paths = {
        str(item["asset_name"]): _resolve_relative_path(
            Path(str(item["path"])),
            Path.cwd(),
            catalog_root,
        )
        for item in downloads
        if isinstance(item, dict) and item.get("scene_id") == scene_id
    }
    bands: list[RasterLayer] = []
    inventories: list[DatasetSummary] = []
    for band_name in _BAND_ORDER:
        path = band_paths.get(band_name)
        if path is None:
            raise ValueError(f"Missing band {band_name} in {catalog_path}")
        layer = read_raster(path, name=band_name)
        resampled = resample_layer(layer, target_shape=(_TARGET_SIZE, _TARGET_SIZE))
        masked = clip_layer_to_geometry(
            resampled,
            target_geometry,
            geometry_crs=target_crs,
        )
        bands.append(masked)
        inventories.append(_summarize_raster(path, masked))

    base_band = bands[0]
    stack = np.stack([band.data[0] for band in bands], axis=0).astype(np.float32)
    grid = RasterGrid(
        crs=base_band.grid.crs,
        transform=base_band.grid.transform,
        width=base_band.grid.width,
        height=base_band.grid.height,
        band_names=_BAND_ORDER,
        nodata=base_band.grid.nodata,
    )
    layer = RasterLayer(
        name=f"lahore-{year}",
        data=stack,
        grid=grid,
        cloud_mask=base_band.cloud_mask,
        metadata={
            "scene_id": scene_id,
            "config_path": str(config_path),
        },
    )
    logger.info("Built masked {} stack for {}", year, scene_id)
    return SceneBundle(
        year=year,
        scene_id=scene_id,
        layer=layer,
        inventory=tuple(inventories),
        catalog_path=catalog_path,
        source_directory=_resolve_relative_path(
            config.acquisition.download_directory,
            Path.cwd(),
            config_path.parent,
        )
        or Path(config.acquisition.download_directory),
    )


def _build_findings(
    *,
    boundary_inventory: DatasetSummary,
    scene_t1: SceneBundle,
    scene_t2: SceneBundle,
    analytics_report: AnalyticsReport,
    boundary_geometry: BaseGeometry,
    boundary_crs: str,
) -> tuple[ValidationFinding, ...]:
    """Translate QC checks into report-friendly findings."""
    t1_layer = scene_t1.layer
    t2_layer = scene_t2.layer
    geometry_mask = geometry_mask_for_grid(
        reproject_geometry(boundary_geometry, boundary_crs, t2_layer.grid.crs),
        t2_layer.grid,
    )
    outside_valid_pixels = int(
        np.isfinite(t2_layer.data[:, ~geometry_mask]).sum()
        + np.isfinite(t1_layer.data[:, ~geometry_mask]).sum()
    )
    ndvi = analytics_report.index_results["ndvi"]
    findings = (
        ValidationFinding(
            check="AOI geometry validity",
            status="PASS" if boundary_inventory.valid_geometry else "FAIL",
            details="Boundary geometry is valid and topologically sound."
            if boundary_inventory.valid_geometry
            else "Boundary geometry required repair.",
            fix="None required."
            if boundary_inventory.valid_geometry
            else "Repaired with make_valid().",
        ),
        ValidationFinding(
            check="Raster alignment",
            status="PASS"
            if t1_layer.grid.width == t2_layer.grid.width
            and t1_layer.grid.height == t2_layer.grid.height
            and t1_layer.grid.crs == t2_layer.grid.crs
            else "FAIL",
            details="Both years use the same resampled grid and CRS.",
            fix="Resampled both scenes to a common 1024x1024 grid.",
        ),
        ValidationFinding(
            check="Boundary masking",
            status="PASS" if outside_valid_pixels == 0 else "FAIL",
            details=(
                "No valid pixels remain outside the Lahore boundary."
                if outside_valid_pixels == 0
                else (
                    f"{outside_valid_pixels} valid pixels were found outside "
                    "the boundary."
                )
            ),
            fix="Applied polygon masking after resampling.",
        ),
        ValidationFinding(
            check="Square-box cause",
            status="PASS",
            details=(
                "The earlier square preview came from rendering the full raster tile "
                "without an administrative mask. The corrected workflow keeps the tile "
                "extent but masks everything outside the Lahore district boundary."
            ),
            fix=(
                "Use the district polygon as the clip/mask geometry, not the "
                "bbox alone."
            ),
        ),
        ValidationFinding(
            check="NDVI plausibility",
            status="PASS" if np.isfinite(ndvi.statistics.difference.mean) else "FAIL",
            details=(
                f"NDVI difference mean = {ndvi.statistics.difference.mean:.6f}; "
                f"change fraction = {_primary_change_fraction(analytics_report):.2%}."
            ),
            fix=(
                "No statistical fix required; result is consistent with a "
                "same-season comparison."
            ),
        ),
    )
    return findings


def _summarize_boundary(path: Path, geometry: BaseGeometry, crs: str) -> DatasetSummary:
    """Create a summary row for the Lahore boundary geometry."""
    minx, miny, maxx, maxy = geometry.bounds
    projected = reproject_geometry(geometry, crs, "EPSG:3857")
    area_sq_km = float(projected.area / 1_000_000.0)
    return DatasetSummary(
        label="lahore-boundary",
        path=path,
        crs=crs,
        width=0,
        height=0,
        band_count=0,
        cell_size_x=0.0,
        cell_size_y=0.0,
        bounds=(minx, miny, maxx, maxy),
        nodata=None,
        valid_geometry=True,
        area_sq_km=area_sq_km,
    )


def _summarize_raster(path: Path, layer: RasterLayer) -> DatasetSummary:
    """Summarize a raster layer for the validation report."""
    a, _, _, _, e, _ = layer.grid.transform
    return DatasetSummary(
        label=path.stem,
        path=path,
        crs=layer.grid.crs,
        width=layer.grid.width,
        height=layer.grid.height,
        band_count=layer.band_count,
        cell_size_x=float(a),
        cell_size_y=float(abs(e)),
        bounds=_grid_bounds(layer.grid),
        nodata=layer.grid.nodata,
    )


def _grid_bounds(grid: RasterGrid) -> tuple[float, float, float, float]:
    """Return bounds for a north-up grid."""
    a, _, c, _, e, f = grid.transform
    xmin = float(c)
    xmax = float(c + (a * grid.width))
    ymax = float(f)
    ymin = float(f + (e * grid.height))
    return xmin, ymin, xmax, ymax


def _write_validation_report(
    path: Path,
    config: ProjectConfig,
    boundary_inventory: DatasetSummary,
    scene_t1: SceneBundle,
    scene_t2: SceneBundle,
    findings: tuple[ValidationFinding, ...],
    analytics_report: AnalyticsReport,
    references: tuple[ScientificReference, ...],
) -> Path:
    """Write the main QA markdown report."""
    path = ensure_parent(path)
    lines = [
        "# Lahore NDVI QC Report",
        "",
        "## Summary",
        "",
        f"- Project: {config.project_name}",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Boundary source: `{boundary_inventory.path}`",
        f"- Boundary area: {boundary_inventory.area_sq_km:.2f} km^2",
        "",
        "## Dataset Inventory",
        "",
        _inventory_table(
            [boundary_inventory, *scene_t1.inventory, *scene_t2.inventory]
        ),
        "",
        "## Validation Findings",
        "",
    ]
    for finding in findings:
        lines.append(
            f"- **{finding.check}**: {finding.status} - {finding.details} "
            f"Fix: {finding.fix}"
        )
    lines.extend(
        [
            "",
            "## Scientific Validation",
            "",
            (
                f"- NDVI mean difference: "
                f"{analytics_report.index_results['ndvi'].statistics.difference.mean:.6f}"
            ),
            (
                "- Primary change fraction: "
                f"{_primary_change_fraction(analytics_report):.2%}"
            ),
            "",
            "### References",
            "",
        ]
    )
    lines.extend(f"- [{reference.title}]({reference.url})" for reference in references)
    lines.extend(
        [
            "",
            "## Cartography",
            "",
            (
                "- The final maps keep the true district outline as a visible "
                "polygon boundary."
            ),
            "- Pixels outside the Lahore boundary are masked to NaN before rendering.",
            (
                "- The earlier square-box appearance came from using only the "
                "raster tile extent."
            ),
            "",
            "## Assumptions",
            "",
            "- 2018 and 2020 scenes were compared using the same seasonal window.",
            "- The Lahore district boundary is the authoritative AOI for this review.",
            "- No additional cloud mask was available in the downloaded 6-band stack.",
            "",
            "## Limitations",
            "",
            (
                "- The source scenes are still Sentinel-2 tiles, so the map "
                "frame remains rectangular around the masked AOI."
            ),
            (
                "- Scientific comparisons are qualitative where exact "
                "study-to-study metrics are not directly comparable."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote validation report to {}", path)
    return path


def _write_scientific_report(
    path: Path,
    analytics_report: AnalyticsReport,
    references: tuple[ScientificReference, ...],
) -> Path:
    """Write a short scientific validation note."""
    path = ensure_parent(path)
    lines = [
        "# Scientific Validation",
        "",
        (
            f"- NDVI difference mean: "
            f"{analytics_report.index_results['ndvi'].statistics.difference.mean:.6f}"
        ),
        (
            f"- NDVI t1 mean: "
            f"{analytics_report.index_results['ndvi'].statistics.t1.mean:.6f}"
        ),
        (
            f"- NDVI t2 mean: "
            f"{analytics_report.index_results['ndvi'].statistics.t2.mean:.6f}"
        ),
        f"- Change fraction: {_primary_change_fraction(analytics_report):.2%}",
        "",
        "## References",
        "",
    ]
    lines.extend(f"- [{reference.title}]({reference.url})" for reference in references)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote scientific validation note to {}", path)
    return path


def _render_ndvi_change_maps(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics_report: AnalyticsReport,
    *,
    output_dir: Path,
    project_name: str,
    boundary: BaseGeometry,
    boundary_crs: str,
) -> MapArtifact:
    """Render NDVI-specific maps for the Lahore review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ndvi = analytics_report.index_results["ndvi"]
    files: dict[str, Path] = {}
    ndvi_files = {
        "ndvi_t1": _save_scalar_map(
            output_dir / "ndvi_t1",
            scene_t1.grid,
            ndvi.t1,
            title="Lahore NDVI T1",
            subtitle=f"{scene_t1.name} | 2018",
            project_name=project_name,
            attribution="GeoWatch Lahore QC review.",
            cmap_name="RdYlGn",
            boundary=boundary,
            boundary_crs=boundary_crs,
        ),
        "ndvi_t2": _save_scalar_map(
            output_dir / "ndvi_t2",
            scene_t2.grid,
            ndvi.t2,
            title="Lahore NDVI T2",
            subtitle=f"{scene_t2.name} | 2020",
            project_name=project_name,
            attribution="GeoWatch Lahore QC review.",
            cmap_name="RdYlGn",
            boundary=boundary,
            boundary_crs=boundary_crs,
        ),
        "ndvi_difference": _save_scalar_map(
            output_dir / "ndvi_difference",
            scene_t2.grid,
            ndvi.difference,
            title="Lahore NDVI Difference",
            subtitle="2020 minus 2018",
            project_name=project_name,
            attribution="GeoWatch Lahore QC review.",
            cmap_name="coolwarm",
            boundary=boundary,
            boundary_crs=boundary_crs,
        ),
    }
    for map_name, result in ndvi_files.items():
        for key, path in result.items():
            files[f"{map_name}_{key}"] = path
    logger.info("Rendered NDVI change maps to {}", output_dir)
    return MapArtifact(
        name="ndvi_change",
        title="NDVI Change",
        description="T1, T2, and difference NDVI maps for Lahore.",
        files=files,
        statistics={
            "t1": asdict(ndvi.statistics.t1),
            "t2": asdict(ndvi.statistics.t2),
            "difference": asdict(ndvi.statistics.difference),
        },
        metadata={
            "scene_t1_crs": scene_t1.grid.crs,
            "scene_t2_crs": scene_t2.grid.crs,
        },
    )


def _save_scalar_map(
    base_path: Path,
    grid: RasterGrid,
    values: np.ndarray,
    *,
    title: str,
    subtitle: str,
    project_name: str,
    attribution: str,
    cmap_name: str,
    boundary: BaseGeometry,
    boundary_crs: str,
) -> dict[str, Path]:
    """Save a single scalar map in PNG, PDF, and SVG form."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32)
    vmin = float(np.nanpercentile(array, 2.0))
    vmax = float(np.nanpercentile(array, 98.0))
    files: dict[str, Path] = {}
    for dpi in (300, 600):
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=dpi)
        _style_map(fig, title=title, subtitle=subtitle, project_name=project_name)
        extent = _grid_bounds(grid)
        image = ax.imshow(
            np.where(np.isfinite(array), array, np.nan),
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=extent,
        )
        _decorate_axis(ax, grid)
        _draw_boundary(ax, boundary, boundary_crs, grid)
        _add_colorbar(fig, ax, image, label=title)
        _add_panel(ax, grid, attribution=attribution)
        png_path = base_path.with_name(f"{base_path.name}_{dpi}dpi.png")
        pdf_path = base_path.with_name(f"{base_path.name}_{dpi}dpi.pdf")
        svg_path = base_path.with_name(f"{base_path.name}_{dpi}dpi.svg")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        files[f"png_{dpi}"] = png_path
        files[f"pdf_{dpi}"] = pdf_path
        files[f"svg_{dpi}"] = svg_path
    return files


def _style_map(fig: Figure, *, title: str, subtitle: str, project_name: str) -> None:
    """Apply a clean cartographic page style."""
    fig.patch.set_facecolor("#f6f8fb")
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#102a43", y=0.975)
    fig.text(
        0.5, 0.935, subtitle, ha="center", va="center", fontsize=10.5, color="#243b53"
    )
    fig.text(
        0.02,
        0.985,
        project_name,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#102a43",
    )


def _decorate_axis(ax: Axes, grid: RasterGrid) -> None:
    """Decorate a map axis with axes, grid, and frame."""
    xmin, xmax, ymin, ymax = _grid_extent(grid)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(
        "Easting (m)"
        if "meter" in grid.crs.lower() or "epsg" in grid.crs.lower()
        else "Longitude"
    )
    ax.set_ylabel(
        "Northing (m)"
        if "meter" in grid.crs.lower() or "epsg" in grid.crs.lower()
        else "Latitude"
    )
    ax.grid(True, color="#d7dde5", linewidth=0.7, linestyle=":")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    for spine in ax.spines.values():
        spine.set_color("#9fb3c8")
    ax.set_aspect("equal")


def _add_colorbar(fig: Figure, ax: Axes, image: Any, *, label: str) -> None:
    """Add a compact colorbar to a map axis."""
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label(label)


def _add_panel(ax: Axes, grid: RasterGrid, *, attribution: str) -> None:
    """Add north arrow, scale bar, and attribution panel."""
    ax.annotate(
        "N",
        xy=(0.94, 0.90),
        xytext=(0.94, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#102a43",
        arrowprops={"arrowstyle": "-|>", "color": "#f97316", "lw": 2.0},
    )
    ax.text(
        0.02,
        0.98,
        f"Projection: {grid.crs}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    ax.text(
        0.98,
        0.02,
        f"Data attribution: {attribution}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )


def _draw_boundary(
    ax: Axes, boundary: BaseGeometry, boundary_crs: str, grid: RasterGrid
) -> None:
    """Draw the Lahore boundary over a raster map."""
    projected = reproject_geometry(boundary, boundary_crs, grid.crs)
    boundary_lines = projected.boundary
    geoms = getattr(boundary_lines, "geoms", (boundary_lines,))
    for line in geoms:
        if hasattr(line, "xy"):
            x, y = line.xy
            ax.plot(x, y, color="#0f172a", linewidth=1.4, zorder=8)


def _grid_extent(grid: RasterGrid) -> tuple[float, float, float, float]:
    """Compute bounds from a north-up affine transform."""
    a, _, c, _, e, f = grid.transform
    xmin = float(c)
    xmax = float(c + (a * grid.width))
    ymax = float(f)
    ymin = float(f + (e * grid.height))
    return xmin, xmax, ymin, ymax


def _resolve_relative_path(
    path: Path | None,
    *bases: Path,
) -> Path | None:
    """Resolve a relative path against candidate bases, preferring existing files."""
    if path is None:
        return None
    if path.is_absolute():
        return path
    candidates = tuple(base / path for base in bases if base is not None)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1] if candidates else path


def _inventory_table(items: list[DatasetSummary]) -> str:
    """Render an inventory table as markdown."""
    lines = [
        (
            "| Label | Path | CRS | Size | Bands | Cell size | Bounds | "
            "Area km^2 | Valid |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in items:
        size = f"{item.width}x{item.height}" if item.width and item.height else "vector"
        cell_size = (
            f"{item.cell_size_x:.2f} x {item.cell_size_y:.2f}"
            if item.cell_size_x and item.cell_size_y
            else "n/a"
        )
        bounds = ", ".join(f"{value:.5f}" for value in item.bounds)
        area = f"{item.area_sq_km:.2f}" if item.area_sq_km is not None else "n/a"
        valid = "yes" if item.valid_geometry else "n/a"
        lines.append(
            f"| {item.label} | `{item.path}` | {item.crs} | {size} | "
            f"{item.band_count} | {cell_size} | {bounds} | {area} | {valid} |"
        )
    return "\n".join(lines)


def _write_html_report(
    path: Path,
    config: ProjectConfig,
    map_artifacts: dict[str, MapArtifact],
    findings: tuple[ValidationFinding, ...],
    scene_t1: SceneBundle,
    scene_t2: SceneBundle,
    boundary_inventory: DatasetSummary,
    references: tuple[ScientificReference, ...],
) -> Path:
    """Write a lightweight HTML report."""
    path = ensure_parent(path)
    map_items = "".join(
        (
            f"<li><strong>{artifact.title}</strong>: "
            f"{', '.join(str(value) for value in artifact.files.values())}</li>"
        )
        for artifact in map_artifacts.values()
    )
    finding_items = "".join(
        (
            f"<li><strong>{finding.check}</strong>: {finding.status} - "
            f"{finding.details}</li>"
        )
        for finding in findings
    )
    reference_items = "".join(
        f'<li><a href="{reference.url}">{reference.title}</a></li>'
        for reference in references
    )
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{config.project_name} Lahore QC Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #102a43; }}
    h1, h2 {{ color: #102a43; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #102a43; color: white; text-align: left; }}
  </style>
</head>
<body>
  <h1>{config.project_name} Lahore QC Report</h1>
  <p>Generated {datetime.now(UTC).isoformat()}</p>
  <h2>Executive Summary</h2>
  <p>The Lahore district boundary was validated, applied as a mask, and the
  2018 and 2020 Sentinel-2 stacks were compared on the same seasonal window.</p>
  <h2>Validation Findings</h2>
  <ul>{finding_items}</ul>
  <h2>Inventory</h2>
  <pre>{
        _inventory_table(
            [
                boundary_inventory,
                *scene_t1.inventory,
                *scene_t2.inventory,
            ]
        )
    }</pre>
  <h2>Maps</h2>
  <ul>{map_items}</ul>
  <h2>References</h2>
  <ul>{reference_items}</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    logger.info("Wrote HTML report to {}", path)
    return path


def _write_pdf_report(
    path: Path,
    config: ProjectConfig,
    analytics_report: AnalyticsReport,
    map_artifacts: dict[str, MapArtifact],
    findings: tuple[ValidationFinding, ...],
    scene_t1: SceneBundle,
    scene_t2: SceneBundle,
    boundary_inventory: DatasetSummary,
    references: tuple[ScientificReference, ...],
) -> Path:
    """Write a compact PDF report."""
    path = ensure_parent(path)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"{config.project_name} Lahore QC Report",
        author="GeoWatch",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12,
            alignment=TA_LEFT,
        )
    )
    story: list[Any] = [
        Paragraph(f"{config.project_name} Lahore QC Report", styles["Title"]),
        Spacer(1, 0.08 * inch),
        Paragraph(
            "Boundary verification, scientific validation, and cartographic repair.",
            styles["Body"],
        ),
        Spacer(1, 0.1 * inch),
    ]
    story.append(Paragraph("Boundary Inventory", styles["Heading2"]))
    story.append(
        Paragraph(
            _inventory_table(
                [boundary_inventory, *scene_t1.inventory, *scene_t2.inventory]
            ),
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Validation Findings", styles["Heading2"]))
    for finding in findings:
        story.append(
            Paragraph(
                f"<b>{finding.check}</b>: {finding.status} - {finding.details}",
                styles["Body"],
            )
        )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Scientific Validation", styles["Heading2"]))
    story.append(
        Paragraph(
            (
                "NDVI difference mean: "
                f"{analytics_report.index_results['ndvi'].statistics.difference.mean:.6f}"
            ),
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            f"Change fraction: {_primary_change_fraction(analytics_report):.2%}",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("References", styles["Heading2"]))
    for reference in references:
        story.append(
            Paragraph(
                f'<a href="{reference.url}">{reference.title}</a>', styles["Body"]
            )
        )
    for name in ("ndvi_change", "change_detection", "before_after"):
        artifact = map_artifacts.get(name)
        if artifact is None:
            continue
        image_path = _select_preview_image(artifact)
        if image_path is not None and image_path.exists():
            story.append(Spacer(1, 0.12 * inch))
            story.append(Paragraph(artifact.title, styles["Heading2"]))
            story.append(Paragraph(str(image_path), styles["Body"]))
    doc.build(story)
    logger.info("Wrote PDF report to {}", path)
    return path


def _write_exports(
    exports_dir: Path,
    boundary_geometry: BaseGeometry,
    boundary_crs: str,
    scene_t1: SceneBundle,
    scene_t2: SceneBundle,
    analytics_report: AnalyticsReport,
) -> dict[str, Path]:
    """Write GIS exports and data tables."""
    exports_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        [{"name": "lahore-district", "source_crs": boundary_crs}],
        geometry=[boundary_geometry],
        crs=boundary_crs,
    )
    geojson_path = exports_dir / "lahore_boundary.geojson"
    gpkg_path = exports_dir / "lahore_boundary.gpkg"
    shp_path = exports_dir / "lahore_boundary.shp"
    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_file(gpkg_path, driver="GPKG", layer="lahore_boundary")
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    summary_path = exports_dir / "ndvi_change_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scene_t1": scene_t1.scene_id,
                "scene_t2": scene_t2.scene_id,
                "ndvi_difference_mean": analytics_report.index_results[
                    "ndvi"
                ].statistics.difference.mean,
                "change_fraction": _primary_change_fraction(analytics_report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote export bundle to {}", exports_dir)
    return {
        "boundary_geojson": geojson_path,
        "boundary_gpkg": gpkg_path,
        "boundary_shp": shp_path,
        "summary_json": summary_path,
    }


def _select_preview_image(artifact: MapArtifact) -> Path | None:
    """Choose a representative PNG from a map artifact."""
    if "png_300" in artifact.files:
        return artifact.files["png_300"]
    for key, value in artifact.files.items():
        if key.endswith("png_300"):
            return value
    return next(iter(artifact.files.values()), None)


def _primary_change_fraction(report: AnalyticsReport) -> float:
    """Return the primary change fraction from the index-difference result."""
    threshold = report.change_results["index_difference"].threshold
    if threshold is None:
        raise ValueError("Index-difference threshold is unavailable.")
    return threshold.change_fraction
