"""Professional cartography helpers for GeoWatch Phase 5."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from PIL import Image as PILImage
from pyproj import CRS, Transformer
from scipy.ndimage import uniform_filter
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from geowatch.analytics.indices import extract_canonical_bands
from geowatch.analytics.models import (
    AnalyticsReport,
    ChangeDetectionResult,
    MapStatistics,
    ThresholdResult,
    summarize_array,
)
from geowatch.cartography.themes import MapTheme, get_map_theme
from geowatch.config.models import ProjectConfig
from geowatch.processing.models import RasterGrid, RasterLayer
from geowatch.reporting.models import HotspotAnalysis, MapArtifact
from geowatch.utils.geometry import load_vector_geometry, reproject_geometry

CLASS_COLORS: dict[str, str] = {
    "Water": "#1f78b4",
    "Urban": "#d95f02",
    "Vegetation": "#31a354",
    "Agriculture": "#ffd166",
    "Bare Soil": "#a6761d",
    "Forest": "#006d2c",
    "Wetlands": "#1f9e9a",
    "Snow/Ice": "#f7fbff",
    "Bright Surface / Uncertain": "#d9d9d9",
}

INDEX_CMAPS: dict[str, str] = {
    "ndvi": "YlGn",
    "ndbi": "OrRd",
    "ndwi": "Blues",
}

SIGNED_CHANGE_COLORS: dict[str, str] = {
    "Loss": "#D55E00",
    "No change": "#B8B8B8",
    "Gain": "#009E73",
}

MAP_NAMES: tuple[str, ...] = (
    "ndvi",
    "ndbi",
    "ndwi",
    "lulc",
    "change_detection",
    "hotspot_analysis",
    "before_after",
)

_NORTH_ARROW_COLOR = "#f97316"
_TITLE_COLOR = "#102a43"
_SUBTITLE_COLOR = "#243b53"
_FOOTER_COLOR = "#334e68"
_PAGE_FACE = "#f4f6f8"
_PANEL_FACE = "#fbfcfe"
_PANEL_EDGE = "#c9d4df"


def render_cartography_suite(
    config: ProjectConfig,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    analytics_report: AnalyticsReport,
    *,
    output_dir: Path,
) -> dict[str, MapArtifact]:
    """Render the Phase 5 cartographic products to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, MapArtifact] = {}
    theme = get_map_theme(config.outputs.map_theme)
    aoi_geometry, aoi_crs = _load_aoi_geometry(config)
    attribution = _scene_attribution(config, scene_t1, scene_t2)
    primary_change = _select_primary_change_result(analytics_report)
    hotspot = (
        analyze_hotspots(primary_change.score) if primary_change is not None else None
    )
    reference_rgb = _scene_to_rgb(scene_t2, max_dimension=1800)
    t1_label = _scene_display_label(scene_t1)
    t2_label = _scene_display_label(scene_t2)
    cartography_specs = (
        (
            "ndvi",
            "NDVI",
            "Normalized Difference Vegetation Index",
            analytics_report.index_results["ndvi"].t2,
            INDEX_CMAPS["ndvi"],
            analytics_report.index_results["ndvi"].statistics.t2,
        ),
        (
            "ndbi",
            "NDBI",
            "Normalized Difference Built-up Index",
            analytics_report.index_results["ndbi"].t2,
            INDEX_CMAPS["ndbi"],
            analytics_report.index_results["ndbi"].statistics.t2,
        ),
        (
            "ndwi",
            "NDWI",
            "Normalized Difference Water Index",
            analytics_report.index_results["ndwi"].t2,
            INDEX_CMAPS["ndwi"],
            analytics_report.index_results["ndwi"].statistics.t2,
        ),
    )
    for name, title, description, values, cmap_name, statistics in cartography_specs:
        artifacts[name] = _render_continuous_map(
            name=name,
            title=title,
            description=description,
            values=values,
            grid=scene_t2.grid,
            output_dir=output_dir / name,
            cmap_name=cmap_name,
            statistics=statistics,
            project_name=config.project_name,
            subtitle=f"{t2_label} | Approved {config.aoi.kind.upper()} AOI",
            attribution=attribution,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        )

    lulc_result = analytics_report.classification_results.get("lulc_t2")
    if lulc_result is not None:
        display_class_names = _cartographic_class_names(
            lulc_result.class_names,
            exploratory=lulc_result.method in {"kmeans", "isodata"},
        )
        artifacts["lulc"] = _render_classification_map(
            name="lulc",
            title="LULC",
            description="Land-use and land-cover classification for the after scene.",
            labels=lulc_result.labels,
            class_names=display_class_names,
            grid=scene_t2.grid,
            output_dir=output_dir / "lulc",
            project_name=config.project_name,
            subtitle=f"{t2_label} | Exploratory LULC classification",
            attribution=f"{attribution} | Exploratory unsupervised LULC",
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        )
    else:
        logger.info("Skipping LULC map because no LULC result is available.")

    if primary_change is not None:
        artifacts["change_detection"] = _render_change_map(
            primary_change,
            grid=scene_t2.grid,
            output_dir=output_dir / "change_detection",
            project_name=config.project_name,
            subtitle=f"{primary_change.method.upper()} | {t1_label} to {t2_label}",
            attribution=attribution,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        )
    else:
        logger.info("Skipping change map because no change result is available.")
    if hotspot is not None:
        artifacts["hotspot_analysis"] = _render_hotspot_map(
            hotspot,
            grid=scene_t2.grid,
            output_dir=output_dir / "hotspot_analysis",
            project_name=config.project_name,
            subtitle=f"Getis-Ord Gi* | {t1_label} to {t2_label}",
            attribution=attribution,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        )
    artifacts["before_after"] = _render_before_after_map(
        scene_t1,
        scene_t2,
        output_dir=output_dir / "before_after",
        project_name=config.project_name,
        subtitle=f"{t1_label} vs {t2_label}",
        attribution=attribution,
        aoi_geometry=aoi_geometry,
        aoi_crs=aoi_crs,
        theme=theme,
    )
    if analytics_report.signed_change is not None:
        signed = analytics_report.signed_change
        artifacts["ndvi_gain_loss"] = _render_classification_map(
            name="ndvi_gain_loss",
            title="NDVI Gain, Stability, and Loss",
            description=(
                "Categorical vegetation change from the signed NDVI difference."
            ),
            labels=np.asarray(signed.labels, dtype=np.int64),
            class_names=signed.class_names,
            grid=scene_t2.grid,
            output_dir=output_dir / "ndvi_gain_loss",
            project_name=config.project_name,
            subtitle=f"Threshold: +/-{signed.threshold:.4f} NDVI",
            attribution=attribution,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            class_colors=SIGNED_CHANGE_COLORS,
            reference_rgb=reference_rgb,
            theme=theme,
        )
    logger.info("Rendered {} cartography products to {}", len(artifacts), output_dir)
    return artifacts


def analyze_hotspots(
    score: NDArray[np.float32],
    *,
    window_size: int = 7,
    z_threshold: float = 1.96,
) -> HotspotAnalysis:
    """Compute a Getis-Ord Gi* hotspot surface from a change score map."""
    array = np.asarray(score, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        raise ValueError("Hotspot analysis requires at least one finite value.")

    filled = np.where(finite_mask, array, 0.0)
    window_area = float(window_size * window_size)
    local_sum = uniform_filter(filled, size=window_size, mode="nearest") * window_area
    global_values = array[finite_mask]
    total_count = float(global_values.size)
    global_mean = float(global_values.mean())
    global_std = float(global_values.std())
    denominator = global_std * np.sqrt(
        np.maximum((total_count * window_area) - (window_area**2), 0.0)
        / max(total_count - 1.0, 1.0)
    )
    numerator = local_sum - (global_mean * window_area)
    if denominator == 0.0:
        gi_star = np.zeros_like(array, dtype=np.float32)
    else:
        gi_star = numerator / denominator
    hotspot_mask = gi_star >= z_threshold
    coldspot_mask = gi_star <= -z_threshold
    statistics = summarize_array("gi_star", gi_star)
    logger.info(
        "Computed hotspot analysis with {} hotspots and {} coldspots",
        int(np.sum(hotspot_mask)),
        int(np.sum(coldspot_mask)),
    )
    return HotspotAnalysis(
        score=array,
        gi_star=np.asarray(gi_star, dtype=np.float32),
        hotspot_mask=hotspot_mask,
        coldspot_mask=coldspot_mask,
        statistics=statistics,
        metadata={
            "window_size": window_size,
            "z_threshold": z_threshold,
            "global_mean": global_mean,
            "global_std": global_std,
        },
    )


def _render_continuous_map(
    *,
    name: str,
    title: str,
    description: str,
    values: NDArray[np.float32],
    grid: RasterGrid,
    output_dir: Path,
    cmap_name: str,
    statistics: MapStatistics,
    project_name: str,
    subtitle: str,
    attribution: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> MapArtifact:
    """Render a continuous thematic map with a colorbar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32)
    vmin, vmax = _display_bounds(array)
    files = _save_figure_bundle(
        fig_builder=lambda dpi: _build_continuous_figure(
            array=array,
            grid=grid,
            title=title,
            subtitle=subtitle,
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax,
            statistics=statistics,
            attribution=attribution,
            dpi=dpi,
            project_name=project_name,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        ),
        base_path=output_dir / name,
    )
    overlay_path = output_dir / f"{name}_overlay.png"
    plt.imsave(overlay_path, array, cmap=cmap_name, vmin=vmin, vmax=vmax)
    files["overlay_png"] = overlay_path
    metadata: dict[str, object] = {
        "grid_crs": grid.crs,
        "bounds_wgs84": _grid_bounds_wgs84(grid),
        "cmap": cmap_name,
        "vmin": vmin,
        "vmax": vmax,
        "map_theme": theme.name,
        "map_theme_label": theme.label,
    }
    return MapArtifact(
        name=name,
        title=title,
        description=description,
        files=files,
        statistics=asdict(statistics),
        metadata=metadata,
    )


def _render_classification_map(
    *,
    name: str,
    title: str,
    description: str,
    labels: NDArray[np.int64],
    class_names: tuple[str, ...],
    grid: RasterGrid,
    output_dir: Path,
    project_name: str,
    subtitle: str,
    attribution: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    class_colors: dict[str, str] | None = None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> MapArtifact:
    """Render a discrete LULC classification map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = class_colors or CLASS_COLORS
    colors = [palette[class_name] for class_name in class_names]
    cmap = mcolors.ListedColormap(colors).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    norm = mcolors.BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, len(class_names))
    files = _save_figure_bundle(
        fig_builder=lambda dpi: _build_classification_figure(
            labels=labels,
            grid=grid,
            title=title,
            subtitle=subtitle,
            class_names=class_names,
            cmap=cmap,
            norm=norm,
            attribution=attribution,
            dpi=dpi,
            project_name=project_name,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            class_colors=palette,
            reference_rgb=reference_rgb,
            theme=theme,
        ),
        base_path=output_dir / name,
    )
    overlay_path = output_dir / f"{name}_overlay.png"
    valid_mask = (labels >= 0) & (labels < len(class_names))
    plt.imsave(
        overlay_path,
        np.ma.array(labels, mask=~valid_mask),
        cmap=cmap,
        vmin=-0.5,
        vmax=float(len(class_names) - 0.5),
    )
    files["overlay_png"] = overlay_path
    counts = {
        class_name: int(np.sum(labels == index))
        for index, class_name in enumerate(class_names)
        if np.any(labels == index)
    }
    metadata: dict[str, object] = {
        "grid_crs": grid.crs,
        "bounds_wgs84": _grid_bounds_wgs84(grid),
        "class_colors": palette,
        "map_theme": theme.name,
        "map_theme_label": theme.label,
    }
    statistics: dict[str, object] = {"counts": counts}
    return MapArtifact(
        name=name,
        title=title,
        description=description,
        files=files,
        statistics=statistics,
        metadata=metadata,
    )


def _render_change_map(
    change_result: ChangeDetectionResult,
    *,
    grid: RasterGrid,
    output_dir: Path,
    project_name: str,
    subtitle: str,
    attribution: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> MapArtifact:
    """Render a change detection score map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(change_result.score, dtype=np.float32)
    threshold = change_result.threshold
    vmin, vmax = _display_bounds(values)
    files = _save_figure_bundle(
        fig_builder=lambda dpi: _build_change_figure(
            values=values,
            threshold=threshold,
            grid=grid,
            title="Change Detection",
            subtitle=subtitle,
            attribution=attribution,
            dpi=dpi,
            project_name=project_name,
            vmin=vmin,
            vmax=vmax,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        ),
        base_path=output_dir / "change_detection",
    )
    overlay_path = output_dir / "change_detection_overlay.png"
    plt.imsave(overlay_path, values, cmap="coolwarm", vmin=vmin, vmax=vmax)
    files["overlay_png"] = overlay_path
    metadata: dict[str, object] = {
        "grid_crs": grid.crs,
        "bounds_wgs84": _grid_bounds_wgs84(grid),
        "threshold_method": threshold.method if threshold is not None else None,
        "map_theme": theme.name,
        "map_theme_label": theme.label,
    }
    statistics: dict[str, object] = {
        "score": asdict(change_result.statistics),
        "changed_fraction": (
            threshold.change_fraction if threshold is not None else 0.0
        ),
        "changed_pixels": threshold.changed_pixels if threshold is not None else 0,
    }
    return MapArtifact(
        name="change_detection",
        title="Change Detection",
        description="Primary change score and detected change fraction.",
        files=files,
        statistics=statistics,
        metadata=metadata,
    )


def _render_hotspot_map(
    hotspot: HotspotAnalysis,
    *,
    grid: RasterGrid,
    output_dir: Path,
    project_name: str,
    subtitle: str,
    attribution: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> MapArtifact:
    """Render a hotspot analysis map from a Gi* surface."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _display_bounds(hotspot.gi_star)
    files = _save_figure_bundle(
        fig_builder=lambda dpi: _build_hotspot_figure(
            hotspot=hotspot,
            grid=grid,
            title="Hotspot Analysis",
            subtitle=subtitle,
            attribution=attribution,
            dpi=dpi,
            project_name=project_name,
            vmin=vmin,
            vmax=vmax,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            reference_rgb=reference_rgb,
            theme=theme,
        ),
        base_path=output_dir / "hotspot_analysis",
    )
    overlay_path = output_dir / "hotspot_analysis_overlay.png"
    plt.imsave(overlay_path, hotspot.gi_star, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    files["overlay_png"] = overlay_path
    metadata: dict[str, object] = {
        "grid_crs": grid.crs,
        "bounds_wgs84": _grid_bounds_wgs84(grid),
        "window_size": hotspot.metadata.get("window_size"),
        "z_threshold": hotspot.metadata.get("z_threshold"),
        "map_theme": theme.name,
        "map_theme_label": theme.label,
    }
    statistics: dict[str, object] = {
        "gi_star": asdict(hotspot.statistics),
        "hotspots": int(np.sum(hotspot.hotspot_mask)),
        "coldspots": int(np.sum(hotspot.coldspot_mask)),
    }
    return MapArtifact(
        name="hotspot_analysis",
        title="Hotspot Analysis",
        description="Getis-Ord Gi* concentration map for change intensity.",
        files=files,
        statistics=statistics,
        metadata=metadata,
    )


def _render_before_after_map(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    *,
    output_dir: Path,
    project_name: str,
    subtitle: str,
    attribution: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    theme: MapTheme,
) -> MapArtifact:
    """Render a before/after RGB comparison map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_t1 = _scene_to_rgb(scene_t1)
    rgb_t2 = _scene_to_rgb(scene_t2)
    files = _save_figure_bundle(
        fig_builder=lambda dpi: _build_before_after_figure(
            scene_t1=scene_t1,
            scene_t2=scene_t2,
            rgb_t1=rgb_t1,
            rgb_t2=rgb_t2,
            title="Before / After Comparison",
            subtitle=subtitle,
            attribution=attribution,
            dpi=dpi,
            project_name=project_name,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            theme=theme,
        ),
        base_path=output_dir / "before_after",
    )
    files.update(_save_slider_images(scene_t1, scene_t2, output_dir))
    metadata: dict[str, object] = {
        "scene_t1_crs": scene_t1.grid.crs,
        "scene_t2_crs": scene_t2.grid.crs,
        "bounds_wgs84": _grid_bounds_wgs84(scene_t1.grid),
        "map_theme": theme.name,
        "map_theme_label": theme.label,
    }
    statistics: dict[str, object] = {
        "scene_t1": _rgb_summary(rgb_t1),
        "scene_t2": _rgb_summary(rgb_t2),
    }
    return MapArtifact(
        name="before_after",
        title="Before / After Comparison",
        description="Side-by-side RGB comparison of the two acquisition dates.",
        files=files,
        statistics=statistics,
        metadata=metadata,
    )


def _save_slider_images(
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    output_dir: Path,
) -> dict[str, Path]:
    """Save matched, web-sized endpoint images for the offline swipe viewer."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_t1 = _scene_to_rgb(scene_t1, max_dimension=1800)
        rgb_t2 = _scene_to_rgb(scene_t2, max_dimension=1800)
        if rgb_t1.shape != rgb_t2.shape:
            logger.warning(
                "Skipping slider images because endpoint shapes differ: {} and {}",
                rgb_t1.shape,
                rgb_t2.shape,
            )
            return {}
        paths = {
            "slider_before": output_dir / "before_slider.png",
            "slider_after": output_dir / "after_slider.png",
        }
        for key, rgb in (
            ("slider_before", rgb_t1),
            ("slider_after", rgb_t2),
        ):
            display = np.where(np.isfinite(rgb), rgb, 0.93)
            pixels = np.asarray(np.rint(display * 255.0), dtype=np.uint8)
            PILImage.fromarray(pixels).save(
                paths[key],
                format="PNG",
                optimize=True,
            )
        logger.info("Wrote matched before/after dashboard slider images")
        return paths
    except (OSError, ValueError) as exc:
        logger.warning("Could not write dashboard slider images: {}", exc)
        return {}


def _save_figure_bundle(
    *,
    fig_builder: Callable[[int], Figure],
    base_path: Path,
) -> dict[str, Path]:
    """Save a figure in the required export formats and return the file map."""
    files: dict[str, Path] = {}
    vector_pdf: Path | None = None
    vector_svg: Path | None = None
    for dpi in (300, 600):
        fig = fig_builder(dpi)
        png_path = base_path.with_name(f"{base_path.name}_{dpi}dpi.png")
        jpeg_path = base_path.with_name(f"{base_path.name}_{dpi}dpi.jpg")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        fig.savefig(
            jpeg_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
        )
        if vector_pdf is None or vector_svg is None:
            vector_pdf = base_path.with_name(f"{base_path.name}.pdf")
            vector_svg = base_path.with_name(f"{base_path.name}.svg")
            fig.savefig(vector_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
            fig.savefig(vector_svg, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        files[f"png_{dpi}"] = png_path
        files[f"jpeg_{dpi}"] = jpeg_path
        files[f"pdf_{dpi}"] = vector_pdf
        files[f"svg_{dpi}"] = vector_svg
        plt.close(fig)
    return files


def _build_continuous_figure(
    *,
    array: NDArray[np.float32],
    grid: RasterGrid,
    title: str,
    subtitle: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    statistics: MapStatistics,
    attribution: str,
    dpi: int,
    project_name: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> Figure:
    """Construct a cartographic figure for continuous rasters."""
    fig, ax, panel_ax = _create_single_map_layout(
        dpi=dpi,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    _style_figure(
        fig,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    extent = _grid_extent(grid)
    masked = np.where(np.isfinite(array), array, np.nan).astype(np.float32, copy=False)
    _add_geometry_background(ax, aoi_geometry, aoi_crs, grid, theme=theme)
    _draw_reference_basemap(ax, grid, reference_rgb, theme=theme)
    image = ax.imshow(
        masked,
        cmap=_transparent_cmap(cmap_name),
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        alpha=theme.raster_alpha,
        zorder=3,
    )
    _decorate_axes(ax, grid, theme=theme)
    _add_colorbar(fig, ax, image, label=title, theme=theme)
    _add_map_annotations(
        ax,
        grid,
        statistics,
        attribution=attribution,
        aoi_geometry=aoi_geometry,
        aoi_crs=aoi_crs,
        panel_ax=panel_ax,
        legend_handles=(
            mpatches.Patch(color=theme.no_data_color, label="No valid observation"),
        ),
        legend_title="Map key",
        theme=theme,
    )
    return fig


def _build_classification_figure(
    *,
    labels: NDArray[np.int64],
    grid: RasterGrid,
    title: str,
    subtitle: str,
    class_names: tuple[str, ...],
    cmap: mcolors.Colormap,
    norm: mcolors.BoundaryNorm,
    attribution: str,
    dpi: int,
    project_name: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    class_colors: dict[str, str],
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> Figure:
    """Construct a cartographic figure for discrete classifications."""
    fig, ax, panel_ax = _create_single_map_layout(
        dpi=dpi,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    _style_figure(
        fig,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    extent = _grid_extent(grid)
    valid_mask = (labels >= 0) & (labels < len(class_names))
    masked_labels = np.ma.array(
        labels,
        mask=~valid_mask,
    )
    display_cmap = cmap.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    _add_geometry_background(ax, aoi_geometry, aoi_crs, grid, theme=theme)
    _draw_reference_basemap(ax, grid, reference_rgb, theme=theme)
    ax.imshow(
        masked_labels,
        cmap=display_cmap,
        norm=norm,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        alpha=theme.classification_alpha,
        zorder=3,
    )
    _decorate_axes(ax, grid, theme=theme)
    present_indices = sorted(int(value) for value in np.unique(labels[valid_mask]))
    legend_handles = [
        mpatches.Patch(color=class_colors[class_name], label=class_name)
        for index, class_name in enumerate(class_names)
        if index in present_indices
    ]
    if np.any(~valid_mask):
        legend_handles.append(
            mpatches.Patch(color=theme.no_data_color, label="No valid observation")
        )
    _add_map_annotations(
        ax,
        grid,
        summarize_array(
            "classification",
            np.where(valid_mask, labels, np.nan).astype(np.float32, copy=False),
        ),
        attribution=attribution,
        aoi_geometry=aoi_geometry,
        aoi_crs=aoi_crs,
        panel_ax=panel_ax,
        legend_handles=legend_handles,
        legend_title="Displayed classes",
        theme=theme,
    )
    return fig


def _build_change_figure(
    *,
    values: NDArray[np.float32],
    threshold: ThresholdResult | None,
    grid: RasterGrid,
    title: str,
    subtitle: str,
    attribution: str,
    dpi: int,
    project_name: str,
    vmin: float,
    vmax: float,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> Figure:
    """Construct a change detection map figure."""
    fig, ax, panel_ax = _create_single_map_layout(
        dpi=dpi,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    _style_figure(
        fig,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    extent = _grid_extent(grid)
    masked = np.where(np.isfinite(values), values, np.nan).astype(
        np.float32, copy=False
    )
    _add_geometry_background(ax, aoi_geometry, aoi_crs, grid, theme=theme)
    _draw_reference_basemap(ax, grid, reference_rgb, theme=theme)
    image = ax.imshow(
        masked,
        cmap=_transparent_cmap("RdBu_r"),
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        alpha=theme.change_alpha,
        zorder=3,
    )
    _decorate_axes(ax, grid, theme=theme)
    _add_colorbar(fig, ax, image, label="Change score", theme=theme)
    if threshold is not None:
        mask = np.asarray(threshold.mask, dtype=bool)
        overlay = np.where(mask, 1.0, np.nan).astype(np.float32, copy=False)
        ax.imshow(
            overlay,
            cmap=mcolors.ListedColormap(["#f97316"]),
            alpha=0.25,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            zorder=4,
        )
    _add_map_annotations(
        ax,
        grid,
        summarize_array("change_detection", values),
        attribution=attribution,
        aoi_geometry=aoi_geometry,
        aoi_crs=aoi_crs,
        panel_ax=panel_ax,
        legend_handles=(
            mpatches.Patch(color=theme.no_data_color, label="No valid observation"),
        ),
        legend_title="Map key",
        theme=theme,
    )
    return fig


def _build_hotspot_figure(
    *,
    hotspot: HotspotAnalysis,
    grid: RasterGrid,
    title: str,
    subtitle: str,
    attribution: str,
    dpi: int,
    project_name: str,
    vmin: float,
    vmax: float,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    reference_rgb: NDArray[np.float32] | None = None,
    theme: MapTheme,
) -> Figure:
    """Construct a hotspot analysis map figure."""
    fig, ax, panel_ax = _create_single_map_layout(
        dpi=dpi,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    _style_figure(
        fig,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    extent = _grid_extent(grid)
    masked = np.where(
        np.isfinite(hotspot.gi_star),
        hotspot.gi_star,
        np.nan,
    ).astype(np.float32, copy=False)
    _add_geometry_background(ax, aoi_geometry, aoi_crs, grid, theme=theme)
    _draw_reference_basemap(ax, grid, reference_rgb, theme=theme)
    image = ax.imshow(
        masked,
        cmap=_transparent_cmap("RdBu_r"),
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        alpha=theme.hotspot_alpha,
        zorder=3,
    )
    _decorate_axes(ax, grid, theme=theme)
    _add_colorbar(fig, ax, image, label="Gi* z-score", theme=theme)
    hotspot_overlay = np.where(
        hotspot.hotspot_mask,
        1.0,
        np.nan,
    ).astype(np.float32, copy=False)
    ax.imshow(
        hotspot_overlay,
        cmap=mcolors.ListedColormap(["#f97316"]),
        alpha=0.22,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        zorder=4,
    )
    _add_map_annotations(
        ax,
        grid,
        hotspot.statistics,
        attribution=attribution,
        aoi_geometry=aoi_geometry,
        aoi_crs=aoi_crs,
        panel_ax=panel_ax,
        legend_handles=(
            mpatches.Patch(color=theme.no_data_color, label="No valid observation"),
        ),
        legend_title="Map key",
        theme=theme,
    )
    return fig


def _build_before_after_figure(
    *,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
    rgb_t1: NDArray[np.float32],
    rgb_t2: NDArray[np.float32],
    title: str,
    subtitle: str,
    attribution: str,
    dpi: int,
    project_name: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    theme: MapTheme,
) -> Figure:
    """Construct a before/after comparison figure."""
    fig = plt.figure(figsize=theme.comparison_map_size, dpi=dpi)
    _style_figure(
        fig,
        title=title,
        subtitle=subtitle,
        project_name=project_name,
        theme=theme,
    )
    grid_spec = fig.add_gridspec(
        1,
        3,
        left=theme.comparison_left,
        right=theme.comparison_right,
        top=theme.comparison_top,
        bottom=theme.comparison_bottom,
        width_ratios=theme.comparison_ratios,
        wspace=theme.comparison_wspace,
    )
    axes = (fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[0, 1]))
    panel_ax = fig.add_subplot(grid_spec[0, 2])
    extent = _grid_extent(scene_t1.grid)
    scene_t1_label = _scene_display_label(scene_t1)
    scene_t2_label = _scene_display_label(scene_t2)
    for axis, rgb, scene, label in (
        (axes[0], rgb_t1, scene_t1, f"Before | {_scene_date_text(scene_t1)}"),
        (axes[1], rgb_t2, scene_t2, f"After | {_scene_date_text(scene_t2)}"),
    ):
        axis.imshow(rgb, origin="upper", extent=extent, interpolation="nearest")
        axis.set_title(
            label,
            color=theme.title_color,
            fontsize=max(theme.subtitle_size + 1.2, 10.5),
            fontweight="bold",
            pad=8,
        )
        _decorate_axes(axis, scene.grid, theme=theme)
        _add_map_annotations(
            axis,
            scene.grid,
            _rgb_summary(rgb),
            attribution=attribution,
            compact=True,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            theme=theme,
        )
    _populate_side_panel(
        panel_ax,
        grid=scene_t1.grid,
        statistics=_rgb_summary(rgb_t2),
        attribution=attribution,
        aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            panel_title="Comparison Details",
            extra_lines=(
                f"Before: {scene_t1_label}",
                f"After: {scene_t2_label}",
                "Panels use matched extent, scale, and CRS.",
            ),
            theme=theme,
        )
    axes[0].text(
        0.02,
        0.94,
        "Before",
        transform=axes[0].transAxes,
        fontsize=max(theme.subtitle_size + 1.0, 10.5),
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": theme.outline_color,
            "alpha": 0.84,
        },
    )
    axes[1].text(
        0.02,
        0.94,
        "After",
        transform=axes[1].transAxes,
        fontsize=max(theme.subtitle_size + 1.0, 10.5),
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": theme.outline_color,
            "alpha": 0.84,
        },
    )
    return fig


def _style_figure(
    fig: Figure,
    *,
    title: str,
    subtitle: str,
    project_name: str,
    theme: MapTheme,
) -> None:
    """Apply a unified publication page style to a map figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": theme.axis_label_size,
            "xtick.labelsize": theme.tick_label_size,
            "ytick.labelsize": theme.tick_label_size,
        }
    )
    plt.rcParams["font.family"] = theme.font_family
    fig.patch.set_facecolor(theme.page_face)
    fig.suptitle(
        title,
        fontsize=theme.title_size,
        fontweight="bold",
        color=theme.title_color,
        y=0.965,
    )
    fig.text(
        0.5,
        0.928,
        subtitle,
        ha="center",
        va="center",
        fontsize=theme.subtitle_size,
        color=theme.subtitle_color,
    )
    fig.text(
        0.025,
        0.972,
        project_name,
        ha="left",
        va="top",
        fontsize=theme.brand_size,
        fontweight="bold",
        color=theme.title_color,
    )


def _create_single_map_layout(
    *,
    dpi: int,
    title: str,
    subtitle: str,
    project_name: str,
    theme: MapTheme,
) -> tuple[Figure, Axes, Axes]:
    """Create the standard GeoWatch single-map page layout."""
    _ = (title, subtitle, project_name)
    fig = plt.figure(figsize=theme.single_map_size, dpi=dpi)
    grid_spec = fig.add_gridspec(
        1,
        2,
        left=theme.single_map_left,
        right=theme.single_map_right,
        top=theme.single_map_top,
        bottom=theme.single_map_bottom,
        width_ratios=theme.single_map_ratios,
        wspace=theme.single_map_wspace,
    )
    ax = fig.add_subplot(grid_spec[0, 0])
    panel_ax = fig.add_subplot(grid_spec[0, 1])
    return fig, ax, panel_ax


def _draw_reference_basemap(
    ax: Axes,
    grid: RasterGrid,
    reference_rgb: NDArray[np.float32] | None,
    theme: MapTheme,
) -> None:
    """Draw a muted processed-scene reference base beneath thematic data."""
    extent = _grid_extent(grid)
    if reference_rgb is None:
        ax.set_facecolor(theme.basemap_face)
        return
    rgb = np.asarray(reference_rgb, dtype=np.float32)
    valid = np.all(np.isfinite(rgb), axis=-1)
    filled = np.nan_to_num(rgb, nan=1.0, posinf=1.0, neginf=0.0)
    luminance = (
        (filled[..., 0] * 0.2126)
        + (filled[..., 1] * 0.7152)
        + (filled[..., 2] * 0.0722)
    )
    gray = np.repeat(luminance[..., np.newaxis], 3, axis=-1)
    muted = np.clip(
        (gray * theme.basemap_gray_weight)
        + (filled * theme.basemap_color_weight)
        + theme.basemap_offset,
        0.0,
        1.0,
    )
    ax.imshow(
        muted,
        origin="upper",
        extent=extent,
        interpolation="bilinear",
        zorder=0,
        alpha=np.where(valid, theme.basemap_alpha, 0.0),
    )


def _transparent_cmap(cmap_name: str) -> mcolors.Colormap:
    """Return a colormap with transparent missing-data pixels."""
    return plt.get_cmap(cmap_name).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))


def _decorate_axes(ax: Axes, grid: RasterGrid, *, theme: MapTheme) -> None:
    """Draw coordinate grid, labels, and map framing."""
    extent = _grid_extent(grid)
    x_label, y_label = _axis_label_pair(grid)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    unit_label = _target_unit_label(grid)
    ax.set_xlabel(f"{x_label} ({unit_label})", color=theme.footer_color)
    ax.set_ylabel(f"{y_label} ({unit_label})", color=theme.footer_color)
    ax.tick_params(colors=theme.footer_color, length=3, width=0.6)
    ax.grid(
        theme.show_grid,
        color=theme.grid_color,
        linewidth=theme.grid_linewidth,
        linestyle=theme.grid_linestyle,
        alpha=theme.grid_alpha,
        zorder=1,
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    x_formatter = mticker.ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    y_formatter = mticker.ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    for spine in ax.spines.values():
        spine.set_color(theme.panel_edge)
        spine.set_linewidth(0.9)
    ax.set_facecolor(theme.axis_face)
    ax.set_aspect("equal")


def _add_colorbar(
    fig: Figure,
    ax: Axes,
    image: AxesImage,
    *,
    label: str,
    theme: MapTheme,
) -> None:
    """Attach a compact publication colorbar below the map frame."""
    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=theme.colorbar_fraction,
        pad=theme.colorbar_pad,
        aspect=38,
    )
    colorbar.set_label(label, color=theme.footer_color, size=theme.colorbar_label_size)
    colorbar.ax.tick_params(
        colors=theme.footer_color,
        labelsize=theme.colorbar_tick_size,
    )


def _add_map_annotations(
    ax: Axes,
    grid: RasterGrid,
    statistics: MapStatistics,
    *,
    attribution: str,
    compact: bool = False,
    aoi_geometry: BaseGeometry | None = None,
    aoi_crs: str | None = None,
    panel_ax: Axes | None = None,
    legend_handles: Sequence[mpatches.Patch] | None = None,
    legend_title: str | None = None,
    theme: MapTheme,
) -> None:
    """Add north arrow, scale bar, statistics, and footer text."""
    _add_north_arrow(ax, theme=theme)
    _add_scale_bar(ax, grid, theme=theme)
    if aoi_geometry is not None and aoi_crs is not None:
        _add_geometry_outline(ax, aoi_geometry, aoi_crs, grid, theme=theme)
    if panel_ax is not None:
        _populate_side_panel(
            panel_ax,
            grid=grid,
            statistics=statistics,
            attribution=attribution,
            aoi_geometry=aoi_geometry,
            aoi_crs=aoi_crs,
            legend_handles=legend_handles,
            legend_title=legend_title,
            theme=theme,
        )
        return
    if compact:
        return
    if aoi_geometry is not None and aoi_crs is not None:
        _add_locator_inset(ax, aoi_geometry, aoi_crs, theme=theme)
    stats_text = _format_stats(statistics)
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=theme.stats_font_size,
        color=theme.title_color,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": theme.panel_face,
            "alpha": 0.92,
            "edgecolor": theme.panel_edge,
        },
    )
    ax.text(
        0.98,
        0.02,
        f"Projection: {grid.crs}\nData attribution: {attribution}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=theme.footer_font_size,
        color=theme.footer_color,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": theme.panel_face,
            "alpha": 0.84,
            "edgecolor": theme.panel_edge,
        },
    )


def _populate_side_panel(
    panel_ax: Axes,
    *,
    grid: RasterGrid,
    statistics: MapStatistics,
    attribution: str,
    aoi_geometry: BaseGeometry | None,
    aoi_crs: str | None,
    legend_handles: Sequence[mpatches.Patch] | None = None,
    legend_title: str | None = None,
    panel_title: str = "Map Details",
    extra_lines: Sequence[str] = (),
    theme: MapTheme,
) -> None:
    """Populate the reusable map-side information panel."""
    panel_ax.set_facecolor(theme.panel_face)
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    panel_ax.set_xlim(0.0, 1.0)
    panel_ax.set_ylim(0.0, 1.0)
    for spine in panel_ax.spines.values():
        spine.set_color(theme.panel_edge)
        spine.set_linewidth(0.9)
    panel_ax.text(
        0.07,
        0.965,
        panel_title,
        ha="left",
        va="top",
        fontsize=theme.panel_title_size,
        fontweight="bold",
        color=theme.title_color,
        transform=panel_ax.transAxes,
    )

    if aoi_geometry is not None and aoi_crs is not None:
        _add_locator_inset(
            panel_ax,
            aoi_geometry,
            aoi_crs,
            bounds=(0.08, 0.705, 0.84, 0.205),
            title="Locator",
            theme=theme,
        )

    next_y = 0.665
    if legend_handles:
        legend = panel_ax.legend(
            handles=list(legend_handles),
            title=legend_title or "Legend",
            loc="upper left",
            bbox_to_anchor=(0.055, next_y),
            frameon=False,
            borderaxespad=0.0,
            labelspacing=0.42,
            handlelength=1.2,
            handleheight=0.8,
            fontsize=theme.legend_font_size,
            title_fontsize=theme.legend_title_size,
        )
        legend.get_title().set_color(theme.title_color)
        for text in legend.get_texts():
            text.set_color(theme.subtitle_color)
        next_y = max(0.315, next_y - (0.05 * len(legend_handles)) - 0.095)

    detail_lines = tuple(extra_lines) + tuple(_format_stats(statistics).splitlines())
    panel_ax.text(
        0.07,
        next_y,
        "\n".join(detail_lines),
        ha="left",
        va="top",
        fontsize=theme.stats_font_size,
        color=theme.subtitle_color,
        linespacing=1.28,
        transform=panel_ax.transAxes,
    )
    source_text = "\n".join(textwrap.wrap(attribution, width=32))
    projection_text = "\n".join(textwrap.wrap(f"Projection: {grid.crs}", width=32))
    panel_ax.text(
        0.07,
        0.045,
        f"{projection_text}\nSource: {source_text}",
        ha="left",
        va="bottom",
        fontsize=theme.footer_font_size,
        color=theme.footer_color,
        linespacing=1.25,
        transform=panel_ax.transAxes,
    )


def _add_geometry_outline(
    ax: Axes,
    geometry: BaseGeometry,
    geometry_crs: str,
    grid: RasterGrid,
    theme: MapTheme,
) -> None:
    """Overlay the AOI boundary on a cartographic axis."""
    projected = reproject_geometry(geometry, geometry_crs, grid.crs)
    for polygon in _iter_polygons(projected):
        exterior_x, exterior_y = polygon.exterior.xy
        ax.plot(
            exterior_x,
            exterior_y,
            color=theme.outline_color,
            linewidth=theme.outline_width,
            zorder=8,
        )
        for interior in polygon.interiors:
            interior_x, interior_y = interior.xy
            ax.plot(
                interior_x,
                interior_y,
                color=theme.outline_color,
                linewidth=max(theme.outline_width * 0.57, 0.8),
                linestyle="--",
                zorder=8,
            )


def _add_geometry_background(
    ax: Axes,
    geometry: BaseGeometry | None,
    geometry_crs: str | None,
    grid: RasterGrid,
    theme: MapTheme,
) -> None:
    """Fill the approved AOI subtly so missing observations remain explicit."""
    if geometry is None or geometry_crs is None:
        return
    projected = reproject_geometry(geometry, geometry_crs, grid.crs)
    for polygon in _iter_polygons(projected):
        exterior_x, exterior_y = polygon.exterior.xy
        ax.fill(
            exterior_x,
            exterior_y,
            facecolor=theme.no_data_color,
            edgecolor="none",
            alpha=0.9,
            zorder=0.5,
        )


def _add_locator_inset(
    ax: Axes,
    geometry: BaseGeometry,
    geometry_crs: str,
    *,
    bounds: tuple[float, float, float, float] = (0.78, 0.62, 0.19, 0.22),
    title: str = "Regional locator",
    theme: MapTheme,
) -> None:
    """Add a regional WGS84 locator showing the approved AOI footprint."""
    located = reproject_geometry(geometry, geometry_crs, "EPSG:4326")
    west, south, east, north = located.bounds
    width = max(east - west, 0.05)
    height = max(north - south, 0.05)
    pad = max(width, height) * 1.75
    inset = ax.inset_axes(bounds)
    inset.set_xlim(west - pad, east + pad)
    inset.set_ylim(south - pad, north + pad)
    inset.set_facecolor(theme.locator_face)
    for polygon in _iter_polygons(located):
        x, y = polygon.exterior.xy
        inset.fill(x, y, color=theme.locator_fill, alpha=0.85, linewidth=0.6)
        inset.plot(x, y, color=theme.outline_color, linewidth=0.6)
    inset.set_title(title, fontsize=7.1, color=theme.title_color, pad=2)
    inset.tick_params(labelsize=5.2, colors=theme.footer_color, length=2)
    inset.grid(theme.show_grid, color=theme.grid_color, linewidth=0.45, alpha=0.7)
    for spine in inset.spines.values():
        spine.set_color(theme.panel_edge)
        spine.set_linewidth(0.65)


def _scene_attribution(
    config: ProjectConfig,
    scene_t1: RasterLayer,
    scene_t2: RasterLayer,
) -> str:
    """Build concise attribution from processed-scene provenance."""
    providers: set[str] = set()
    scene_ids: list[str] = []
    dates: list[str] = []
    for scene in (scene_t1, scene_t2):
        provider = scene.metadata.get("provider")
        if provider:
            providers.add(str(provider))
        raw_ids = scene.metadata.get("source_scene_ids", [])
        raw_dates = scene.metadata.get("source_dates", [])
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, str):
            scene_ids.extend(str(item) for item in raw_ids)
        if isinstance(raw_dates, Sequence) and not isinstance(raw_dates, str):
            dates.extend(str(item) for item in raw_dates)
    provider_text = ", ".join(sorted(providers)) or "catalog provider"
    unique_dates = tuple(dict.fromkeys(date[:10] for date in dates))
    date_text = ", ".join(unique_dates) or "dates in report"
    scene_text = f"{len(set(scene_ids))} scene IDs in report"
    return (
        f"{provider_text} | {date_text} | {scene_text} | "
        f"AOI: {config.aoi.kind} | GeoWatch Project"
    )


def _scene_display_label(scene: RasterLayer) -> str:
    """Return a publication-friendly mission and acquisition-date label."""
    dataset_labels = {
        "sentinel-2-l2a": "Sentinel-2 L2A",
        "landsat-5-c2-l2": "Landsat 5 Collection 2 L2",
        "landsat-7-c2-l2": "Landsat 7 Collection 2 L2",
        "landsat-8-c2-l2": "Landsat 8 Collection 2 L2",
        "landsat-9-c2-l2": "Landsat 9 Collection 2 L2",
    }
    dataset = str(scene.metadata.get("dataset", "")).strip().lower()
    mission = dataset_labels.get(dataset)
    if mission is None:
        mission = scene.name.replace("_", " ").strip().title() or "Processed imagery"

    raw_dates = scene.metadata.get("source_dates", ())
    parsed_dates: set[datetime] = set()
    if isinstance(raw_dates, Sequence) and not isinstance(raw_dates, str):
        for raw_date in raw_dates:
            try:
                parsed_dates.add(datetime.fromisoformat(str(raw_date)[:10]))
            except ValueError:
                logger.debug(
                    "Ignoring malformed cartographic source date: {}", raw_date
                )

    dates = sorted(parsed_dates)
    if not dates:
        return mission
    if len(dates) == 1:
        date_text = f"{dates[0].day} {dates[0]:%B %Y}"
    else:
        first = f"{dates[0].day} {dates[0]:%B %Y}"
        last = f"{dates[-1].day} {dates[-1]:%B %Y}"
        date_text = f"{first} to {last}"
    return f"{mission} | {date_text}"


def _scene_date_text(scene: RasterLayer) -> str:
    """Return only the readable date portion of a scene display label."""
    display_label = _scene_display_label(scene)
    _, separator, date_text = display_label.partition(" | ")
    return date_text if separator else "Acquisition date in report"


def _iter_polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Collect all polygon parts from a geometry."""
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    polygons: list[Polygon] = []
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            polygons.extend(_iter_polygons(part))
    return polygons


def _load_aoi_geometry(config: ProjectConfig) -> tuple[BaseGeometry | None, str | None]:
    """Load the configured AOI geometry for map overlays."""
    if config.aoi.path is None:
        return None, None
    path = config.aoi.path
    if not path.is_absolute():
        path = Path.cwd() / path
    loaded = load_vector_geometry(path)
    return loaded.geometry, loaded.crs


def _add_north_arrow(ax: Axes, *, theme: MapTheme) -> None:
    """Add a north arrow in axes coordinates."""
    ax.annotate(
        "N",
        xy=(0.955, 0.955),
        xytext=(0.955, 0.875),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=max(theme.subtitle_size + 0.8, 10.0),
        fontweight="bold",
        color=theme.title_color,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": theme.panel_face,
            "alpha": 0.8,
            "edgecolor": theme.panel_edge,
        },
        arrowprops={"arrowstyle": "-|>", "color": theme.north_arrow_color, "lw": 1.7},
    )


def _add_scale_bar(ax: Axes, grid: RasterGrid, *, theme: MapTheme) -> None:
    """Add a metric scale bar to the map."""
    xmin, xmax, _, _ = _grid_extent(grid)
    width = xmax - xmin
    bar_units = _nice_length(width * 0.25)
    bar_fraction = bar_units / width if width else 0.25
    x_start = 0.08
    y_start = 0.055
    ax.plot(
        [x_start, x_start + bar_fraction],
        [y_start, y_start],
        transform=ax.transAxes,
        color="white",
        linewidth=6.5,
        solid_capstyle="butt",
        zorder=9,
    )
    ax.plot(
        [x_start, x_start + bar_fraction],
        [y_start, y_start],
        transform=ax.transAxes,
        color=theme.title_color,
        linewidth=3.2,
        solid_capstyle="butt",
        zorder=10,
    )
    ax.plot(
        [x_start, x_start],
        [y_start - 0.01, y_start + 0.01],
        transform=ax.transAxes,
        color=theme.title_color,
        linewidth=1.8,
        zorder=10,
    )
    ax.plot(
        [x_start + bar_fraction, x_start + bar_fraction],
        [y_start - 0.01, y_start + 0.01],
        transform=ax.transAxes,
        color=theme.title_color,
        linewidth=1.8,
        zorder=10,
    )
    ax.text(
        x_start + bar_fraction / 2.0,
        y_start + 0.023,
        _format_distance(bar_units, _target_unit_label(grid)),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=theme.footer_font_size + 1.0,
        color=theme.footer_color,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": theme.panel_face,
            "alpha": 0.8,
            "edgecolor": theme.panel_edge,
        },
        zorder=10,
    )


def _format_stats(statistics: MapStatistics) -> str:
    """Format a statistics panel string."""
    return (
        f"Valid pixels: {statistics.valid_pixels:,}/{statistics.total_pixels:,}\n"
        f"Mean: {statistics.mean:.4f}\n"
        f"Std: {statistics.standard_deviation:.4f}\n"
        f"Min: {statistics.minimum:.4f}\n"
        f"Max: {statistics.maximum:.4f}"
    )


def _format_distance(value: float, unit: str) -> str:
    """Format a map distance with readable metric units."""
    normalized_unit = unit.lower()
    if normalized_unit.startswith("met") and value >= 1000.0:
        return f"{value / 1000.0:g} km"
    return f"{value:g} {unit}"


def _nice_length(value: float) -> float:
    """Round a scale bar length to a visually pleasing value."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = value / (10**exponent)
    nice: float
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10.0**exponent)


def _target_unit_label(grid: RasterGrid) -> str:
    """Return a readable map unit label from the CRS."""
    try:
        crs = CRS.from_user_input(grid.crs)
        if crs.axis_info:
            return crs.axis_info[0].unit_name
    except Exception:  # pragma: no cover - fallback for malformed CRS inputs
        logger.debug("Falling back to generic map units for CRS {}", grid.crs)
    return "units"


def _axis_label_pair(grid: RasterGrid) -> tuple[str, str]:
    """Return axis labels suited to the CRS type."""
    try:
        crs = CRS.from_user_input(grid.crs)
        if crs.is_geographic:
            return "Longitude", "Latitude"
    except Exception:  # pragma: no cover - fallback for malformed CRS inputs
        logger.debug("Falling back to projected axis labels for CRS {}", grid.crs)
    return "Easting", "Northing"


def _grid_extent(grid: RasterGrid) -> tuple[float, float, float, float]:
    """Compute raster bounds from the north-up affine transform."""
    a, _, c, _, e, f = grid.transform
    xmin = float(c)
    xmax = float(c + (a * grid.width))
    ymax = float(f)
    ymin = float(f + (e * grid.height))
    return xmin, xmax, ymin, ymax


def _grid_bounds_wgs84(grid: RasterGrid) -> tuple[float, float, float, float]:
    """Convert grid bounds into WGS84 for Leaflet overlays."""
    xmin, xmax, ymin, ymax = _grid_extent(grid)
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(xmin, ymin)
    east, north = transformer.transform(xmax, ymax)
    return float(west), float(south), float(east), float(north)


def _scene_to_rgb(
    scene: RasterLayer,
    *,
    max_dimension: int | None = None,
) -> NDArray[np.float32]:
    """Convert a multispectral scene to a normalized RGB composite."""
    bands = extract_canonical_bands(scene)
    step = 1
    if max_dimension is not None:
        step = max(
            1,
            math.ceil(max(scene.grid.height, scene.grid.width) / max_dimension),
        )
    red = _normalize_band(bands["red"][::step, ::step])
    green = _normalize_band(bands["green"][::step, ::step])
    blue = _normalize_band(bands["blue"][::step, ::step])
    rgb = np.stack([red, green, blue], axis=-1)
    return np.asarray(rgb, dtype=np.float32)


def _normalize_band(band: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize a band to a 0..1 display range."""
    array = np.asarray(band, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low = float(np.nanpercentile(finite, 2.0))
    high = float(np.nanpercentile(finite, 98.0))
    if np.isclose(low, high):
        return np.clip(array, 0.0, 1.0)
    normalized = (array - low) / (high - low)
    return np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


def _display_bounds(
    values: NDArray[np.float32],
    *,
    default: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Return robust display bounds for sparse or constant rasters."""
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return default
    low = float(np.percentile(finite, 2.0))
    high = float(np.percentile(finite, 98.0))
    if not np.isfinite(low) or not np.isfinite(high):
        return default
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 0.01)
        return low - padding, high + padding
    return low, high


def _rgb_summary(rgb: NDArray[np.float32]) -> MapStatistics:
    """Summarize an RGB composite for reporting."""
    finite = np.isfinite(rgb)
    channel_count = finite.sum(axis=-1)
    channel_sum = np.where(finite, rgb, 0.0).sum(axis=-1)
    gray = np.full(channel_count.shape, np.nan, dtype=np.float32)
    np.divide(channel_sum, channel_count, out=gray, where=channel_count > 0)
    return summarize_array("rgb", gray)


def _select_primary_change_result(
    analytics_report: AnalyticsReport,
) -> ChangeDetectionResult | None:
    """Choose the most informative change result from the analytics bundle."""
    for preferred in ("mad", "irmad", "index_difference", "cva"):
        if preferred in analytics_report.change_results:
            return analytics_report.change_results[preferred]
    return next(iter(analytics_report.change_results.values()), None)


def _cartographic_class_names(
    class_names: tuple[str, ...],
    *,
    exploratory: bool,
) -> tuple[str, ...]:
    """Use cautious labels for classes unsupported by exploratory training data."""
    if not exploratory:
        return class_names
    return tuple(
        "Bright Surface / Uncertain" if name == "Snow/Ice" else name
        for name in class_names
    )
