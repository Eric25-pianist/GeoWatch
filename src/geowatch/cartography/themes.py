"""Professional map-theme definitions for GeoWatch cartography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

MapThemeName = Literal[
    "academic",
    "government",
    "journal",
    "presentation",
    "dark",
]


@dataclass(frozen=True)
class MapTheme:
    """Declarative cartographic styling for one GeoWatch export theme."""

    name: MapThemeName
    label: str
    page_face: str
    panel_face: str
    panel_edge: str
    axis_face: str
    title_color: str
    subtitle_color: str
    footer_color: str
    accent_color: str
    north_arrow_color: str
    outline_color: str
    outline_width: float
    locator_face: str
    locator_fill: str
    grid_color: str
    grid_linewidth: float
    grid_linestyle: str
    grid_alpha: float
    show_grid: bool
    font_family: str
    title_size: float
    subtitle_size: float
    brand_size: float
    axis_label_size: float
    tick_label_size: float
    colorbar_label_size: float
    colorbar_tick_size: float
    legend_font_size: float
    legend_title_size: float
    panel_title_size: float
    stats_font_size: float
    footer_font_size: float
    single_map_size: tuple[float, float]
    single_map_top: float
    single_map_bottom: float
    single_map_left: float
    single_map_right: float
    single_map_ratios: tuple[float, float]
    single_map_wspace: float
    comparison_map_size: tuple[float, float]
    comparison_top: float
    comparison_bottom: float
    comparison_left: float
    comparison_right: float
    comparison_ratios: tuple[float, float, float]
    comparison_wspace: float
    colorbar_fraction: float
    colorbar_pad: float
    basemap_face: str
    basemap_gray_weight: float
    basemap_color_weight: float
    basemap_offset: float
    basemap_alpha: float
    raster_alpha: float
    classification_alpha: float
    change_alpha: float
    hotspot_alpha: float
    no_data_color: str


_MAP_THEMES: dict[MapThemeName, MapTheme] = {
    "academic": MapTheme(
        name="academic",
        label="Academic Thesis",
        page_face="#f7f8fa",
        panel_face="#fbfcfe",
        panel_edge="#c9d4df",
        axis_face="#f8fafc",
        title_color="#102a43",
        subtitle_color="#243b53",
        footer_color="#334e68",
        accent_color="#275dad",
        north_arrow_color="#275dad",
        outline_color="#0f172a",
        outline_width=1.35,
        locator_face="#edf2f7",
        locator_fill="#2c7fb8",
        grid_color="#d6dee7",
        grid_linewidth=0.55,
        grid_linestyle=":",
        grid_alpha=1.0,
        show_grid=True,
        font_family="DejaVu Serif",
        title_size=17.0,
        subtitle_size=9.4,
        brand_size=9.2,
        axis_label_size=8.5,
        tick_label_size=7.0,
        colorbar_label_size=7.8,
        colorbar_tick_size=7.0,
        legend_font_size=7.2,
        legend_title_size=7.8,
        panel_title_size=9.2,
        stats_font_size=7.4,
        footer_font_size=6.3,
        single_map_size=(12.7, 8.5),
        single_map_top=0.84,
        single_map_bottom=0.135,
        single_map_left=0.065,
        single_map_right=0.97,
        single_map_ratios=(4.6, 1.28),
        single_map_wspace=0.08,
        comparison_map_size=(14.5, 8.4),
        comparison_top=0.84,
        comparison_bottom=0.105,
        comparison_left=0.055,
        comparison_right=0.97,
        comparison_ratios=(1.0, 1.0, 0.36),
        comparison_wspace=0.055,
        colorbar_fraction=0.045,
        colorbar_pad=0.075,
        basemap_face="#eef2f6",
        basemap_gray_weight=0.82,
        basemap_color_weight=0.18,
        basemap_offset=0.08,
        basemap_alpha=0.78,
        raster_alpha=0.88,
        classification_alpha=0.91,
        change_alpha=0.86,
        hotspot_alpha=0.86,
        no_data_color="#e3e8ed",
    ),
    "government": MapTheme(
        name="government",
        label="Government Report",
        page_face="#f3f5f8",
        panel_face="#f7f9fb",
        panel_edge="#b8c6d2",
        axis_face="#f9fbfc",
        title_color="#102a43",
        subtitle_color="#1f3c5a",
        footer_color="#3d5368",
        accent_color="#0f4c81",
        north_arrow_color="#0f4c81",
        outline_color="#0b1f38",
        outline_width=1.55,
        locator_face="#edf3f8",
        locator_fill="#2b6cb0",
        grid_color="#ccd7e2",
        grid_linewidth=0.6,
        grid_linestyle="-.",
        grid_alpha=0.9,
        show_grid=True,
        font_family="DejaVu Sans",
        title_size=18.0,
        subtitle_size=9.6,
        brand_size=9.6,
        axis_label_size=8.6,
        tick_label_size=7.1,
        colorbar_label_size=7.9,
        colorbar_tick_size=7.1,
        legend_font_size=7.3,
        legend_title_size=7.9,
        panel_title_size=9.5,
        stats_font_size=7.45,
        footer_font_size=6.35,
        single_map_size=(12.9, 8.6),
        single_map_top=0.845,
        single_map_bottom=0.13,
        single_map_left=0.06,
        single_map_right=0.972,
        single_map_ratios=(4.5, 1.35),
        single_map_wspace=0.075,
        comparison_map_size=(14.8, 8.5),
        comparison_top=0.845,
        comparison_bottom=0.1,
        comparison_left=0.05,
        comparison_right=0.972,
        comparison_ratios=(1.0, 1.0, 0.38),
        comparison_wspace=0.05,
        colorbar_fraction=0.046,
        colorbar_pad=0.072,
        basemap_face="#edf2f5",
        basemap_gray_weight=0.8,
        basemap_color_weight=0.2,
        basemap_offset=0.07,
        basemap_alpha=0.74,
        raster_alpha=0.89,
        classification_alpha=0.92,
        change_alpha=0.87,
        hotspot_alpha=0.87,
        no_data_color="#dde4ea",
    ),
    "journal": MapTheme(
        name="journal",
        label="Minimal Journal",
        page_face="#ffffff",
        panel_face="#ffffff",
        panel_edge="#d5dce3",
        axis_face="#ffffff",
        title_color="#111827",
        subtitle_color="#374151",
        footer_color="#4b5563",
        accent_color="#2563eb",
        north_arrow_color="#1f2937",
        outline_color="#111827",
        outline_width=1.2,
        locator_face="#f5f7fa",
        locator_fill="#3b82f6",
        grid_color="#e5e7eb",
        grid_linewidth=0.45,
        grid_linestyle=":",
        grid_alpha=0.6,
        show_grid=False,
        font_family="DejaVu Sans",
        title_size=15.5,
        subtitle_size=8.8,
        brand_size=8.6,
        axis_label_size=8.0,
        tick_label_size=6.8,
        colorbar_label_size=7.2,
        colorbar_tick_size=6.8,
        legend_font_size=6.9,
        legend_title_size=7.4,
        panel_title_size=8.9,
        stats_font_size=7.0,
        footer_font_size=6.0,
        single_map_size=(11.6, 7.7),
        single_map_top=0.83,
        single_map_bottom=0.12,
        single_map_left=0.06,
        single_map_right=0.97,
        single_map_ratios=(4.95, 1.05),
        single_map_wspace=0.05,
        comparison_map_size=(13.4, 7.8),
        comparison_top=0.83,
        comparison_bottom=0.095,
        comparison_left=0.05,
        comparison_right=0.97,
        comparison_ratios=(1.0, 1.0, 0.3),
        comparison_wspace=0.045,
        colorbar_fraction=0.038,
        colorbar_pad=0.065,
        basemap_face="#f4f5f7",
        basemap_gray_weight=0.9,
        basemap_color_weight=0.1,
        basemap_offset=0.05,
        basemap_alpha=0.68,
        raster_alpha=0.9,
        classification_alpha=0.92,
        change_alpha=0.88,
        hotspot_alpha=0.88,
        no_data_color="#eceff3",
    ),
    "presentation": MapTheme(
        name="presentation",
        label="Presentation",
        page_face="#f4f7fb",
        panel_face="#ffffff",
        panel_edge="#c7d5e0",
        axis_face="#f8fbff",
        title_color="#0f172a",
        subtitle_color="#1d3557",
        footer_color="#36536b",
        accent_color="#ef4444",
        north_arrow_color="#ef4444",
        outline_color="#111827",
        outline_width=1.5,
        locator_face="#eef3f9",
        locator_fill="#2563eb",
        grid_color="#cfdae4",
        grid_linewidth=0.62,
        grid_linestyle=":",
        grid_alpha=0.95,
        show_grid=True,
        font_family="DejaVu Sans",
        title_size=20.0,
        subtitle_size=10.6,
        brand_size=10.4,
        axis_label_size=9.0,
        tick_label_size=7.6,
        colorbar_label_size=8.2,
        colorbar_tick_size=7.4,
        legend_font_size=7.8,
        legend_title_size=8.4,
        panel_title_size=10.0,
        stats_font_size=7.8,
        footer_font_size=6.5,
        single_map_size=(13.8, 8.3),
        single_map_top=0.855,
        single_map_bottom=0.12,
        single_map_left=0.05,
        single_map_right=0.975,
        single_map_ratios=(4.55, 1.4),
        single_map_wspace=0.075,
        comparison_map_size=(15.7, 8.8),
        comparison_top=0.855,
        comparison_bottom=0.095,
        comparison_left=0.045,
        comparison_right=0.975,
        comparison_ratios=(1.0, 1.0, 0.4),
        comparison_wspace=0.05,
        colorbar_fraction=0.05,
        colorbar_pad=0.08,
        basemap_face="#eef4f9",
        basemap_gray_weight=0.78,
        basemap_color_weight=0.22,
        basemap_offset=0.08,
        basemap_alpha=0.8,
        raster_alpha=0.9,
        classification_alpha=0.93,
        change_alpha=0.89,
        hotspot_alpha=0.89,
        no_data_color="#dfe7ef",
    ),
    "dark": MapTheme(
        name="dark",
        label="Dark Dashboard",
        page_face="#0f172a",
        panel_face="#111c30",
        panel_edge="#334155",
        axis_face="#142033",
        title_color="#f8fafc",
        subtitle_color="#cbd5e1",
        footer_color="#cbd5e1",
        accent_color="#38bdf8",
        north_arrow_color="#f59e0b",
        outline_color="#f8fafc",
        outline_width=1.45,
        locator_face="#1e293b",
        locator_fill="#38bdf8",
        grid_color="#334155",
        grid_linewidth=0.55,
        grid_linestyle=":",
        grid_alpha=0.9,
        show_grid=True,
        font_family="DejaVu Sans",
        title_size=17.8,
        subtitle_size=9.8,
        brand_size=9.6,
        axis_label_size=8.6,
        tick_label_size=7.1,
        colorbar_label_size=7.8,
        colorbar_tick_size=7.0,
        legend_font_size=7.2,
        legend_title_size=7.8,
        panel_title_size=9.4,
        stats_font_size=7.4,
        footer_font_size=6.3,
        single_map_size=(12.9, 8.4),
        single_map_top=0.845,
        single_map_bottom=0.13,
        single_map_left=0.06,
        single_map_right=0.972,
        single_map_ratios=(4.55, 1.32),
        single_map_wspace=0.08,
        comparison_map_size=(14.9, 8.4),
        comparison_top=0.845,
        comparison_bottom=0.1,
        comparison_left=0.05,
        comparison_right=0.972,
        comparison_ratios=(1.0, 1.0, 0.38),
        comparison_wspace=0.055,
        colorbar_fraction=0.046,
        colorbar_pad=0.074,
        basemap_face="#172235",
        basemap_gray_weight=0.52,
        basemap_color_weight=0.18,
        basemap_offset=0.01,
        basemap_alpha=0.56,
        raster_alpha=0.95,
        classification_alpha=0.95,
        change_alpha=0.92,
        hotspot_alpha=0.92,
        no_data_color="#1e293b",
    ),
}

MAP_THEME_NAMES: tuple[MapThemeName, ...] = tuple(_MAP_THEMES)
MAP_THEME_LABELS: dict[MapThemeName, str] = {
    name: theme.label for name, theme in _MAP_THEMES.items()
}
MAP_THEME_CHOICES: tuple[tuple[MapThemeName, str], ...] = tuple(
    (name, _MAP_THEMES[name].label) for name in MAP_THEME_NAMES
)


def get_map_theme(name: MapThemeName | str) -> MapTheme:
    """Return a validated cartographic theme definition."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "academic_thesis": "academic",
        "government_report": "government",
        "minimal_journal": "journal",
        "dark_dashboard": "dark",
    }
    normalized = aliases.get(normalized, normalized)
    key = normalized.replace("_", "")
    shorthand = {
        "academic": "academic",
        "government": "government",
        "journal": "journal",
        "presentation": "presentation",
        "dark": "dark",
    }
    if key in shorthand:
        return _MAP_THEMES[cast(MapThemeName, shorthand[key])]
    if normalized in _MAP_THEMES:
        return _MAP_THEMES[normalized]
    raise ValueError(
        "Unknown map theme "
        f"'{name}'. Choose one of: {', '.join(MAP_THEME_NAMES)}."
    )
