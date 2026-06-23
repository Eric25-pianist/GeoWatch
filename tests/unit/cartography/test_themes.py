"""Unit tests for the professional GeoWatch map theme system."""

from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from geowatch.analytics import run_analytics_pipeline
from geowatch.cartography.themes import MAP_THEME_NAMES, get_map_theme
from geowatch.config.loader import load_config
from geowatch.reporting.cartography import render_cartography_suite
from geowatch.reporting.demo import build_demo_publication_inputs


def test_get_map_theme_supports_all_required_presets() -> None:
    """All declared presets should resolve to stable professional labels."""
    labels = {name: get_map_theme(name).label for name in MAP_THEME_NAMES}

    assert labels == {
        "academic": "Academic Thesis",
        "government": "Government Report",
        "journal": "Minimal Journal",
        "presentation": "Presentation",
        "dark": "Dark Dashboard",
    }


def test_get_map_theme_accepts_human_friendly_aliases() -> None:
    """CLI and config aliases should normalize to the expected canonical key."""
    assert get_map_theme("academic-thesis").name == "academic"
    assert get_map_theme("government_report").name == "government"
    assert get_map_theme("dark dashboard").name == "dark"


def test_get_map_theme_rejects_unknown_values() -> None:
    """Unsupported theme names should raise an actionable validation error."""
    with pytest.raises(ValueError, match="Unknown map theme"):
        get_map_theme("retro-atlas")


def test_render_cartography_suite_uses_selected_theme(
    tmp_path, sample_config_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendered map artifacts should record the configured professional theme."""
    config = load_config(sample_config_path)
    config.outputs.map_theme = "government"
    inputs = build_demo_publication_inputs(config, width=80, height=60)
    analytics_report = run_analytics_pipeline(
        inputs.scene_t1,
        inputs.scene_t2,
        output_root=tmp_path / "analytics",
        classification_method="kmeans",
        training_labels_t1=inputs.training_labels_t1,
        training_labels_t2=inputs.training_labels_t2,
        reference_labels_t1=inputs.reference_labels_t1,
        reference_labels_t2=inputs.reference_labels_t2,
    )

    def _fast_save_bundle(*, fig_builder, base_path: Path) -> dict[str, Path]:
        fig = fig_builder(90)
        output_path = base_path.with_name(f"{base_path.name}_smoke.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=90)
        plt.close(fig)
        return {"png_300": output_path}

    def _skip_slider_images(*_args: object, **_kwargs: object) -> dict[str, Path]:
        return {}

    monkeypatch.setattr(
        "geowatch.reporting.cartography._save_figure_bundle",
        _fast_save_bundle,
    )
    monkeypatch.setattr(
        "geowatch.reporting.cartography._save_slider_images",
        _skip_slider_images,
    )

    artifacts = render_cartography_suite(
        config,
        inputs.scene_t1,
        inputs.scene_t2,
        analytics_report,
        output_dir=tmp_path / "maps",
    )

    assert artifacts["ndvi"].metadata["map_theme"] == "government"
    assert artifacts["before_after"].metadata["map_theme_label"] == "Government Report"
    assert artifacts["change_detection"].files["png_300"].exists()
