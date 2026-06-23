"""Tabular export helpers for GeoWatch Phase 5."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from loguru import logger

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import AnalyticsReport
from geowatch.reporting.models import MapArtifact


def export_publication_tables(
    analytics_report: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    map_artifacts: Mapping[str, MapArtifact],
    output_dir: Path,
) -> dict[str, Path]:
    """Export summary tables to CSV, JSON, and Excel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _build_summary_frame(analytics_report, sources, map_artifacts)
    indices_df = _build_indices_frame(analytics_report)
    change_df = _build_change_frame(analytics_report)
    classification_df = _build_classification_frame(analytics_report)
    sources_df = _build_sources_frame(sources)
    maps_df = _build_maps_frame(map_artifacts)

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    xlsx_path = output_dir / "summary.xlsx"
    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_df.to_dict(orient="records"),
                "indices": indices_df.to_dict(orient="records"),
                "change": change_df.to_dict(orient="records"),
                "classification": classification_df.to_dict(orient="records"),
                "sources": sources_df.to_dict(orient="records"),
                "maps": maps_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        indices_df.to_excel(writer, sheet_name="indices", index=False)
        change_df.to_excel(writer, sheet_name="change", index=False)
        classification_df.to_excel(writer, sheet_name="classification", index=False)
        sources_df.to_excel(writer, sheet_name="sources", index=False)
        maps_df.to_excel(writer, sheet_name="maps", index=False)

    logger.info("Wrote publication exports to {}", output_dir)
    return {
        "csv": csv_path,
        "json": json_path,
        "xlsx": xlsx_path,
    }


def _build_summary_frame(
    analytics_report: AnalyticsReport,
    sources: Sequence[SceneMetadata],
    map_artifacts: Mapping[str, MapArtifact],
) -> pd.DataFrame:
    """Build a one-row publication summary."""
    index_names = ", ".join(analytics_report.index_results)
    change_methods = ", ".join(analytics_report.change_results)
    lulc_methods = ", ".join(analytics_report.classification_results)
    data = [
        {
            "project_phase": analytics_report.phase,
            "index_count": len(analytics_report.index_results),
            "change_count": len(analytics_report.change_results),
            "classification_count": len(analytics_report.classification_results),
            "source_count": len(sources),
            "map_count": len(map_artifacts),
            "index_names": index_names,
            "change_methods": change_methods,
            "lulc_products": lulc_methods,
        }
    ]
    return pd.DataFrame(data)


def _build_indices_frame(analytics_report: AnalyticsReport) -> pd.DataFrame:
    """Build a tabular view of spectral index statistics."""
    rows: list[dict[str, object]] = []
    for name, result in analytics_report.index_results.items():
        rows.append(
            {
                "index": name,
                "t1_mean": result.statistics.t1.mean,
                "t2_mean": result.statistics.t2.mean,
                "difference_mean": result.statistics.difference.mean,
                "difference_std": result.statistics.difference.standard_deviation,
                "valid_fraction": result.statistics.difference.valid_fraction,
            }
        )
    return pd.DataFrame(rows)


def _build_change_frame(analytics_report: AnalyticsReport) -> pd.DataFrame:
    """Build a tabular view of change detection statistics."""
    rows: list[dict[str, object]] = []
    for name, result in analytics_report.change_results.items():
        threshold = result.threshold
        rows.append(
            {
                "method": name,
                "score_mean": result.statistics.mean,
                "score_std": result.statistics.standard_deviation,
                "changed_pixels": (
                    threshold.changed_pixels if threshold is not None else 0
                ),
                "change_fraction": (
                    threshold.change_fraction if threshold is not None else 0.0
                ),
                "threshold_method": threshold.method if threshold is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _build_classification_frame(analytics_report: AnalyticsReport) -> pd.DataFrame:
    """Build a tabular view of classification counts."""
    rows: list[dict[str, object]] = []
    for name, result in analytics_report.classification_results.items():
        row: dict[str, object] = {
            "scene": name,
            "method": result.method,
            "model_name": result.model_name,
        }
        row.update(result.counts)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_sources_frame(sources: Sequence[SceneMetadata]) -> pd.DataFrame:
    """Build a tabular view of satellite source metadata."""
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.append(
            {
                "scene_id": source.scene_id,
                "provider": source.provider,
                "dataset": source.dataset,
                "acquired_at": source.acquired_at.isoformat()
                if source.acquired_at
                else None,
                "cloud_cover": source.cloud_cover,
                "source_url": source.source_url,
            }
        )
    return pd.DataFrame(rows)


def _build_maps_frame(map_artifacts: Mapping[str, MapArtifact]) -> pd.DataFrame:
    """Build a tabular view of generated map artifacts."""
    rows: list[dict[str, object]] = []
    for name, artifact in map_artifacts.items():
        rows.append(
            {
                "map": name,
                "title": artifact.title,
                "description": artifact.description,
                "file_count": len(artifact.files),
            }
        )
    return pd.DataFrame(rows)
