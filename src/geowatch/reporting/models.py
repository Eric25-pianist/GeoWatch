"""Typed reporting models for Phase 5 cartography and publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from geowatch.acquisition.models import SceneMetadata
from geowatch.analytics.models import AnalyticsReport, MapStatistics
from geowatch.config.models import AOIConfig


@dataclass(frozen=True)
class HotspotAnalysis:
    """Getis-Ord Gi* hotspot analysis for a raster surface."""

    score: NDArray[np.float32]
    gi_star: NDArray[np.float32]
    hotspot_mask: NDArray[np.bool_]
    coldspot_mask: NDArray[np.bool_]
    statistics: MapStatistics
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MapArtifact:
    """Published map outputs for a single cartographic theme."""

    name: str
    title: str
    description: str
    files: dict[str, Path]
    statistics: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicationBundle:
    """Complete Phase 5 publication artifact bundle."""

    project_name: str
    generated_at: datetime
    aoi: AOIConfig
    sources: tuple[SceneMetadata, ...]
    analytics_report: AnalyticsReport
    maps: dict[str, MapArtifact]
    exports: dict[str, Path]
    html_report: Path
    pdf_report: Path
    dashboard: Path
    interpretation: Path
    build_report: Path
    portfolio_exports: dict[str, Path]
    example_outputs: dict[str, Path]

    def summary(self) -> str:
        """Render a concise publication summary."""
        lines = [f"GeoWatch Phase 5 publication for {self.project_name}"]
        lines.append(f"- Maps: {len(self.maps)}")
        lines.append(f"- Exports: {len(self.exports)}")
        lines.append(f"- Sources: {len(self.sources)}")
        lines.append(f"- HTML report: {self.html_report}")
        lines.append(f"- PDF report: {self.pdf_report}")
        lines.append(f"- Dashboard: {self.dashboard}")
        lines.append(f"- Interpretation: {self.interpretation}")
        lines.append(f"- Portfolio exports: {len(self.portfolio_exports)}")
        return "\n".join(lines)
