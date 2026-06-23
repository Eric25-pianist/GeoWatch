"""Typed models for raster processing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RasterGrid:
    """Spatial metadata describing a raster array."""

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    band_names: tuple[str, ...] = ()
    nodata: float | int | None = None

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert raster indices to world coordinates for north-up rasters."""
        a, b, c, d, e, f = self.transform
        if b != 0 or d != 0:
            raise ValueError("Rotated rasters are not supported by this helper.")
        x = c + (col * a)
        y = f + (row * e)
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to pixel indices for north-up rasters."""
        a, b, c, d, e, f = self.transform
        if b != 0 or d != 0:
            raise ValueError("Rotated rasters are not supported by this helper.")
        col = int((x - c) / a)
        row = int((y - f) / e)
        return row, col

    def subset(self, row_slice: slice, col_slice: slice) -> RasterGrid:
        """Return a grid updated for a row/column subset."""
        a, b, c, d, e, f = self.transform
        start_row = row_slice.start or 0
        start_col = col_slice.start or 0
        new_transform = (
            a,
            b,
            c + (start_col * a),
            d,
            e,
            f + (start_row * e),
        )
        return RasterGrid(
            crs=self.crs,
            transform=new_transform,
            width=len(range(*col_slice.indices(self.width))),
            height=len(range(*row_slice.indices(self.height))),
            band_names=self.band_names,
            nodata=self.nodata,
        )


@dataclass(slots=True)
class RasterLayer:
    """In-memory raster stack with spatial metadata."""

    name: str
    data: NDArray[np.generic]
    grid: RasterGrid
    cloud_mask: NDArray[np.bool_] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the raster array shape."""
        if self.data.ndim != 3:
            raise ValueError("RasterLayer.data must have shape (bands, rows, cols).")
        bands, rows, cols = self.data.shape
        if rows != self.grid.height or cols != self.grid.width:
            raise ValueError("RasterLayer.data shape does not match grid dimensions.")
        if self.cloud_mask is not None and self.cloud_mask.shape != (rows, cols):
            raise ValueError("cloud_mask must match the raster height and width.")
        if self.grid.band_names and len(self.grid.band_names) != bands:
            raise ValueError("band_names count must match the raster band count.")

    @property
    def band_count(self) -> int:
        """Return the number of bands."""
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the band-first array shape."""
        bands, rows, cols = self.data.shape
        return int(bands), int(rows), int(cols)

    def band_name(self, index: int) -> str:
        """Return a band name or a generated fallback."""
        if self.grid.band_names:
            return self.grid.band_names[index]
        return f"band_{index + 1}"

    def with_data(
        self,
        data: NDArray[np.generic],
        *,
        grid: RasterGrid | None = None,
        cloud_mask: NDArray[np.bool_] | None = None,
        name: str | None = None,
    ) -> RasterLayer:
        """Return a copy with updated data or metadata."""
        return RasterLayer(
            name=name or self.name,
            data=data,
            grid=grid or self.grid,
            cloud_mask=self.cloud_mask if cloud_mask is None else cloud_mask,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class RasterStatistics:
    """Summary statistics for a raster layer."""

    layer_name: str
    valid_pixels: int
    cloud_pixels: int
    nodata_pixels: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    cloud_coverage: float


@dataclass(frozen=True)
class ProcessingReport:
    """Result bundle returned by the Phase 3 pipeline."""

    phase: int
    messages: tuple[str, ...]
    statistics: tuple[RasterStatistics, ...]
    artifacts: dict[str, Path]

    def summary(self) -> str:
        """Render a concise text summary."""
        lines = [f"GeoWatch Phase {self.phase} raster processing report"]
        lines.extend(f"- {message}" for message in self.messages)
        return "\n".join(lines)
