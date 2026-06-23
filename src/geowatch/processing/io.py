"""Raster I/O helpers with optional Rasterio and GDAL support."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET

import numpy as np
from loguru import logger
from PIL import Image

from geowatch.processing.errors import ProcessingError, RasterDependencyError
from geowatch.processing.models import RasterGrid, RasterLayer

Image.MAX_IMAGE_PIXELS = None


def has_rasterio() -> bool:
    """Return whether Rasterio is importable."""
    return importlib.util.find_spec("rasterio") is not None


def has_gdal() -> bool:
    """Return whether GDAL bindings are importable."""
    return importlib.util.find_spec("osgeo") is not None


def read_raster(path: Path, *, name: str | None = None) -> RasterLayer:
    """Read a raster file into memory when Rasterio is available."""
    if not has_rasterio():
        return _read_raster_with_pillow(path, name=name)
    import rasterio

    try:
        with rasterio.open(path) as dataset:
            data = dataset.read()
            transform_values = tuple(float(value) for value in dataset.transform)[:6]
            grid = RasterGrid(
                crs=str(dataset.crs) if dataset.crs else "EPSG:4326",
                transform=cast(
                    tuple[float, float, float, float, float, float],
                    transform_values,
                ),
                width=dataset.width,
                height=dataset.height,
                band_names=tuple(
                    description or f"band_{index + 1}"
                    for index, description in enumerate(dataset.descriptions)
                ),
                nodata=dataset.nodata,
            )
            logger.info("Read raster {}", path)
            metadata: dict[str, object] = {"source_path": str(path)}
            for key, value in dataset.tags().items():
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    metadata[key] = value
            return RasterLayer(
                name=name or path.stem,
                data=data,
                grid=grid,
                metadata=metadata,
            )
    except Exception as exc:  # pragma: no cover - depends on optional library
        logger.exception("Failed to read raster {}", path)
        raise ProcessingError(f"Could not read raster: {path}") from exc


def _read_raster_with_pillow(path: Path, *, name: str | None = None) -> RasterLayer:
    """Read a GeoTIFF into memory using Pillow when Rasterio is unavailable."""
    try:
        with Image.open(path) as image:
            array = np.asarray(image)
            grid = _grid_from_tiff_tags(image, path)
    except Exception as exc:  # pragma: no cover - depends on local image decoder
        logger.exception("Failed to read raster {} with Pillow", path)
        raise ProcessingError(f"Could not read raster: {path}") from exc

    if array.ndim == 2:
        data = array[np.newaxis, :, :]
    elif array.ndim == 3:
        data = np.moveaxis(array, -1, 0)
    else:
        raise ProcessingError(f"Unsupported raster rank for {path}")

    layer = RasterLayer(
        name=name or path.stem,
        data=data,
        grid=grid,
        metadata={"source_path": str(path)},
    )
    logger.info("Read raster {} using Pillow fallback", path)
    return layer


def _grid_from_tiff_tags(image: Image.Image, path: Path) -> RasterGrid:
    """Build a raster grid from GeoTIFF tags."""
    tags = cast(Any, image).tag_v2
    pixel_scale = tags.get(33550)
    tie_points = tags.get(33922)
    geokeys = tags.get(34735)
    nodata = _parse_nodata_tag(tags.get(42113))
    epsg = _parse_epsg_code(geokeys)
    if pixel_scale is None or tie_points is None or epsg is None:
        raise ProcessingError(f"Missing GeoTIFF georeferencing tags in {path}")
    if len(pixel_scale) < 2 or len(tie_points) < 6:
        raise ProcessingError(f"Invalid GeoTIFF georeferencing tags in {path}")

    scale_x = float(pixel_scale[0])
    scale_y = float(pixel_scale[1])
    tie_x = float(tie_points[3])
    tie_y = float(tie_points[4])
    transform = (scale_x, 0.0, tie_x, 0.0, -scale_y, tie_y)
    return RasterGrid(
        crs=f"EPSG:{epsg}",
        transform=transform,
        width=image.size[0],
        height=image.size[1],
        band_names=(path.stem,),
        nodata=nodata,
    )


def _parse_epsg_code(geokeys: tuple[int, ...] | None) -> int | None:
    """Extract the EPSG code from GeoKeyDirectoryTag values."""
    if geokeys is None or len(geokeys) < 4:
        return None
    index = 4
    while index + 3 < len(geokeys):
        key_id, _tag_location, _count, value = geokeys[index : index + 4]
        if key_id == 3072:
            return int(value)
        index += 4
    return None


def _parse_nodata_tag(value: object | None) -> float | int | None:
    """Parse a GeoTIFF nodata tag when present."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:  # pragma: no cover - fallback for odd encodings
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    if isinstance(value, (int, float)):
        return value
    return None


def write_raster(
    layer: RasterLayer,
    path: Path,
    *,
    driver: str = "GTiff",
    compress: str = "deflate",
) -> Path:
    """Write a raster layer using Rasterio when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not has_rasterio():
        fallback = path.with_suffix(".npz")

        np.savez_compressed(
            fallback,
            data=layer.data,
            grid=json.dumps(_grid_to_dict(layer.grid)),
        )
        logger.info("Wrote fallback raster archive {}", fallback)
        return fallback

    import rasterio
    from rasterio.crs import CRS

    profile = {
        "driver": driver,
        "height": layer.grid.height,
        "width": layer.grid.width,
        "count": layer.band_count,
        "dtype": str(layer.data.dtype),
        "crs": CRS.from_string(layer.grid.crs),
        "transform": layer.grid.transform,
        "nodata": layer.grid.nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if driver == "COG":
        profile.pop("tiled")
        profile.pop("blockxsize")
        profile.pop("blockysize")
        profile["blocksize"] = 512
        profile["overview_resampling"] = (
            "nearest" if np.issubdtype(layer.data.dtype, np.integer) else "average"
        )
    try:
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(layer.data)
            if layer.grid.band_names:
                dataset.descriptions = layer.grid.band_names
            tags = {
                key: json.dumps(value, default=str)
                for key, value in layer.metadata.items()
            }
            if tags:
                dataset.update_tags(**tags)
        logger.info("Wrote raster {}", path)
        return path
    except Exception as exc:  # pragma: no cover - depends on optional library
        logger.exception("Failed to write raster {}", path)
        raise ProcessingError(f"Could not write raster: {path}") from exc


def reproject_layer(
    layer: RasterLayer,
    *,
    target_crs: str,
    resampling: str = "bilinear",
) -> RasterLayer:
    """Reproject a raster layer when Rasterio is available."""
    if layer.grid.crs == target_crs:
        return layer
    if not has_rasterio():
        return _reproject_without_rasterio(
            layer,
            target_crs=target_crs,
            resampling=resampling,
        )

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import (
        calculate_default_transform,
        reproject,
    )

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    transform = Affine(*layer.grid.transform)
    source_crs = rasterio.crs.CRS.from_string(layer.grid.crs)
    destination_crs = rasterio.crs.CRS.from_string(target_crs)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        source_crs,
        destination_crs,
        layer.grid.width,
        layer.grid.height,
        *rasterio.transform.array_bounds(
            layer.grid.height,
            layer.grid.width,
            transform,
        ),
    )
    destination = np.empty(
        (layer.band_count, dst_height, dst_width),
        dtype=layer.data.dtype,
    )
    for band_index in range(layer.band_count):
        reproject(
            source=layer.data[band_index],
            destination=destination[band_index],
            src_transform=transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=destination_crs,
            resampling=resampling_map[resampling],
        )
    new_grid = RasterGrid(
        crs=target_crs,
        transform=tuple(dst_transform)[:6],
        width=dst_width,
        height=dst_height,
        band_names=layer.grid.band_names,
        nodata=layer.grid.nodata,
    )
    return layer.with_data(destination, grid=new_grid)


def _reproject_without_rasterio(
    layer: RasterLayer,
    *,
    target_crs: str,
    resampling: str,
) -> RasterLayer:
    """Reproject a raster layer with NumPy, SciPy, and pyproj."""
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates

    source_a, _, source_c, _, source_e, source_f = layer.grid.transform
    if source_a == 0 or source_e == 0:
        raise RasterDependencyError("Source grid transform must be north-up.")

    forward_transformer = Transformer.from_crs(
        layer.grid.crs,
        target_crs,
        always_xy=True,
    )
    inverse_transformer = Transformer.from_crs(
        target_crs,
        layer.grid.crs,
        always_xy=True,
    )
    corners = (
        (0.0, 0.0),
        (float(layer.grid.width), 0.0),
        (0.0, float(layer.grid.height)),
        (float(layer.grid.width), float(layer.grid.height)),
    )
    projected_corners = [
        forward_transformer.transform(
            source_c + (col * source_a),
            source_f + (row * source_e),
        )
        for col, row in corners
    ]
    xs = [point[0] for point in projected_corners]
    ys = [point[1] for point in projected_corners]
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)

    top_left = forward_transformer.transform(source_c, source_f)
    top_right = forward_transformer.transform(source_c + source_a, source_f)
    bottom_left = forward_transformer.transform(source_c, source_f + source_e)
    x_res = max(
        abs(top_right[0] - top_left[0]),
        abs(top_right[1] - top_left[1]),
        1e-9,
    )
    y_res = max(
        abs(bottom_left[0] - top_left[0]),
        abs(bottom_left[1] - top_left[1]),
        1e-9,
    )
    target_width = max(1, int(np.ceil((east - west) / x_res)))
    target_height = max(1, int(np.ceil((north - south) / y_res)))
    target_transform = (x_res, 0.0, west, 0.0, -y_res, north)

    target_cols = (np.arange(target_width, dtype=np.float64) + 0.5) * x_res + west
    target_rows = north - ((np.arange(target_height, dtype=np.float64) + 0.5) * y_res)
    target_xs, target_ys = np.meshgrid(target_cols, target_rows)
    source_xs, source_ys = inverse_transformer.transform(target_xs, target_ys)
    source_cols = (source_xs - source_c) / source_a - 0.5
    source_rows = (source_ys - source_f) / source_e - 0.5
    coordinates = np.vstack([source_rows.ravel(), source_cols.ravel()])

    order_map = {"nearest": 0, "bilinear": 1, "cubic": 3}
    order = order_map.get(resampling, 1)
    output_dtype = (
        layer.data.dtype if np.issubdtype(layer.data.dtype, np.floating) else np.float32
    )
    reprojected = np.empty(
        (layer.band_count, target_height, target_width), dtype=output_dtype
    )
    for band_index in range(layer.band_count):
        sampled = map_coordinates(
            layer.data[band_index].astype(np.float32),
            coordinates,
            order=order,
            mode="nearest",
        )
        reprojected[band_index] = sampled.reshape(target_height, target_width)

    new_grid = RasterGrid(
        crs=target_crs,
        transform=target_transform,
        width=target_width,
        height=target_height,
        band_names=layer.grid.band_names,
        nodata=layer.grid.nodata,
    )
    logger.info("Reprojected {} without Rasterio fallback", layer.name)
    return layer.with_data(reprojected, grid=new_grid)


def write_vrt(layers: tuple[RasterLayer, ...], path: Path) -> Path:
    """Write a simple VRT document referencing the source rasters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("VRTDataset")
    if layers:
        root.set("rasterXSize", str(layers[0].grid.width))
        root.set("rasterYSize", str(layers[0].grid.height))
    for layer in layers:
        band = ET.SubElement(root, "VRTRasterBand", dataType=str(layer.data.dtype))
        if layer.grid.band_names:
            band.set("bandName", layer.grid.band_names[0])
        source = ET.SubElement(band, "SimpleSource")
        src = ET.SubElement(source, "SourceFilename", relativeToVRT="0")
        src.text = str(layer.metadata.get("source_path", layer.name))
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    logger.info("Wrote VRT {}", path)
    return path


def write_cog_profile(
    grid: RasterGrid,
    *,
    compress: str = "deflate",
) -> dict[str, object]:
    """Return a COG-friendly profile for optional Rasterio writes."""
    return {
        "driver": "COG",
        "height": grid.height,
        "width": grid.width,
        "count": len(grid.band_names) or 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": grid.nodata,
        "compress": compress,
        "blocksize": 256,
    }


def _grid_to_dict(grid: RasterGrid) -> dict[str, object]:
    """Convert a grid to JSON-serializable metadata."""
    return {
        "crs": grid.crs,
        "transform": list(grid.transform),
        "width": grid.width,
        "height": grid.height,
        "band_names": list(grid.band_names),
        "nodata": grid.nodata,
    }
