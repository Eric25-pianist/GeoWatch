"""Administrative boundary discovery, validation, provenance, and preview."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log10
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from loguru import logger
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import make_valid

from geowatch.core.errors import GeoWatchError
from geowatch.utils.geometry import load_vector_geometry

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "GeoWatch/0.1 (local research GIS application)"
BoundarySearchKind = Literal["auto", "city", "district", "state", "urban"]


@dataclass(frozen=True)
class BoundaryCandidate:
    """One ranked administrative boundary with complete provenance."""

    name: str
    display_name: str
    country_code: str | None
    administrative_level: str | None
    source: str
    source_url: str
    license: str
    geometry: BaseGeometry
    centroid: tuple[float, float]
    area_sq_km: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return WGS84 geometry bounds."""
        west, south, east, north = self.geometry.bounds
        return float(west), float(south), float(east), float(north)

    @property
    def part_count(self) -> int:
        """Return the number of polygon parts in the boundary."""
        if self.geometry.geom_type == "MultiPolygon":
            return len(self.geometry.geoms)
        return 1

    @property
    def bbox_span_degrees(self) -> tuple[float, float]:
        """Return longitude and latitude span of the boundary extent."""
        west, south, east, north = self.bounds
        return float(east - west), float(north - south)

    @property
    def bbox_to_area_ratio(self) -> float:
        """Return projected bounding-box area divided by actual AOI area."""
        if self.area_sq_km <= 0:
            return float("inf")
        return _area_sq_km(box(*self.bounds)) / self.area_sq_km

    @property
    def is_spatially_dispersed(self) -> bool:
        """Return True when a boundary likely includes remote islands or exclaves."""
        lon_span, lat_span = self.bbox_span_degrees
        return (
            (self.part_count >= 4 and max(lon_span, lat_span) >= 2.0)
            or (self.part_count >= 2 and max(lon_span, lat_span) >= 8.0)
            or (self.part_count >= 3 and self.bbox_to_area_ratio >= 25.0)
        )


def search_boundaries(
    location: str,
    country: str,
    region: str | None = None,
    *,
    boundary_kind: BoundarySearchKind = "city",
    limit: int = 5,
) -> tuple[BoundaryCandidate, ...]:
    """Search OpenStreetMap for polygonal administrative candidates."""
    candidates: list[BoundaryCandidate] = []
    seen_sources: set[str] = set()
    queries = _boundary_queries(location, country, region, boundary_kind)
    for query in queries:
        for candidate in _search_nominatim_query(query, location, limit=limit):
            if candidate.source_url in seen_sources:
                continue
            seen_sources.add(candidate.source_url)
            candidates.append(candidate)
        if len(candidates) >= max(limit * 2, 8):
            break
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda candidate: _candidate_rank(
                candidate, location, boundary_kind
            ),
        )
        return tuple(ranked[:limit])
    attempted = "; ".join(queries)
    raise GeoWatchError(
        "No polygon boundary found. Tried: "
        f"{attempted}. Try a district/division name or supply a local boundary file."
    )


def _search_nominatim_query(
    query: str,
    location: str,
    *,
    limit: int,
) -> tuple[BoundaryCandidate, ...]:
    """Search one Nominatim query and return polygonal candidates only."""
    params = urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "polygon_geojson": 1,
            "addressdetails": 1,
            "extratags": 1,
            "limit": limit,
        }
    )
    payload = _get_json(f"{NOMINATIM_URL}?{params}")
    if not isinstance(payload, list):
        raise GeoWatchError("Boundary service returned an invalid response.")
    candidates: list[BoundaryCandidate] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("geojson"), dict):
            continue
        geometry = _validate_geometry(shape(item["geojson"]))
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        raw_address = item.get("address")
        raw_extra = item.get("extratags")
        address: dict[str, object] = (
            raw_address if isinstance(raw_address, dict) else {}
        )
        extra: dict[str, object] = raw_extra if isinstance(raw_extra, dict) else {}
        if not _is_administrative_boundary(item, extra):
            continue
        candidate = BoundaryCandidate(
            name=str(item.get("name") or location),
            display_name=str(item.get("display_name") or query),
            country_code=_optional_string(address.get("country_code")),
            administrative_level=_optional_string(extra.get("admin_level")),
            source="OpenStreetMap Nominatim",
            source_url=(
                "https://www.openstreetmap.org/"
                f"{item.get('osm_type', 'relation')}/{item.get('osm_id', '')}"
            ),
            license="Open Database License (ODbL)",
            geometry=geometry,
            centroid=(float(geometry.centroid.x), float(geometry.centroid.y)),
            area_sq_km=_area_sq_km(geometry),
        )
        candidates.append(candidate)
    return tuple(candidates)


def _is_administrative_boundary(
    item: dict[str, object],
    extra: dict[str, object],
) -> bool:
    """Return True only for administrative boundary features."""
    item_class = _optional_string(item.get("class"))
    item_type = _optional_string(item.get("type"))
    boundary_tag = _optional_string(extra.get("boundary"))
    admin_level = _optional_string(extra.get("admin_level"))
    return (
        admin_level is not None
        or boundary_tag == "administrative"
        or (item_class == "boundary" and item_type == "administrative")
    )


def _boundary_queries(
    location: str,
    country: str,
    region: str | None,
    boundary_kind: BoundarySearchKind = "city",
) -> tuple[str, ...]:
    """Build forgiving administrative-boundary queries for common naming patterns."""
    location = location.strip()
    country = country.strip()
    region = region.strip() if region else None
    base = ", ".join(value for value in (location, region, country) if value)
    names_by_kind: dict[BoundarySearchKind, tuple[str, ...]] = {
        "auto": (
            location,
            f"{location} City",
            f"City of {location}",
            f"{location} Municipality",
            f"{location} District",
            f"District {location}",
            f"{location} Division",
            f"{location} Province",
            f"{location} Prefecture",
            f"{location} State",
            f"{location} Metropolitan",
        ),
        "city": (
            f"{location} City",
            f"City of {location}",
            f"{location} Municipality",
            f"{location} Metropolitan Municipality",
            f"{location} District",
            location,
        ),
        "district": (
            f"{location} District",
            f"District {location}",
            f"{location} Division",
            f"{location} County",
            f"{location} Governorate",
            location,
        ),
        "state": (
            f"{location} Province",
            f"{location} Prefecture",
            f"{location} State",
            f"{location} Region",
            f"{location} Metropolitan",
            location,
        ),
        "urban": (
            f"{location} special wards",
            f"{location} 23 special wards",
            f"Special wards of {location}",
            f"{location} urban area",
            f"{location} City",
            f"City of {location}",
            location,
        ),
    }
    names = names_by_kind[boundary_kind]
    queries = [base]
    for name in names:
        query = ", ".join(value for value in (name, region, country) if value)
        queries.append(query)
    queries.append(", ".join(value for value in (location, country) if value))
    return tuple(dict.fromkeys(query for query in queries if query))


def boundary_warning_messages(
    candidate: BoundaryCandidate,
    *,
    requested_kind: BoundarySearchKind = "auto",
) -> tuple[str, ...]:
    """Return non-fatal warnings that require careful user confirmation."""
    warnings: list[str] = []
    level = _admin_level_int(candidate)
    lon_span, lat_span = candidate.bbox_span_degrees
    if candidate.is_spatially_dispersed:
        warnings.append(
            "This boundary is multipart and geographically dispersed "
            f"({candidate.part_count} parts spanning {lon_span:.1f} deg longitude "
            f"and {lat_span:.1f} deg latitude). It may include islands, exclaves, "
            "or distant administrative areas."
        )
    if requested_kind in {"city", "urban"} and level is not None and level <= 4:
        warnings.append(
            "The selected candidate is a high-level state/prefecture-style "
            "boundary, not a city/municipality boundary. If you expected an "
            "urban core, reject it and choose a ward/municipality boundary or "
            "provide a local official file."
        )
    if requested_kind == "urban" and candidate.area_sq_km > 2_500:
        warnings.append(
            "The selected area is much larger than a typical urban-core map. "
            "Confirm only if you intentionally want the whole administrative "
            "region."
        )
    return tuple(warnings)


def candidate_from_file(path: Path, *, name: str) -> BoundaryCandidate:
    """Load a user boundary and wrap it in the same provenance contract."""
    loaded = load_vector_geometry(path)
    geometry = _validate_geometry(loaded.geometry)
    if loaded.crs != "EPSG:4326":
        transformer = Transformer.from_crs(loaded.crs, "EPSG:4326", always_xy=True)
        geometry = transform(transformer.transform, geometry)
    return BoundaryCandidate(
        name=name,
        display_name=f"{name} (user-provided boundary)",
        country_code=None,
        administrative_level=None,
        source="User-provided file",
        source_url=str(path.resolve()),
        license="User supplied; verify reuse rights",
        geometry=geometry,
        centroid=(float(geometry.centroid.x), float(geometry.centroid.y)),
        area_sq_km=_area_sq_km(geometry),
    )


def validate_candidate(
    candidate: BoundaryCandidate,
    *,
    expected_country_code: str | None = None,
) -> tuple[str, ...]:
    """Validate geometry, location, coordinate ranges, and plausible area."""
    findings: list[str] = []
    geometry = _validate_geometry(candidate.geometry)
    west, south, east, north = geometry.bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise GeoWatchError("Boundary coordinates are outside valid WGS84 ranges.")
    if candidate.area_sq_km <= 0:
        raise GeoWatchError("Boundary has zero or negative area.")
    centroid = Point(candidate.centroid)
    if not geometry.covers(centroid):
        findings.append("Geometry centroid falls outside a concave/multipart boundary.")
    if (
        expected_country_code
        and candidate.country_code
        and candidate.country_code.casefold() != expected_country_code.casefold()
    ):
        raise GeoWatchError("Boundary country does not match the requested country.")
    findings.append("Geometry is valid and polygonal.")
    findings.append(f"Boundary area is {candidate.area_sq_km:,.2f} km2.")
    findings.append(f"Boundary contains {candidate.part_count} polygon part(s).")
    lon_span, lat_span = candidate.bbox_span_degrees
    findings.append(
        "Boundary extent spans "
        f"{lon_span:.2f} deg longitude by {lat_span:.2f} deg latitude."
    )
    findings.append("Boundary coordinates are valid WGS84 longitude/latitude values.")
    return tuple(findings)


def save_boundary_candidate(
    candidate: BoundaryCandidate,
    *,
    source_path: Path,
    validated_path: Path,
    metadata_path: Path,
    requested_kind: BoundarySearchKind = "auto",
) -> tuple[Path, Path, Path]:
    """Persist original/validated GeoJSON and a provenance record."""
    for path in (source_path, validated_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    feature = {
        "type": "Feature",
        "properties": {
            "name": candidate.name,
            "source": candidate.source,
            "source_url": candidate.source_url,
            "license": candidate.license,
            "area_sq_km": candidate.area_sq_km,
        },
        "geometry": mapping(candidate.geometry),
    }
    collection = {"type": "FeatureCollection", "features": [feature]}
    encoded = json.dumps(collection, indent=2)
    source_path.write_text(encoded, encoding="utf-8")
    validated_path.write_text(encoded, encoding="utf-8")
    metadata = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source": candidate.source,
        "source_url": candidate.source_url,
        "license": candidate.license,
        "administrative_level": candidate.administrative_level,
        "display_name": candidate.display_name,
        "area_sq_km": candidate.area_sq_km,
        "centroid": candidate.centroid,
        "bbox": candidate.bounds,
        "part_count": candidate.part_count,
        "bbox_span_degrees": candidate.bbox_span_degrees,
        "bbox_to_area_ratio": candidate.bbox_to_area_ratio,
        "boundary_warnings": boundary_warning_messages(
            candidate, requested_kind=requested_kind
        ),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved validated boundary to {}", validated_path)
    return source_path, validated_path, metadata_path


def render_boundary_preview(candidate: BoundaryCandidate, path: Path) -> Path:
    """Render a clean boundary preview for terminal confirmation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    _plot_geometry(axis, candidate.geometry)
    axis.set_title(_preview_title(candidate), fontsize=11)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#d1d5db", linewidth=0.5, linestyle=":")
    axis.set_aspect("equal", adjustable="datalim")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def _plot_geometry(axis: Any, geometry: BaseGeometry) -> None:
    polygons = (
        list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    )
    for polygon in polygons:
        x_values, y_values = polygon.exterior.xy
        axis.fill(
            x_values, y_values, color="#dbeafe", edgecolor="#075985", linewidth=1.5
        )


def _validate_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        raise GeoWatchError("Boundary geometry is empty.")
    repaired = geometry if geometry.is_valid else make_valid(geometry)
    if repaired.is_empty or not repaired.is_valid:
        raise GeoWatchError("Boundary geometry could not be repaired safely.")
    return repaired


def _area_sq_km(geometry: BaseGeometry) -> float:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    return float(transform(transformer.transform, geometry).area / 1_000_000)


def _preview_title(candidate: BoundaryCandidate) -> str:
    """Return an ASCII-safe preview title for headless Windows rendering."""
    for value in (candidate.display_name, candidate.name):
        cleaned = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", errors="ignore")
            .decode("ascii")
            .strip(" ,")
        )
        if cleaned:
            return cleaned
    if candidate.country_code:
        return f"Administrative Boundary ({candidate.country_code.upper()})"
    return "Administrative Boundary Preview"


def _get_json(url: str) -> object:
    request = Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.exception("Boundary request failed: {}", url)
        raise GeoWatchError(
            "Boundary service is unavailable. Use a local boundary file."
        ) from exc


def _candidate_rank(
    candidate: BoundaryCandidate,
    location: str,
    boundary_kind: BoundarySearchKind,
) -> tuple[int, float, str]:
    """Rank candidates by requested administrative intent and plausibility."""
    level = _admin_level_int(candidate)
    preferred_levels: dict[BoundarySearchKind, tuple[int, ...]] = {
        "auto": (5, 6, 7, 8, 4),
        "city": (6, 7, 8, 9),
        "district": (5, 6, 7),
        "state": (3, 4),
        "urban": (8, 9, 10, 7),
    }
    if level is None:
        level_penalty = 5
    else:
        level_penalty = min(
            abs(level - value) for value in preferred_levels[boundary_kind]
        )
    text = _normalize_name(f"{candidate.name} {candidate.display_name}")
    location_text = _normalize_name(location)
    name_penalty = 0 if location_text and location_text in text else 2
    dispersed_penalty = (
        6
        if boundary_kind in {"auto", "city", "urban"}
        and candidate.is_spatially_dispersed
        else 0
    )
    area_penalty = log10(max(candidate.area_sq_km, 1.0))
    if boundary_kind == "state":
        area_penalty = 0.0
    return (
        level_penalty + name_penalty + dispersed_penalty,
        area_penalty,
        candidate.display_name,
    )


def _admin_level_int(candidate: BoundaryCandidate) -> int | None:
    """Return numeric OSM admin_level when available."""
    if candidate.administrative_level is None:
        return None
    try:
        return int(candidate.administrative_level)
    except ValueError:
        return None


def _normalize_name(value: str) -> str:
    """Normalize display names for loose ranking comparisons."""
    return " ".join(
        unicodedata.normalize("NFKD", value)
        .encode("ascii", errors="ignore")
        .decode("ascii")
        .casefold()
        .replace(",", " ")
        .split()
    )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
