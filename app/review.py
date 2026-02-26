from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import math

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class BasemapProvider(ABC):
    """Abstract provider metadata used during operator review."""

    key: str
    display_name: str
    imagery_source: str

    @abstractmethod
    def attribution(self) -> str:
        """Human-readable attribution text for QA output."""


@dataclass(frozen=True)
class GoogleBasemapProvider(BasemapProvider):
    def attribution(self) -> str:
        return "Google imagery basemap"


@dataclass(frozen=True)
class OSMBasemapProvider(BasemapProvider):
    def attribution(self) -> str:
        return "OpenStreetMap standard basemap"


@dataclass(frozen=True)
class SatelliteBasemapProvider(BasemapProvider):
    def attribution(self) -> str:
        return "Generic satellite imagery basemap"


BASEMAP_PROVIDERS: dict[str, BasemapProvider] = {
    "google": GoogleBasemapProvider(
        key="google",
        display_name="Google",
        imagery_source="google",
    ),
    "osm": OSMBasemapProvider(
        key="osm",
        display_name="OpenStreetMap",
        imagery_source="osm",
    ),
    "satellite": SatelliteBasemapProvider(
        key="satellite",
        display_name="Satellite",
        imagery_source="satellite",
    ),
}


def get_basemap_provider(name: str) -> BasemapProvider:
    provider = BASEMAP_PROVIDERS.get(name.lower())
    if provider is None:
        supported = ", ".join(sorted(BASEMAP_PROVIDERS))
        raise ValueError(f"Unsupported basemap provider '{name}'. Supported values: {supported}")
    return provider


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    dot = (bax * bcx) + (bay * bcy)
    mag_ba = (bax**2 + bay**2) ** 0.5
    mag_bc = (bcx**2 + bcy**2) ** 0.5
    if mag_ba == 0 or mag_bc == 0:
        return 180.0
    cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return float(math.degrees(math.acos(cos_theta)))


def _right_angle_confidence(polygon: Polygon, tolerance_deg: float = 7.5) -> float:
    coords = list(polygon.exterior.coords)
    if len(coords) < 5:
        return 0.0

    rightish = 0
    corners = len(coords) - 1
    for i in range(corners):
        prev_pt = coords[i - 1]
        curr_pt = coords[i]
        next_pt = coords[(i + 1) % corners]
        candidate = _angle(prev_pt, curr_pt, next_pt)
        if abs(candidate - 90) <= tolerance_deg:
            rightish += 1
    return rightish / corners if corners else 0.0


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if isinstance(part, Polygon) and not part.is_empty]
    return []


def _right_angle_confidence_geometry(geometry, tolerance_deg: float = 7.5) -> float:
    parts = _polygon_parts(geometry)
    if not parts:
        return 0.0

    weighted_confidence = 0.0
    total_corners = 0
    for part in parts:
        corners = max(0, len(list(part.exterior.coords)) - 1)
        if corners == 0:
            continue
        weighted_confidence += _right_angle_confidence(part, tolerance_deg=tolerance_deg) * corners
        total_corners += corners
    return weighted_confidence / total_corners if total_corners else 0.0


def _compactness(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    perimeter = geometry.length
    area = geometry.area
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def _boundary_complexity(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    area = geometry.area
    perimeter = geometry.length
    if area <= 0:
        return 0.0
    return float(perimeter / math.sqrt(area))


def _auto_corner_fix(
    geometry,
    tolerance: float,
    closing_distance: float | None = None,
    max_closing_area_delta_ratio: float = 0.02,
):
    """Remove tiny spikes/slivers while keeping geometry topology-safe."""
    if geometry is None or geometry.is_empty:
        return geometry

    cleaned = geometry.buffer(0)
    cleaned = cleaned.simplify(tolerance=tolerance, preserve_topology=True)
    cleaned = cleaned.buffer(0)

    if closing_distance is None:
        closing_distance = max(tolerance * 1.6, 0.001)

    closed_candidate = cleaned.buffer(closing_distance).buffer(-closing_distance).buffer(0)
    cleaned_area = cleaned.area
    closed_area = closed_candidate.area if closed_candidate is not None and not closed_candidate.is_empty else 0.0
    area_delta_ratio = abs(closed_area - cleaned_area) / cleaned_area if cleaned_area > 0 else 1.0

    if area_delta_ratio <= max_closing_area_delta_ratio:
        return closed_candidate
    return cleaned


def run_corner_fix_review(
    gdf: gpd.GeoDataFrame,
    basemap: str,
    tolerance: float = 0.05,
    area_delta_threshold: float = 0.02,
    min_right_angle_confidence: float = 0.55,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, int | str]]:
    """Two-mode pass: high-confidence auto-clean + manual review queue."""
    provider = get_basemap_provider(basemap)
    out = gdf.copy()
    review_timestamp = datetime.now(timezone.utc).isoformat()

    out["reviewed"] = False
    out["review_action"] = "pending"
    out["review_notes"] = ""
    out["review_source"] = provider.display_name
    out["review_provider"] = provider.imagery_source
    out["review_attrib"] = provider.attribution()
    out["reviewed_ts"] = review_timestamp

    auto_fixed_count = 0
    needs_review_count = 0

    for idx, geom in out.geometry.items():
        if geom is None or geom.is_empty:
            out.at[idx, "review_action"] = "needs_review"
            out.at[idx, "review_notes"] = "Empty geometry"
            needs_review_count += 1
            continue

        before_area = geom.area
        before_compactness = _compactness(geom)
        before_complexity = _boundary_complexity(geom)

        candidate = _auto_corner_fix(
            geom,
            tolerance=tolerance,
            max_closing_area_delta_ratio=area_delta_threshold,
        )
        after_area = candidate.area if candidate is not None and not candidate.is_empty else 0.0
        area_delta_ratio = abs(after_area - before_area) / before_area if before_area > 0 else 1.0

        right_angle_confidence = _right_angle_confidence_geometry(candidate)
        after_compactness = _compactness(candidate)
        after_complexity = _boundary_complexity(candidate)

        compactness_gain = max(0.0, after_compactness - before_compactness)
        complexity_reduction = max(0.0, before_complexity - after_complexity)
        notch_cleaned = compactness_gain >= 0.004 and complexity_reduction >= 0.015

        notch_score = (
            (0.6 * min(compactness_gain / 0.006, 1.0))
            + (0.4 * min(complexity_reduction / 0.02, 1.0))
        )
        confidence = max(right_angle_confidence, notch_score)

        if area_delta_ratio <= area_delta_threshold and confidence >= min_right_angle_confidence:
            out.at[idx, "geometry"] = candidate
            out.at[idx, "reviewed"] = True
            out.at[idx, "review_action"] = "auto_cleaned"
            notch_note = "notch cleaned" if notch_cleaned else "spike/sliver cleaned"
            out.at[idx, "review_notes"] = (
                f"Auto cleaned ({notch_note}); area_delta_ratio={area_delta_ratio:.4f}; "
                f"confidence={confidence:.2f}; right_angle_confidence={right_angle_confidence:.2f}; "
                f"compactness_gain={compactness_gain:.4f}; complexity_reduction={complexity_reduction:.4f}"
            )
            auto_fixed_count += 1
        else:
            reason = "manual review required"
            if not notch_cleaned:
                reason = "manual review required (notch unresolved)"
            out.at[idx, "review_action"] = "needs_review"
            out.at[idx, "review_notes"] = (
                f"{reason}; area_delta_ratio={area_delta_ratio:.4f}; confidence={confidence:.2f}; "
                f"right_angle_confidence={right_angle_confidence:.2f}; compactness_gain={compactness_gain:.4f}; "
                f"complexity_reduction={complexity_reduction:.4f}"
            )
            needs_review_count += 1

    needs_review_layer = out[out["review_action"] == "needs_review"].copy()
    needs_review_layer.attrs.update(out.attrs)
    stats = {
        "provider": provider.key,
        "auto_cleaned_count": auto_fixed_count,
        "needs_review_count": needs_review_count,
    }
    return out, needs_review_layer, stats


def corner_cleaning_pass(gdf: gpd.GeoDataFrame, tolerance: float = 0.05) -> tuple[gpd.GeoDataFrame, int]:
    """
    Backward-compatible wrapper around the two-mode corner-fix workflow.
    """
    out, _, stats = run_corner_fix_review(gdf, basemap="google", tolerance=tolerance)
    out["corner_reviewed"] = out["reviewed"]
    return out, int(stats["auto_cleaned_count"])
