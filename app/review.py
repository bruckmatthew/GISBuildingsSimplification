from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import math

import geopandas as gpd
from shapely.geometry import Polygon


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


def _auto_corner_fix(geometry, tolerance: float):
    """Remove tiny spikes/slivers while keeping geometry topology-safe."""
    if geometry is None or geometry.is_empty:
        return geometry

    cleaned = geometry.buffer(0)
    cleaned = cleaned.simplify(tolerance=tolerance, preserve_topology=True)
    return cleaned.buffer(0)


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
        candidate = _auto_corner_fix(geom, tolerance=tolerance)
        after_area = candidate.area if candidate is not None and not candidate.is_empty else 0.0
        area_delta_ratio = abs(after_area - before_area) / before_area if before_area > 0 else 1.0

        confidence = 0.0
        if hasattr(candidate, "geom_type") and candidate.geom_type == "Polygon":
            confidence = _right_angle_confidence(candidate)

        if area_delta_ratio <= area_delta_threshold and confidence >= min_right_angle_confidence:
            out.at[idx, "geometry"] = candidate
            out.at[idx, "reviewed"] = True
            out.at[idx, "review_action"] = "auto_cleaned"
            out.at[idx, "review_notes"] = (
                f"Auto cleaned spike/sliver artifacts; area_delta_ratio={area_delta_ratio:.4f}; "
                f"right_angle_confidence={confidence:.2f}"
            )
            auto_fixed_count += 1
        else:
            out.at[idx, "review_action"] = "needs_review"
            out.at[idx, "review_notes"] = (
                f"Manual review required; area_delta_ratio={area_delta_ratio:.4f}; "
                f"right_angle_confidence={confidence:.2f}"
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
