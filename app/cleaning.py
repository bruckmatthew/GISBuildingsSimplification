from __future__ import annotations

import pandas as pd
import geopandas as gpd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon


def _strip_small_holes(geom, min_hole_area: float = 1.0):
    if geom is None or geom.is_empty:
        return geom

    def filter_polygon(poly: Polygon) -> Polygon:
        holes = []
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.area >= min_hole_area:
                holes.append(ring)
        return Polygon(poly.exterior, holes)

    if isinstance(geom, Polygon):
        return filter_polygon(geom)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([filter_polygon(p) for p in geom.geoms])
    return geom


def simplify_geometry(gdf: gpd.GeoDataFrame, basemap: str) -> tuple[gpd.GeoDataFrame, float]:
    tol_map = {
        "google": 0.15,
        "osm": 0.25,
        "satellite": 0.1,
    }
    tolerance = tol_map.get(basemap.lower(), 0.2)

    out = gdf.copy()
    out.geometry = out.geometry.simplify(tolerance=tolerance, preserve_topology=True)
    out.geometry = out.geometry.map(_strip_small_holes)
    return out, tolerance


def topology_qa_and_fixes(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    out = gdf.copy()

    invalid_mask = ~out.geometry.is_valid
    invalid_fixed_count = int(invalid_mask.sum())
    if invalid_fixed_count:
        out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask, "geometry"].map(make_valid)

    before = len(out)
    out["_geom_wkb"] = out.geometry.to_wkb()
    out = out.drop_duplicates(subset=["_geom_wkb"]).drop(columns=["_geom_wkb"])
    duplicate_removed_count = before - len(out)

    overlap_flagged_count = 0
    if len(out) > 1:
        sindex = out.sindex
        checked = set()
        for idx, geom in out.geometry.items():
            for cand in sindex.intersection(geom.bounds):
                if cand == idx:
                    continue
                key = tuple(sorted((idx, cand)))
                if key in checked:
                    continue
                checked.add(key)
                other = out.loc[cand, "geometry"]
                if geom.intersects(other) and geom.intersection(other).area > 0:
                    overlap_flagged_count += 1

    return out, {
        "invalid_fixed_count": invalid_fixed_count,
        "duplicate_removed_count": duplicate_removed_count,
        "overlap_flagged_count": overlap_flagged_count,
    }


def commercial_industrial_merge_pass(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    use_col = "planning_z" if "planning_z" in out.columns else None
    if use_col is None:
        return out

    target_mask = out[use_col].fillna("").str.lower().str.contains("commercial|industrial")
    if not target_mask.any():
        out["merge_pass"] = "original"
        return out

    target = out[target_mask].copy()
    non_target = out[~target_mask].copy()

    merged_geom = target.dissolve().explode(index_parts=False).reset_index(drop=True)
    merged_geom[use_col] = "Commercial/Industrial"
    merged_geom["merge_pass"] = "merged"

    non_target["merge_pass"] = "original"
    combined = pd.concat([non_target, merged_geom], ignore_index=True, sort=False)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
