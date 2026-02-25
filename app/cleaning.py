from __future__ import annotations

import hashlib

import pandas as pd
import geopandas as gpd
from shapely import make_valid, normalize, set_precision
from shapely.geometry import MultiPolygon, Polygon


def _strip_small_holes(geom, min_hole_area: float = 1.0, remove_all_holes: bool = True):
    if geom is None or geom.is_empty:
        return geom, 0, 0

    removed_count = 0
    preserved_count = 0

    def filter_polygon(poly: Polygon) -> Polygon:
        nonlocal removed_count, preserved_count
        holes = []
        for ring in poly.interiors:
            hole = Polygon(ring)
            if remove_all_holes:
                removed_count += 1
                continue

            if hole.area >= min_hole_area:
                holes.append(ring)
                preserved_count += 1
            else:
                removed_count += 1
        return Polygon(poly.exterior, holes)

    if isinstance(geom, Polygon):
        return filter_polygon(geom), removed_count, preserved_count
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([filter_polygon(p) for p in geom.geoms]), removed_count, preserved_count
    return geom, 0, 0


def strip_small_holes(
    gdf: gpd.GeoDataFrame,
    min_hole_area: float = 25.0,
    remove_all_holes: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """
    Remove holes from polygonal geometries and report counts.

    By default this strips all holes so no courtyards/interior voids remain.
    Set `remove_all_holes=False` to keep larger holes using `min_hole_area`.
    The threshold is expressed in square meters in the default pipeline flow
    because geometries are loaded into a metric CRS in `app.io`.
    """
    out = gdf.copy()

    hole_removed_count = 0
    hole_preserved_count = 0
    cleaned_geoms = []
    for geom in out.geometry:
        cleaned, removed, preserved = _strip_small_holes(
            geom,
            min_hole_area=min_hole_area,
            remove_all_holes=remove_all_holes,
        )
        cleaned_geoms.append(cleaned)
        hole_removed_count += removed
        hole_preserved_count += preserved

    out.geometry = cleaned_geoms
    return out, {
        "holes_removed_count": hole_removed_count,
        "holes_preserved_count": hole_preserved_count,
    }


def _normalized_geom_hash(geom, snap_grid_m: float = 0.1) -> str:
    if geom is None or geom.is_empty:
        return ""

    rounded = set_precision(geom, grid_size=snap_grid_m) if snap_grid_m > 0 else geom

    canonical = normalize(rounded)
    return hashlib.sha1(canonical.wkb).hexdigest()


def simplify_geometry(
    gdf: gpd.GeoDataFrame,
    basemap: str,
    tolerance_m: float | None = None,
) -> tuple[gpd.GeoDataFrame, float, int]:
    tol_map = {
        "google": 0.2,
        "osm": 0.35,
        "satellite": 0.15,
    }
    tolerance = tolerance_m if tolerance_m is not None else tol_map.get(basemap.lower(), 0.25)

    out = gdf.copy()
    original_wkb = out.geometry.to_wkb()
    out.geometry = out.geometry.simplify(tolerance=tolerance, preserve_topology=True)
    simplified_count = int((original_wkb != out.geometry.to_wkb()).sum())
    return out, tolerance, simplified_count


def topology_qa_and_fixes(
    gdf: gpd.GeoDataFrame,
    overlap_area_threshold: float = 0.5,
    min_hole_area: float = 25.0,
    near_duplicate_grid_m: float = 0.1,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    out = gdf.copy()

    invalid_mask = ~out.geometry.is_valid
    invalid_fixed_count = int(invalid_mask.sum())
    if invalid_fixed_count:
        out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask, "geometry"].map(make_valid)

    before = len(out)
    out["_geom_wkb"] = out.geometry.to_wkb()
    out = out.drop_duplicates(subset=["_geom_wkb"]).drop(columns=["_geom_wkb"])
    exact_duplicate_removed_count = before - len(out)

    before_near = len(out)
    out["_near_hash"] = out.geometry.map(lambda geom: _normalized_geom_hash(geom, near_duplicate_grid_m))
    out = out.drop_duplicates(subset=["_near_hash"]).drop(columns=["_near_hash"])
    near_duplicate_removed_count = before_near - len(out)
    duplicate_removed_count = exact_duplicate_removed_count + near_duplicate_removed_count

    overlap_fixed_count = 0
    if len(out) > 1:
        sindex = out.sindex
        checked = set()
        updated_geoms = out.geometry.copy()
        for idx, geom in out.geometry.items():
            for cand in sindex.intersection(geom.bounds):
                if cand == idx:
                    continue
                key = tuple(sorted((idx, cand)))
                if key in checked:
                    continue
                checked.add(key)
                if idx not in updated_geoms.index or cand not in updated_geoms.index:
                    continue

                a = updated_geoms.loc[idx]
                b = updated_geoms.loc[cand]
                if a is None or b is None or a.is_empty or b.is_empty:
                    continue
                if not a.intersects(b):
                    continue

                inter_area = a.intersection(b).area
                if inter_area <= overlap_area_threshold:
                    continue

                if a.area >= b.area:
                    fixed = make_valid(b.difference(a))
                    updated_geoms.loc[cand] = fixed
                else:
                    fixed = make_valid(a.difference(b))
                    updated_geoms.loc[idx] = fixed
                overlap_fixed_count += 1

        out.geometry = updated_geoms
        out = out[~out.geometry.is_empty].copy()

    out, hole_stats = strip_small_holes(
        out,
        min_hole_area=min_hole_area,
        remove_all_holes=True,
    )

    return out, {
        "invalid_fixed_count": invalid_fixed_count,
        "duplicate_removed_count": duplicate_removed_count,
        "exact_duplicate_removed_count": exact_duplicate_removed_count,
        "near_duplicate_removed_count": near_duplicate_removed_count,
        "overlap_fixed_count": overlap_fixed_count,
        "holes_removed_count": hole_stats["holes_removed_count"],
        "holes_preserved_count": hole_stats["holes_preserved_count"],
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
