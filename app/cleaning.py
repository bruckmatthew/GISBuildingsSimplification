from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd
import geopandas as gpd
from shapely import make_valid, normalize, set_precision
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


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


def _longest_shared_boundary(shared_edge) -> float:
    if shared_edge is None or shared_edge.is_empty:
        return 0.0
    return float(shared_edge.length)


def _is_commercial_or_industrial(value: object) -> bool:
    """Return True only for explicit planning categories targeted by the merge pass."""
    if value is None:
        return False

    text = str(value).strip().lower()
    if not text:
        return False

    accepted = {
        "offices, retail outlets",
        "industrial/utilities",
    }
    return text in accepted


def _blocked_by_barriers(
    shared_edge,
    barriers_union,
    tol: float = 0.01,
) -> bool:
    if barriers_union is None or shared_edge.is_empty:
        return False
    barrier_cut = shared_edge.buffer(tol, cap_style=2).intersection(barriers_union)
    return not barrier_cut.is_empty


def _pick_representative_row(cluster: gpd.GeoDataFrame, source_id_col: str) -> pd.Series:
    areas = cluster.geometry.area
    ranked = cluster.assign(_area_rank=areas).sort_values(["_area_rank", source_id_col], ascending=[False, True])
    return ranked.iloc[0]


def commercial_industrial_merge_pass(
    gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame | None = None,
    parcels_gdf: gpd.GeoDataFrame | None = None,
    min_shared_edge_m: float = 1.0,
) -> gpd.GeoDataFrame:
    """Merge adjacent eligible target classes while preserving original planning categories.

    This pass only groups adjacent features in accepted planning classes and never creates
    synthetic planning category values.
    """
    out = gdf.copy()
    use_col = "planning_z" if "planning_z" in out.columns else None
    if use_col is None:
        return out

    out["source_geom_id"] = out.index.map(lambda idx: f"geom_{idx}")
    out["merge_pass"] = "original"
    out["merge_cluster_id"] = pd.NA
    out["merged_from_ids"] = pd.NA

    target_mask = out[use_col].map(_is_commercial_or_industrial)
    if not target_mask.any():
        out.attrs["merge_log"] = []
        return out

    target = out[target_mask].copy()
    non_target = out[~target_mask].copy()

    barrier_geoms = []
    if roads_gdf is not None and not roads_gdf.empty:
        barrier_geoms.extend([geom for geom in roads_gdf.geometry if geom is not None and not geom.is_empty])
    if parcels_gdf is not None and not parcels_gdf.empty:
        barrier_geoms.extend([geom for geom in parcels_gdf.geometry if geom is not None and not geom.is_empty])
    barriers_union = unary_union(barrier_geoms) if barrier_geoms else None

    sindex = target.sindex
    parent = {idx: idx for idx in target.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    index_labels = target.index.to_numpy()
    for pos, (idx, geom) in enumerate(target.geometry.items()):
        if geom is None or geom.is_empty:
            continue
        candidates = [cand_pos for cand_pos in sindex.intersection(geom.bounds) if cand_pos > pos]
        for cand_pos in candidates:
            cand = index_labels[cand_pos]
            other = target.geometry.iloc[cand_pos]
            if other is None or other.is_empty:
                continue
            shared_edge = geom.boundary.intersection(other.boundary)
            shared_len = _longest_shared_boundary(shared_edge)
            if shared_len < min_shared_edge_m:
                continue
            if _blocked_by_barriers(shared_edge, barriers_union):
                continue
            union(idx, cand)

    cluster_members = defaultdict(list)
    for idx in target.index:
        cluster_members[find(idx)].append(idx)

    merged_rows: list[dict] = []
    merge_log: list[dict[str, object]] = []
    cluster_num = 1

    for members in cluster_members.values():
        members_sorted = sorted(members)
        cluster = target.loc[members_sorted].copy()
        rep = _pick_representative_row(cluster, "source_geom_id")
        cluster_id = f"ci_cluster_{cluster_num:04d}"

        if len(cluster) == 1:
            row = rep.drop(labels=["_area_rank"], errors="ignore").to_dict()
            row["merge_cluster_id"] = cluster_id
            row["merged_from_ids"] = rep["source_geom_id"]
            merged_rows.append(row)
            merge_log.append(
                {
                    "cluster_id": cluster_id,
                    "before_ids": [rep["source_geom_id"]],
                    "after_id": rep["source_geom_id"],
                    "member_count": 1,
                }
            )
            cluster_num += 1
            continue

        merged_geom = unary_union(cluster.geometry.tolist())
        source_ids = sorted(cluster["source_geom_id"].astype(str).tolist())
        row = rep.drop(labels=["_area_rank"], errors="ignore").to_dict()
        row["geometry"] = merged_geom
        row["merge_pass"] = "merged"
        row["merge_cluster_id"] = cluster_id
        row["merged_from_ids"] = "|".join(source_ids)
        merged_rows.append(row)
        merge_log.append(
            {
                "cluster_id": cluster_id,
                "before_ids": source_ids,
                "after_id": cluster_id,
                "member_count": len(source_ids),
            }
        )
        cluster_num += 1

    merged_target = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=gdf.crs)
    combined = pd.concat([non_target, merged_target], ignore_index=True, sort=False)
    result = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    result.attrs["merge_log"] = merge_log
    return result
