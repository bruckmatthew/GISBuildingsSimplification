from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd
import geopandas as gpd
from shapely import make_valid, normalize, set_precision
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
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


def _polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if part is not None and not part.is_empty]
    if isinstance(geom, GeometryCollection):
        parts = []
        for sub_geom in geom.geoms:
            parts.extend(_polygon_parts(sub_geom))
        return parts
    return []


def _polygonal_or_original(geom):
    """Return polygonal components when available, otherwise return original geometry."""
    parts = _polygon_parts(geom)
    if not parts:
        return geom
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def _minimum_width_estimate(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    rect = geom.minimum_rotated_rectangle
    if rect.is_empty:
        return 0.0
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return 0.0
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        edges.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return float(min(edges)) if edges else 0.0


def _is_small_narrow_removed_piece(piece, width_threshold_m: float, area_threshold_m2: float) -> bool:
    if piece is None or piece.is_empty:
        return True

    for component in _polygon_parts(piece):
        if component.area > area_threshold_m2:
            return False
        if _minimum_width_estimate(component) > width_threshold_m:
            return False
    return True


def remove_narrow_ledges(
    gdf: gpd.GeoDataFrame,
    width_threshold_m: float = 2.5,
    area_threshold_m2: float = 50.0,
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    """Remove narrow protrusions via morphological opening when loss is controlled."""
    out = gdf.copy()
    cleaned_geoms = []

    ledge_fixed_count = 0
    ledge_removed_area_total = 0.0
    ledge_skipped_count = 0

    base_distance = max(width_threshold_m / 2.0, 0.01)
    opening_distances = (base_distance * 0.75, base_distance, base_distance * 1.25)
    max_loss_ratio = 0.25

    for geom in out.geometry:
        if geom is None or geom.is_empty:
            cleaned_geoms.append(geom)
            continue

        valid_geom = _polygonal_or_original(make_valid(geom))
        if valid_geom.is_empty:
            cleaned_geoms.append(geom)
            continue

        original_area = float(valid_geom.area)
        current_geom = valid_geom
        removed_area_accum = 0.0

        for distance in opening_distances:
            opened = make_valid(current_geom.buffer(-distance, join_style=2).buffer(distance, join_style=2))
            opened = _polygonal_or_original(opened)
            if opened.is_empty:
                continue

            removed = make_valid(current_geom.difference(opened))
            removed = _polygonal_or_original(removed)
            removed_area = float(removed.area) if not removed.is_empty else 0.0
            if removed_area <= 0.0:
                continue

            area_loss_ratio = removed_area / original_area if original_area > 0 else 0.0
            if area_loss_ratio > max_loss_ratio:
                continue

            should_replace = _is_small_narrow_removed_piece(
                removed,
                width_threshold_m=width_threshold_m,
                area_threshold_m2=area_threshold_m2,
            )
            if not should_replace:
                continue

            candidate = _polygonal_or_original(make_valid(current_geom.difference(removed)))
            if candidate.is_empty:
                continue

            current_geom = candidate
            removed_area_accum += removed_area

        cumulative_loss = removed_area_accum / original_area if original_area > 0 else 0.0
        if removed_area_accum > 0 and cumulative_loss <= 0.35:
            cleaned_geoms.append(current_geom)
            ledge_fixed_count += 1
            ledge_removed_area_total += removed_area_accum
        else:
            cleaned_geoms.append(valid_geom)
            if removed_area_accum > 0:
                ledge_skipped_count += 1

    out.geometry = cleaned_geoms
    return out, {
        "ledge_fixed_count": int(ledge_fixed_count),
        "ledge_removed_area_total": float(ledge_removed_area_total),
        "ledge_skipped_count": int(ledge_skipped_count),
    }


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


def _normalize_planning_z_text(value: object) -> str:
    """Normalize planning_z text for strict-yet-robust category matching."""
    text = str(value).replace(" ", " ").strip().lower()
    text = " ".join(text.split())
    return text


def _planning_z_tokens(value: object) -> tuple[str, ...]:
    """Tokenize planning text while ignoring punctuation separators."""
    text = _normalize_planning_z_text(value)
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return tuple(tok for tok in cleaned.split() if tok)


def _accepted_target_tokens() -> set[tuple[str, ...]]:
    """Accepted planning_z token sequences for the adjacent target merge pass."""
    return {
        ("offices", "retail", "outlets"),
        ("industrial", "utilities"),
    }


def _is_commercial_or_industrial(value: object) -> bool:
    """Return True only for explicit planning categories targeted by the merge pass."""
    if value is None:
        return False

    tokens = _planning_z_tokens(value)
    return tokens in _accepted_target_tokens()


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
    areas = cluster.geometry.map(lambda geom: 0.0 if geom is None else float(geom.area))
    ranked = cluster.assign(_area_rank=areas).sort_values(["_area_rank", source_id_col], ascending=[False, True])
    return ranked.iloc[0]


def _effective_min_shared_edge(gdf: gpd.GeoDataFrame, min_shared_edge_m: float) -> float:
    """Use the caller threshold for projected data and relax it for geographic CRS."""
    crs = getattr(gdf, "crs", None)
    if crs is not None and getattr(crs, "is_geographic", False):
        return 0.0
    return float(min_shared_edge_m)


def commercial_industrial_merge_pass(
    gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame | None = None,
    parcels_gdf: gpd.GeoDataFrame | None = None,
    min_shared_edge_m: float = 1.0,
) -> gpd.GeoDataFrame:
    """Merge adjacent eligible target classes while preserving original planning categories.

    This pass only groups adjacent features in accepted planning classes and never creates
    synthetic planning category values. For geographic CRS inputs (degrees), the minimum
    shared-edge threshold is relaxed so adjacency is still detected.
    """
    out = gdf.copy()
    use_col = "planning_z" if "planning_z" in out.columns else None
    if use_col is None:
        return out

    out["source_geom_id"] = out.index.map(lambda idx: f"geom_{idx}")
    out["merge_pass"] = "original"
    out["merge_cluster_id"] = pd.NA
    out["merged_from_ids"] = pd.NA

    planning_tokens = out[use_col].map(_planning_z_tokens)
    accepted_tokens = _accepted_target_tokens()
    target_mask = planning_tokens.map(lambda t: t in accepted_tokens)
    target_count = int(target_mask.sum())

    observed_token_counts = (
        planning_tokens.value_counts(dropna=False)
        .head(10)
        .to_dict()
    )
    observed_token_counts = {" ".join(k): int(v) for k, v in observed_token_counts.items()}
    accepted_token_labels = [" ".join(t) for t in sorted(accepted_tokens)]

    if not target_mask.any():
        out.attrs["merge_log"] = []
        out.attrs["merge_stats"] = {
            "target_candidate_count": target_count,
            "merged_cluster_count": 0,
            "merged_feature_count": 0,
            "accepted_target_tokens": accepted_token_labels,
            "observed_top_planning_tokens": observed_token_counts,
        }
        return out

    target = out[target_mask].copy()
    non_target = out[~target_mask].copy()

    barrier_geoms = []
    if roads_gdf is not None and not roads_gdf.empty:
        barrier_geoms.extend([geom for geom in roads_gdf.geometry if geom is not None and not geom.is_empty])
    if parcels_gdf is not None and not parcels_gdf.empty:
        barrier_geoms.extend([geom for geom in parcels_gdf.geometry if geom is not None and not geom.is_empty])
    barriers_union = unary_union(barrier_geoms) if barrier_geoms else None

    min_shared_edge = _effective_min_shared_edge(out, min_shared_edge_m)

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
            if shared_len < min_shared_edge:
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
            row[use_col] = rep[use_col]
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
        row[use_col] = rep[use_col]
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
    merged_clusters = sum(1 for entry in merge_log if int(entry.get("member_count", 0)) > 1)
    merged_members = sum(int(entry.get("member_count", 0)) for entry in merge_log if int(entry.get("member_count", 0)) > 1)
    result.attrs["merge_stats"] = {
        "target_candidate_count": target_count,
        "merged_cluster_count": int(merged_clusters),
        "merged_feature_count": int(merged_members),
        "accepted_target_tokens": accepted_token_labels,
        "observed_top_planning_tokens": observed_token_counts,
    }
    return result
