from __future__ import annotations

import geopandas as gpd


def corner_cleaning_pass(gdf: gpd.GeoDataFrame, tolerance: float = 0.05) -> tuple[gpd.GeoDataFrame, int]:
    """
    Lightweight corner-cleaning hook.

    This pass keeps topology intact while gently smoothing tiny spikes to improve
    manual review outcomes.
    """
    out = gdf.copy()
    before = out.geometry
    out.geometry = out.geometry.simplify(tolerance=tolerance, preserve_topology=True)

    changed = int((before.to_wkb() != out.geometry.to_wkb()).sum())
    out["corner_reviewed"] = changed > 0
    return out, changed
