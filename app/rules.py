from __future__ import annotations

import geopandas as gpd


def recategorize_small_garages(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    out = gdf.copy()

    area_col = "area" if "area" in out.columns else None
    residential_col = "is_residen" if "is_residen" in out.columns else None

    if area_col is None:
        out["building_use_clean"] = "Unchanged"
        out["is_small_garage"] = False
        return out, 0

    area_vals = out[area_col]
    residential_mask = (
        out[residential_col].fillna("").astype(str).str.lower().isin(["yes", "true", "1"])
        if residential_col
        else True
    )

    garage_mask = area_vals.fillna(0).astype(float).lt(40.0) & residential_mask

    source_label_col = "planning_z" if "planning_z" in out.columns else None
    if source_label_col:
        out["building_use_clean"] = out[source_label_col].fillna("Unknown").astype(str)
    else:
        out["building_use_clean"] = "Unknown"

    out.loc[garage_mask, "building_use_clean"] = "Garage"
    out["is_small_garage"] = garage_mask
    return out, int(garage_mask.sum())
