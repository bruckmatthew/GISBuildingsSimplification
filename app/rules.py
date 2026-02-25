from __future__ import annotations

import geopandas as gpd
import pandas as pd


def _area_square_meters(gdf: gpd.GeoDataFrame) -> pd.Series:
    if gdf.crs is None:
        metric_gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:6933")
        return metric_gdf.geometry.area

    if gdf.crs.is_geographic:
        try:
            metric_crs = gdf.estimate_utm_crs()
            if metric_crs is None:
                metric_crs = "EPSG:6933"
            metric_gdf = gdf.to_crs(metric_crs)
        except Exception:
            metric_gdf = gdf.to_crs("EPSG:6933")
        return metric_gdf.geometry.area

    axis_info = getattr(gdf.crs, "axis_info", None) or []
    is_meter_crs = any(getattr(axis, "unit_name", "").lower() in {"metre", "meter"} for axis in axis_info)
    if is_meter_crs:
        return gdf.geometry.area

    return gdf.to_crs("EPSG:6933").geometry.area


def recategorize_small_garages(
    gdf: gpd.GeoDataFrame,
    garage_threshold_m2: float = 50.0,
    garage_reclass: str = "Shed",
) -> tuple[gpd.GeoDataFrame, int]:
    out = gdf.copy()

    planning_col = "planning_z" if "planning_z" in out.columns else None
    if planning_col is None:
        out["area_m2"] = _area_square_meters(out)
        out["recat_reason"] = out.get("recat_reason", "")
        return out, 0

    out["area_m2"] = _area_square_meters(out)
    garage_mask = (
        out[planning_col].fillna("").astype(str).eq("Garage")
        & out["area_m2"].fillna(0).lt(float(garage_threshold_m2))
    )

    out["building_use_clean"] = out[planning_col].fillna("Unknown").astype(str)
    out["is_small_garage"] = garage_mask
    out["recat_reason"] = out.get("recat_reason", "")

    out.loc[garage_mask, "building_use_clean"] = garage_reclass
    threshold_label = f"{garage_threshold_m2:g}".replace(".", "p")
    out.loc[garage_mask, "recat_reason"] = f"garage_lt_{threshold_label}m2"

    return out, int(garage_mask.sum())
