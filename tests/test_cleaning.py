import geopandas as gpd
from shapely.geometry import box

from app.cleaning import _is_commercial_or_industrial, commercial_industrial_merge_pass


def test_is_commercial_or_industrial_accepts_only_explicit_values():
    assert _is_commercial_or_industrial(" Offices, Retail Outlets ")
    assert _is_commercial_or_industrial("industrial/utilities")

    assert not _is_commercial_or_industrial("Commercial/Business")
    assert not _is_commercial_or_industrial("Industrial Park")
    assert not _is_commercial_or_industrial("")
    assert not _is_commercial_or_industrial(None)


def test_merge_pass_preserves_representative_planning_category_after_merge():
    gdf = gpd.GeoDataFrame(
        {
            "planning_z": ["Offices, Retail Outlets", "Industrial/Utilities", "Residential"],
            "name": ["a", "b", "c"],
            "geometry": [
                box(0, 0, 2, 1),  # area=2 (representative in merged cluster)
                box(2, 0, 3, 1),  # area=1
                box(10, 10, 11, 11),
            ],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    out = commercial_industrial_merge_pass(gdf, min_shared_edge_m=0.5)

    merged = out[out["merge_pass"] == "merged"]
    assert len(merged) == 1

    merged_row = merged.iloc[0]
    assert merged_row["planning_z"] == "Offices, Retail Outlets"
    assert merged_row["merged_from_ids"] == "geom_0|geom_1"
