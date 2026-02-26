import geopandas as gpd
from shapely.geometry import box

from app.rules import remove_small_industrial_utilities


def test_remove_small_industrial_utilities_drops_target_classes_below_threshold():
    gdf = gpd.GeoDataFrame(
        {
            "planning_z": [
                "Industrial/Utilities",
                "Offices, Retail Outlets",
                "Industrial/Utilities",
                "Residential",
            ],
            "geometry": [
                box(0, 0, 10, 10),   # 100 -> remove
                box(20, 0, 35, 20),  # 300 -> remove
                box(40, 0, 70, 20),  # 600 -> keep
                box(80, 0, 90, 10),  # non-target -> keep
            ],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    out, removed_count = remove_small_industrial_utilities(gdf, min_area_m2=200.0)

    assert removed_count == 1
    assert len(out) == 3
    assert set(out["planning_z"].tolist()) == {"Offices, Retail Outlets", "Industrial/Utilities", "Residential"}


def test_remove_small_industrial_utilities_handles_spacing_and_case():
    gdf = gpd.GeoDataFrame(
        {
            "planning_z": [" offices, retail outlets ", "INDUSTRIAL/UTILITIES", "Residential"],
            "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 70, 20)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    out, removed_count = remove_small_industrial_utilities(gdf, min_area_m2=150.0)

    assert removed_count == 1
    assert len(out) == 2
    assert set(out["planning_z"].tolist()) == {" offices, retail outlets ", "Residential"}
