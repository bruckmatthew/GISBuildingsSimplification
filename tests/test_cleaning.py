import geopandas as gpd
from shapely.geometry import Polygon, box

from app.cleaning import (
    _is_commercial_or_industrial,
    commercial_industrial_merge_pass,
    fill_narrow_indents,
    remove_narrow_ledges,
)


def test_is_commercial_or_industrial_accepts_only_explicit_values():
    assert _is_commercial_or_industrial(" Offices, Retail Outlets ")
    assert _is_commercial_or_industrial("industrial/utilities")

    assert not _is_commercial_or_industrial("Commercial/Business")
    assert not _is_commercial_or_industrial("Industrial Park")
    assert not _is_commercial_or_industrial("Offices and Retail Outlets")
    assert not _is_commercial_or_industrial("")
    assert not _is_commercial_or_industrial(None)


def test_is_commercial_or_industrial_handles_spacing_variants_for_offices_retail():
    assert _is_commercial_or_industrial("Offices,Retail Outlets")
    assert _is_commercial_or_industrial("Offices,  Retail Outlets")
    assert _is_commercial_or_industrial("Offices, Retail Outlets")
    assert _is_commercial_or_industrial("Offices / Retail Outlets")


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
    assert merged_row["planning_z"] != "Commercial/Industrial"
    assert merged_row["merged_from_ids"] == "geom_0|geom_1"


def test_merge_pass_merges_adjacent_offices_retail_features():
    gdf = gpd.GeoDataFrame(
        {
            "planning_z": ["Offices, Retail Outlets", "Offices, Retail Outlets", "Residential"],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(5, 5, 6, 6),
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


def test_merge_pass_handles_geographic_crs_with_default_threshold():
    gdf = gpd.GeoDataFrame(
        {
            "planning_z": ["Offices, Retail Outlets", "Offices, Retail Outlets"],
            "geometry": [
                box(0.0, 0.0, 0.01, 0.01),
                box(0.01, 0.0, 0.02, 0.01),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    out = commercial_industrial_merge_pass(gdf)

    merged = out[out["merge_pass"] == "merged"]
    assert len(merged) == 1
    assert merged.iloc[0]["planning_z"] == "Offices, Retail Outlets"


def test_merge_pass_sets_merge_stats_in_attrs():
    gdf = gpd.GeoDataFrame(
        {
            "planning_z": ["Offices, Retail Outlets", "Offices, Retail Outlets", "Residential"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )

    out = commercial_industrial_merge_pass(gdf, min_shared_edge_m=0.5)
    stats = out.attrs.get("merge_stats", {})

    assert stats.get("target_candidate_count") == 2
    assert stats.get("merged_cluster_count") == 1
    assert stats.get("merged_feature_count") == 2
    assert "offices retail outlets" in stats.get("accepted_target_tokens", [])
    assert stats.get("observed_top_planning_tokens", {}).get("offices retail outlets") == 2


def test_remove_narrow_ledges_removes_small_side_protrusion():
    base = box(0, 0, 10, 10)
    ledge = box(10, 4, 11, 6)
    geom = base.union(ledge)

    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, stats = remove_narrow_ledges(gdf, width_threshold_m=2.5, area_threshold_m2=3.0)

    assert stats["ledge_fixed_count"] == 1
    assert stats["ledge_removed_area_total"] > 1.5
    cleaned_geom = out.geometry.iloc[0]
    assert cleaned_geom.area < geom.area
    assert cleaned_geom.area > base.area - 0.1


def test_remove_narrow_ledges_skips_wide_or_large_removals():
    base = box(0, 0, 10, 10)
    wide_wing = box(10, 2, 14, 8)
    geom = base.union(wide_wing)

    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, stats = remove_narrow_ledges(gdf, width_threshold_m=2.0, area_threshold_m2=3.0)

    assert stats["ledge_fixed_count"] == 0
    assert out.geometry.iloc[0].equals_exact(geom, tolerance=1e-6)


def test_remove_narrow_ledges_handles_multipolygon_parts():
    poly_a = Polygon([(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)])
    poly_b = box(20, 20, 26, 26).union(box(26, 22, 27, 23))
    multi = poly_a.union(poly_b)

    gdf = gpd.GeoDataFrame({"geometry": [multi]}, geometry="geometry", crs="EPSG:3857")

    out, stats = remove_narrow_ledges(gdf, width_threshold_m=2.2, area_threshold_m2=2.0)

    assert stats["ledge_fixed_count"] == 1
    assert out.geometry.iloc[0].area < multi.area


def test_fill_narrow_indents_fills_small_inward_notch():
    geom = Polygon(
        [
            (0, 0),
            (10, 0),
            (10, 10),
            (6, 10),
            (6, 8),
            (5, 8),
            (5, 10),
            (0, 10),
            (0, 0),
        ]
    )

    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, stats = fill_narrow_indents(gdf, width_threshold_m=2.0, area_threshold_m2=3.0)

    assert stats["indent_fixed_count"] == 1
    cleaned_geom = out.geometry.iloc[0]
    assert cleaned_geom.area > geom.area




def test_fill_narrow_indents_uses_multiscale_close_for_wider_notch():
    geom = Polygon(
        [
            (0, 0),
            (30, 0),
            (30, 20),
            (18, 20),
            (18, 16),
            (14, 16),
            (14, 20),
            (0, 20),
            (0, 0),
        ]
    )

    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, stats = fill_narrow_indents(gdf, width_threshold_m=1.2, area_threshold_m2=10.0)

    assert stats["indent_fixed_count"] >= 1
    cleaned_geom = out.geometry.iloc[0]
    assert cleaned_geom.area > geom.area + 8.0

def test_fill_narrow_indents_skips_large_area_gains():
    geom = Polygon(
        [
            (0, 0),
            (12, 0),
            (12, 10),
            (10, 10),
            (10, 4),
            (2, 4),
            (2, 10),
            (0, 10),
            (0, 0),
        ]
    )

    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, stats = fill_narrow_indents(gdf, width_threshold_m=5.0, area_threshold_m2=40.0)

    assert stats["indent_fixed_count"] == 0
    assert stats["indent_skipped_count"] == 1
    assert out.geometry.iloc[0].equals_exact(geom, tolerance=1e-6)
