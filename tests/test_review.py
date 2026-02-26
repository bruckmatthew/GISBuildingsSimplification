import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from app.review import run_corner_fix_review


def _notched_rectangle():
    # 10x10 shell with a narrow 1x2 inward notch on the top edge.
    return Polygon(
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


def test_corner_fix_auto_cleans_narrow_notch_with_small_area_change():
    geom = _notched_rectangle()
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, needs_review, stats = run_corner_fix_review(
        gdf,
        basemap="google",
        tolerance=0.35,
        area_delta_threshold=0.03,
        min_right_angle_confidence=0.55,
    )

    cleaned = out.geometry.iloc[0]
    assert stats["auto_cleaned_count"] == 1
    assert len(needs_review) == 0
    assert out.loc[0, "review_action"] == "auto_cleaned"
    assert "notch cleaned" in out.loc[0, "review_notes"]

    # Notch should be filled while preserving footprint scale.
    assert cleaned.area > geom.area
    assert abs(cleaned.area - geom.area) / geom.area < 0.03
    notch_region = Polygon([(5.0, 8.0), (6.0, 8.0), (6.0, 10.0), (5.0, 10.0)])
    assert cleaned.intersection(notch_region).area > 1.5


def test_corner_fix_evaluates_multipolygon_confidence_for_auto_clean():
    notched = _notched_rectangle()
    second = Polygon([(20, 0), (28, 0), (28, 8), (20, 8), (20, 0)])
    multi = MultiPolygon([notched, second])

    gdf = gpd.GeoDataFrame({"geometry": [multi]}, geometry="geometry", crs="EPSG:3857")

    out, needs_review, stats = run_corner_fix_review(
        gdf,
        basemap="osm",
        tolerance=0.35,
        area_delta_threshold=0.03,
        min_right_angle_confidence=0.55,
    )

    assert stats["auto_cleaned_count"] == 1
    assert len(needs_review) == 0
    assert out.loc[0, "review_action"] == "auto_cleaned"
    assert "confidence=" in out.loc[0, "review_notes"]


def test_corner_fix_default_tolerance_cleans_scaled_narrow_indent_without_vertex_explosion():
    # Large footprint with a very narrow inward slit artifact.
    geom = Polygon(
        [
            (0, 0),
            (120, 0),
            (120, 100),
            (62, 100),
            (62, 65),
            (58, 65),
            (58, 100),
            (0, 100),
            (0, 0),
        ]
    )
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs="EPSG:3857")

    out, needs_review, stats = run_corner_fix_review(
        gdf,
        basemap="satellite",
        area_delta_threshold=0.02,
        min_right_angle_confidence=0.55,
    )

    cleaned = out.geometry.iloc[0]
    indent_region = Polygon([(58.0, 65.0), (62.0, 65.0), (62.0, 100.0), (58.0, 100.0)])

    assert stats["auto_cleaned_count"] == 1
    assert len(needs_review) == 0
    assert out.loc[0, "review_action"] == "auto_cleaned"
    assert "notch cleaned" in out.loc[0, "review_notes"]
    assert abs(cleaned.area - geom.area) / geom.area < 0.02
    assert cleaned.intersection(indent_region).area > 130.0

    original_vertices = len(list(geom.exterior.coords)) - 1
    cleaned_vertices = len(list(cleaned.exterior.coords)) - 1
    assert cleaned_vertices <= original_vertices + 12
