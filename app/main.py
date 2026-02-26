from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

def run_pipeline(
    input_path: str,
    output_path: str,
    basemap: str = "google",
    garage_threshold_m2: float = 50.0,
    garage_reclass: str = "Shed",
) -> dict[str, Any]:
    from app.cleaning import (
        commercial_industrial_merge_pass,
        fill_inter_polygon_voids,
        fill_narrow_indents,
        remove_narrow_ledges,
        resolve_overlaps,
        simplify_geometry,
        strip_small_holes,
        topology_qa_and_fixes,
    )
    from app.io import (
        load_buildings,
        validate_input_shapefile,
        write_qa_report,
        write_qa_summary,
        write_review_layer,
        write_shapefile,
    )
    from app.review import run_corner_fix_review
    from app.rules import recategorize_small_garages, remove_small_industrial_utilities

    # 1) Load and validate input
    validation = validate_input_shapefile(input_path)
    gdf = load_buildings(input_path)

    # 2) Simplify geometry
    gdf, simplify_tolerance, simplified_count = simplify_geometry(gdf, basemap=basemap)

    # 3) Topology QA/fixes (duplicates/overlaps/holes)
    gdf, topo_stats = topology_qa_and_fixes(gdf)

    # 4) Recategorize small garages
    gdf, recategorized_count = recategorize_small_garages(
        gdf,
        garage_threshold_m2=garage_threshold_m2,
        garage_reclass=garage_reclass,
    )

    # 5) Remove small Industrial/Utilities buildings
    gdf, removed_small_target_count = remove_small_industrial_utilities(gdf, min_area_m2=200.0)

    # 6) Corner-cleaning pass (auto-clean + review queue)
    gdf, needs_review_layer, review_stats = run_corner_fix_review(gdf, basemap=basemap)

    # 7) Adjacent target merge pass (no recategorization)
    before_adjacent_target_merge_count = len(gdf)
    gdf = commercial_industrial_merge_pass(gdf)
    after_adjacent_target_merge_count = len(gdf)
    adjacent_target_merge_log = gdf.attrs.get("merge_log", [])
    adjacent_target_merge_stats = gdf.attrs.get("merge_stats", {})

    # 8) Fill narrow inward indents after topology + merge
    gdf, indent_stats = fill_narrow_indents(gdf)

    # 9) Remove narrow ledges after indent fill
    gdf, ledge_stats = remove_narrow_ledges(gdf)

    # 10) Final overlap cleanup after all geometry-modifying passes
    gdf, post_process_overlap_fixed_count = resolve_overlaps(gdf, overlap_area_threshold=0.0)
    topo_stats["overlap_fixed_count"] += post_process_overlap_fixed_count
    topo_stats["post_process_overlaps_fixed_count"] = post_process_overlap_fixed_count

    # 11) Fill enclosed voids formed between polygons
    gdf, inter_polygon_voids_filled_count = fill_inter_polygon_voids(gdf, min_void_area=0.0)
    topo_stats["inter_polygon_voids_filled_count"] = inter_polygon_voids_filled_count

    # 12) Re-run strict overlap cleanup after void filling
    gdf, post_void_overlap_fixed_count = resolve_overlaps(gdf, overlap_area_threshold=0.0)
    topo_stats["overlap_fixed_count"] += post_void_overlap_fixed_count
    topo_stats["post_void_overlaps_fixed_count"] = post_void_overlap_fixed_count

    # 13) Final hole cleanup after all geometry-modifying passes
    gdf, post_merge_hole_stats = strip_small_holes(gdf)
    topo_stats["holes_removed_count"] += post_merge_hole_stats["holes_removed_count"]
    topo_stats["holes_preserved_count"] += post_merge_hole_stats["holes_preserved_count"]
    topo_stats["post_merge_holes_removed_count"] = post_merge_hole_stats["holes_removed_count"]
    topo_stats["post_merge_holes_preserved_count"] = post_merge_hole_stats["holes_preserved_count"]

    topo_stats["ledge_fixed_count"] = ledge_stats["ledge_fixed_count"]
    topo_stats["ledge_removed_area_total"] = ledge_stats["ledge_removed_area_total"]
    topo_stats["ledge_skipped_count"] = ledge_stats["ledge_skipped_count"]
    topo_stats["indent_fixed_count"] = indent_stats["indent_fixed_count"]
    topo_stats["indent_filled_area_total"] = indent_stats["indent_filled_area_total"]
    topo_stats["indent_skipped_count"] = indent_stats["indent_skipped_count"]

    # 14) Export outputs and QA report
    export_info = write_shapefile(gdf, output_path)
    review_export = write_review_layer(needs_review_layer, output_path)
    if review_export is not None:
        export_info.update(review_export)

    qa_summary = {
        "count_simplified": simplified_count,
        "duplicates_removed": topo_stats["duplicate_removed_count"],
        "overlaps_fixed": topo_stats["overlap_fixed_count"],
        "holes_removed": topo_stats["holes_removed_count"],
        "holes_preserved": topo_stats["holes_preserved_count"],
        "indents_fixed": topo_stats["indent_fixed_count"],
        "ledges_fixed": topo_stats["ledge_fixed_count"],
    }

    qa_report = {
        "input": {
            "path": str(Path(input_path)),
            "row_count": validation.row_count,
            "missing_required_sidecars": validation.missing_required_sidecars,
            "present_optional_sidecars": validation.present_optional_sidecars,
        },
        "pipeline": {
            "basemap": basemap,
            "garage_threshold_m2": garage_threshold_m2,
            "garage_reclass": garage_reclass,
            "simplify_tolerance": simplify_tolerance,
            "qa_summary": qa_summary,
            **topo_stats,
            "recategorized_small_garages": recategorized_count,
            "removed_small_industrial_utilities": removed_small_target_count,
            "corner_cleaned_features": review_stats["auto_cleaned_count"],
            "corner_needs_review_features": review_stats["needs_review_count"],
            "review_basemap_provider": review_stats["provider"],
            "feature_count_before_adjacent_target_merge": before_adjacent_target_merge_count,
            "feature_count_after_adjacent_target_merge": after_adjacent_target_merge_count,
            "adjacent_target_merge_log_count": len(adjacent_target_merge_log),
            "adjacent_target_merge_log": adjacent_target_merge_log,
            "adjacent_target_target_candidate_count": adjacent_target_merge_stats.get("target_candidate_count", 0),
            "adjacent_target_merged_cluster_count": adjacent_target_merge_stats.get("merged_cluster_count", 0),
            "adjacent_target_merged_feature_count": adjacent_target_merge_stats.get("merged_feature_count", 0),
            "adjacent_target_accepted_tokens": adjacent_target_merge_stats.get("accepted_target_tokens", []),
            "adjacent_target_observed_top_tokens": adjacent_target_merge_stats.get("observed_top_planning_tokens", {}),
        },
        "export": export_info,
    }
    qa_path = write_qa_report(qa_report, output_path)
    qa_summary_path = write_qa_summary(qa_summary, output_path)
    qa_report["export"]["qa_report_path"] = str(qa_path)
    qa_report["export"]["qa_summary_path"] = str(qa_summary_path)
    return qa_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Building footprint cleaning pipeline")
    parser.add_argument("--input", required=True, help="Input buildings shapefile (.shp)")
    parser.add_argument("--output", required=True, help="Output cleaned shapefile (.shp)")
    parser.add_argument(
        "--basemap",
        default="google",
        choices=["google", "osm", "satellite"],
        help="Basemap profile to tune simplification",
    )
    parser.add_argument(
        "--garage-threshold-m2",
        type=float,
        default=50.0,
        help="Area threshold in square meters for reclassifying planning_z='Garage' features",
    )
    parser.add_argument(
        "--garage-reclass",
        default="Shed",
        help="Target category to assign to garages under the threshold",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_pipeline(
        args.input,
        args.output,
        args.basemap,
        garage_threshold_m2=args.garage_threshold_m2,
        garage_reclass=args.garage_reclass,
    )

    print("Pipeline complete")
    print(f"- Output shapefile: {report['export']['output_path']}")
    print(f"- Download bundle: {report['export']['bundle_zip']}")
    print(f"- QA report: {report['export']['qa_report_path']}")


if __name__ == "__main__":
    main()
