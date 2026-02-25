from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

def run_pipeline(input_path: str, output_path: str, basemap: str = "google") -> dict[str, Any]:
    from app.cleaning import (
        commercial_industrial_merge_pass,
        simplify_geometry,
        topology_qa_and_fixes,
    )
    from app.io import (
        load_buildings,
        validate_input_shapefile,
        write_qa_report,
        write_qa_summary,
        write_shapefile,
    )
    from app.review import corner_cleaning_pass
    from app.rules import recategorize_small_garages

    # 1) Load and validate input
    validation = validate_input_shapefile(input_path)
    gdf = load_buildings(input_path)

    # 2) Simplify geometry
    gdf, simplify_tolerance, simplified_count = simplify_geometry(gdf, basemap=basemap)

    # 3) Topology QA/fixes (duplicates/overlaps/holes)
    gdf, topo_stats = topology_qa_and_fixes(gdf)

    # 4) Recategorize small garages
    gdf, recategorized_count = recategorize_small_garages(gdf)

    # 5) Corner-cleaning pass
    gdf, corner_changed_count = corner_cleaning_pass(gdf)

    # 6) Commercial/industrial merge pass
    before_merge_count = len(gdf)
    gdf = commercial_industrial_merge_pass(gdf)
    after_merge_count = len(gdf)

    # 7) Export outputs and QA report
    export_info = write_shapefile(gdf, output_path)

    qa_summary = {
        "count_simplified": simplified_count,
        "duplicates_removed": topo_stats["duplicate_removed_count"],
        "overlaps_fixed": topo_stats["overlap_fixed_count"],
        "holes_removed": topo_stats["holes_removed_count"],
        "holes_preserved": topo_stats["holes_preserved_count"],
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
            "simplify_tolerance": simplify_tolerance,
            "qa_summary": qa_summary,
            **topo_stats,
            "recategorized_small_garages": recategorized_count,
            "corner_cleaned_features": corner_changed_count,
            "feature_count_before_merge": before_merge_count,
            "feature_count_after_merge": after_merge_count,
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_pipeline(args.input, args.output, args.basemap)

    print("Pipeline complete")
    print(f"- Output shapefile: {report['export']['output_path']}")
    print(f"- Download bundle: {report['export']['bundle_zip']}")
    print(f"- QA report: {report['export']['qa_report_path']}")


if __name__ == "__main__":
    main()
