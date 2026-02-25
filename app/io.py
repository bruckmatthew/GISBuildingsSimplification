from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd

REQUIRED_SIDECARS = (".shx", ".dbf", ".prj")
OPTIONAL_SIDECARS = (".cpg",)
REQUIRED_ATTRIBUTE_FIELDS = ("planning_z",)


@dataclass
class InputValidationResult:
    path: Path
    row_count: int
    missing_required_sidecars: list[str]
    present_optional_sidecars: list[str]


def _sidecar_path(shp_path: Path, suffix: str) -> Path:
    return shp_path.with_suffix(suffix)


def validate_input_shapefile(input_path: str | Path) -> InputValidationResult:
    shp_path = Path(input_path)
    if shp_path.suffix.lower() != ".shp":
        raise ValueError(f"Input must be a .shp file, got: {shp_path}")
    if not shp_path.exists():
        raise FileNotFoundError(f"Input shapefile not found: {shp_path}")

    missing_required = [
        suffix for suffix in REQUIRED_SIDECARS if not _sidecar_path(shp_path, suffix).exists()
    ]
    if missing_required:
        missing_display = ", ".join(missing_required)
        raise FileNotFoundError(
            "Input shapefile is missing required sidecar files "
            f"for '{shp_path.name}': {missing_display}."
        )

    present_optional = [
        suffix for suffix in OPTIONAL_SIDECARS if _sidecar_path(shp_path, suffix).exists()
    ]

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Input shapefile loaded but has no features.")
    if "geometry" not in gdf.columns:
        raise ValueError("Input layer has no geometry column.")

    missing_fields = [field for field in REQUIRED_ATTRIBUTE_FIELDS if field not in gdf.columns]
    if missing_fields:
        missing_fields_display = ", ".join(missing_fields)
        raise ValueError(
            f"Input layer is missing required attribute field(s): {missing_fields_display}."
        )

    return InputValidationResult(
        path=shp_path,
        row_count=len(gdf),
        missing_required_sidecars=missing_required,
        present_optional_sidecars=present_optional,
    )


def load_buildings(input_path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(input_path)

    if gdf.crs is None:
        raise ValueError("Input layer has no CRS information (.prj is required).")

    original_crs = gdf.crs
    gdf.attrs["original_crs"] = original_crs
    gdf.attrs["input_crs_was_geographic"] = bool(original_crs.is_geographic)

    if original_crs.is_geographic:
        metric_crs = gdf.estimate_utm_crs() or "EPSG:3857"
        gdf = gdf.to_crs(metric_crs)
        gdf.attrs["original_crs"] = original_crs
        gdf.attrs["input_crs_was_geographic"] = True
        gdf.attrs["working_metric_crs"] = metric_crs

    return gdf


def write_shapefile(gdf: gpd.GeoDataFrame, output_path: str | Path) -> dict[str, Any]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    export_gdf = gdf
    original_crs = gdf.attrs.get("original_crs")
    if original_crs is not None and gdf.crs is not None and str(original_crs) != str(gdf.crs):
        export_gdf = gdf.to_crs(original_crs)

    export_gdf.to_file(out)

    shapefile_group = []
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        candidate = out.with_suffix(suffix)
        if candidate.exists():
            shapefile_group.append(candidate)

    missing_required = [
        suffix for suffix in REQUIRED_SIDECARS if not out.with_suffix(suffix).exists()
    ]

    bundle_path = out.with_name(f"{out.stem}_bundle.zip")
    with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for member in shapefile_group:
            zf.write(member, arcname=member.name)

    return {
        "output_path": str(out),
        "created_files": [str(p) for p in shapefile_group],
        "bundle_zip": str(bundle_path),
        "missing_required_sidecars": missing_required,
        "output_crs": str(export_gdf.crs) if export_gdf.crs is not None else None,
        "original_input_crs": str(original_crs) if original_crs is not None else None,
    }


def write_review_layer(gdf: gpd.GeoDataFrame, output_path: str | Path) -> dict[str, Any] | None:
    if "review_action" not in gdf.columns:
        return None

    review_candidates = gdf[gdf["review_action"] == "needs_review"].copy()
    if review_candidates.empty:
        return None

    out = Path(output_path)
    review_path = out.with_name(f"{out.stem}_needs_review.shp")

    original_crs = gdf.attrs.get("original_crs")
    if original_crs is not None and review_candidates.crs is not None:
        if str(original_crs) != str(review_candidates.crs):
            review_candidates = review_candidates.to_crs(original_crs)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_candidates.to_file(review_path)

    review_files = []
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        candidate = review_path.with_suffix(suffix)
        if candidate.exists():
            review_files.append(candidate)

    return {
        "review_layer_path": str(review_path),
        "review_created_files": [str(path) for path in review_files],
        "review_feature_count": int(len(review_candidates)),
    }


def write_qa_report(report: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    qa_path = out.with_name(f"{out.stem}_qa_report.json")
    qa_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return qa_path


def write_qa_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    summary_path = out.with_name(f"{out.stem}_qa_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
