from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd

REQUIRED_SIDECARS = (".shx", ".dbf")
OPTIONAL_SIDECARS = (".prj", ".cpg", ".qix")


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
    present_optional = [
        suffix for suffix in OPTIONAL_SIDECARS if _sidecar_path(shp_path, suffix).exists()
    ]

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Input shapefile loaded but has no features.")
    if "geometry" not in gdf.columns:
        raise ValueError("Input layer has no geometry column.")

    return InputValidationResult(
        path=shp_path,
        row_count=len(gdf),
        missing_required_sidecars=missing_required,
        present_optional_sidecars=present_optional,
    )


def load_buildings(input_path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(input_path)


def write_shapefile(gdf: gpd.GeoDataFrame, output_path: str | Path) -> dict[str, Any]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out)

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
    }


def write_qa_report(report: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    qa_path = out.with_name(f"{out.stem}_qa_report.json")
    qa_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return qa_path
