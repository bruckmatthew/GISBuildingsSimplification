# GIS Buildings Simplification

Python CLI pipeline for cleaning building footprint shapefiles.

## Project layout

- `app/main.py` - CLI entrypoint and `run_pipeline` orchestrator.
- `app/io.py` - read/write shapefiles, sidecar validation, QA report writing.
- `app/cleaning.py` - simplification, topology QA/fixes, commercial/industrial merge pass.
- `app/rules.py` - attribute recategorization rules (small garages).
- `app/review.py` - manual corner-fix workflow hook.

## Install

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If script execution is blocked in PowerShell, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`


## Run

### macOS / Linux

```bash
python -m app.main --input <buildings.shp> --output <cleaned.shp> --basemap google
```

### Windows PowerShell

```powershell
python -m app.main --input <buildings.shp> --output <cleaned.shp> --basemap google
```

Outputs created next to `<cleaned.shp>`:

- cleaned shapefile sidecar set (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` when available)
- downloadable bundle zip: `<cleaned_stem>_bundle.zip`
- QA report JSON: `<cleaned_stem>_qa_report.json`
- QA summary JSON: `<cleaned_stem>_qa_summary.json`
