# GeoWatch

GeoWatch is a terminal-based GIS and remote-sensing application for comparing
satellite imagery through time. It guides a user from administrative-boundary
selection and imagery acquisition through cloud masking, raster alignment,
spectral indices, land-cover classification, change detection, quality review,
professional maps, reports, an offline dashboard, and portfolio exports.

The application is designed for students, researchers, and GIS analysts who
want a reproducible local workflow without manually assembling every processing
step in desktop GIS software.

> GeoWatch results are analytical products, not automatic ground truth.
> Unsupervised land-cover classifications are exploratory until validated with
> independent reference data.

## Highlights

- Beginner-friendly interactive workflow with `geowatch wizard`.
- Resumable, location-based projects with checksummed downloads and manifests.
- User-provided GeoJSON, Shapefile, or GeoPackage boundaries, or OpenStreetMap
  administrative-boundary discovery with preview and confirmation.
- Sentinel-2 Level-2A and Landsat Collection 2 Level-2 processing.
- Cloud, shadow, snow, fill, saturation, nodata, and outside-AOI masking.
- Reprojection to a suitable projected CRS, common-grid alignment, polygon
  clipping, seasonal compositing, and Cloud Optimized GeoTIFF output.
- Ten spectral indices: NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI, NDMI, GNDVI,
  and NBR.
- Index differencing, CVA, PCA, MAD, IRMAD, image ratioing, magnitude mapping,
  and categorical NDVI gain/no-change/loss products.
- Exploratory K-Means and ISODATA LULC, plus Random Forest, XGBoost, and SVM
  when labeled training data are supplied.
- Five reusable map themes: Academic Thesis, Government Report, Minimal
  Journal, Presentation, and Dark Dashboard.
- Publication maps, static HTML/PDF reports, an offline interactive dashboard,
  rule-based interpretation, a transparent quality score, and a portfolio
  package.

## Requirements

- Windows 10 or 11 is the best-supported setup path.
- Internet access is required for first-time installation, boundary search, and
  satellite acquisition.
- Several gigabytes of free disk space may be needed. Large AOIs and long time
  ranges can require substantially more.
- Python 3.12 is recommended. The Windows installer creates a project-local
  Python 3.12 environment with GDAL, Rasterio, PROJ, GEOS, and the scientific
  stack.

## Install on Windows

Open PowerShell in the project directory. If cloning from GitHub:

```powershell
git clone <repository-url> D:\ProjectGeoWatch
cd D:\ProjectGeoWatch
```

Create or update the reproducible project-local environment:

```powershell
.\setup-micromamba.ps1
```

The script downloads a local Micromamba runtime when needed, creates
`.mamba-root\envs\geowatch`, installs GeoWatch with that environment's Python,
and stops if any required validation fails. It does not fall back to the system
Python installation.

For the current PowerShell session, define these shortcuts:

```powershell
$mm = "D:\ProjectGeoWatch\.tools\Library\bin\micromamba.exe"
$root = "D:\ProjectGeoWatch\.mamba-root"
```

If Micromamba was already installed globally, `setup-micromamba.ps1` may report
a different executable path. Use the exact launch command printed by the setup
script in that case.

Verify the production environment:

```powershell
& $mm --root-prefix $root run -n geowatch geowatch doctor --strict
```

The strict doctor must pass Python, pip, GDAL, Rasterio, GeoPandas, Shapely,
PyProj, and the GeoWatch package before real processing.

### Experienced pip users

Pip installation is supported when compatible GDAL/Rasterio wheels and native
libraries are already available:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[geo,ml,dev]"
geowatch doctor --strict
```

For Windows, Micromamba remains the recommended path because it resolves the
native GIS libraries together.

## Quick Start

Start the guided application:

```powershell
& $mm --root-prefix $root run -n geowatch geowatch wizard
```

The wizard asks for the location, country/region, comparison years, seasonal
window, boundary, sensor/provider preferences, cloud limit, LULC method, and map
theme. It displays boundary candidates and requires confirmation before using
an online administrative boundary.

By default, the wizard creates the project and begins processing. To create and
inspect the specification without downloading imagery:

```powershell
& $mm --root-prefix $root run -n geowatch geowatch wizard --setup-only
```

Then run, resume, or inspect that saved project:

```powershell
& $mm --root-prefix $root run -n geowatch geowatch process outputs\Karachi\project.yaml
& $mm --root-prefix $root run -n geowatch geowatch resume outputs\Karachi\project.yaml
& $mm --root-prefix $root run -n geowatch geowatch status outputs\Karachi\project.yaml
& $mm --root-prefix $root run -n geowatch geowatch quality outputs\Karachi\project.yaml
```

Override the saved map theme for a processing run:

```powershell
& $mm --root-prefix $root run -n geowatch geowatch process outputs\Karachi\project.yaml --map-theme government
```

Available theme names are `academic`, `government`, `journal`, `presentation`,
and `dark`.

## Example Workflow

For a same-season endpoint comparison such as Karachi 2018 versus 2020:

1. Run `geowatch wizard --setup-only`.
2. Enter `Karachi`, `Pakistan`, and `Sindh` when prompted.
3. Enter `2018` as the start year and `2020` as the end year.
4. Choose the same month range for both endpoints.
5. Review the proposed boundary name, area, extent, and preview carefully.
6. Accept automatic sensor selection, or choose Sentinel-2/Landsat in advanced
   settings.
7. Review `outputs\Karachi\project.yaml`.
8. Run `geowatch process outputs\Karachi\project.yaml`.
9. If a network or download interruption occurs, use `geowatch resume`.

Do not enter `2018-2020` in a year prompt; enter one integer in each year field.

## Commands

| Command | Purpose |
| --- | --- |
| `geowatch doctor --strict` | Validate the active production GIS environment. |
| `geowatch wizard` | Create and optionally process a guided location project. |
| `geowatch process PROJECT` | Run a saved project from the beginning. |
| `geowatch resume PROJECT` | Continue without repeating verified stages. |
| `geowatch status PROJECT` | Show acquisition, processing, analytics, and publication state. |
| `geowatch quality PROJECT` | Print the exported run-quality summary. |
| `geowatch validate CONFIG` | Validate a YAML/JSON compatibility configuration. |
| `geowatch acquire CONFIG --download` | Search/download through the configuration workflow. |
| `geowatch publish CONFIG` | Generate compatibility publication outputs. |
| `geowatch init` | Create foundation folders and sample inputs. |
| `geowatch version` | Print the installed GeoWatch version. |

Run `geowatch --help` or `geowatch COMMAND --help` for the live option list.

## Repository Layout

```text
GeoWatch/
|-- .github/workflows/     Continuous-integration checks
|-- configs/               Default, example, and schema configurations
|-- scripts/               Maintenance and output-validation utilities
|-- src/geowatch/          Application package
|   |-- acquisition/       Catalog search, authentication, ranking, downloads
|   |-- analytics/         Indices, change detection, classification
|   |-- application/       Wizard, project layout, orchestration, manifests
|   |-- cartography/       Declarative professional map themes
|   |-- cli/               Typer command-line interface
|   |-- config/            Pydantic models and YAML/JSON loading
|   |-- portfolio/         Portfolio package generation
|   |-- processing/        Raster preprocessing, grids, masks, COG I/O
|   |-- reporting/         Maps, dashboard, reports, interpretation
|   `-- validation/        Environment checks and quality scoring
|-- tests/                 Unit, CLI, and integration tests
|-- environment.yml        Reproducible Python 3.12 GIS environment
|-- pyproject.toml         Package and tool configuration
|-- setup-micromamba.ps1   Recommended Windows installer
`-- README.md              Project documentation
```

Runtime folders such as `outputs/`, `data/`, `logs/`, `.mamba-root/`, and
`.tools/` are created locally and intentionally excluded from Git.

## Project Output Layout

Each wizard run creates a filesystem-safe location directory:

```text
outputs/<Location>/
|-- project.yaml                 Reusable run specification
|-- run_manifest.json            Resumable stage state
|-- boundary/                    Source, validated geometry, provenance, preview
|-- raw/<year>/                  Downloaded scenes and QA bands
|-- processed/<year>/            Masked, aligned surface-reflectance composites
|-- indices/                     T1, T2, differences, and index statistics
|-- classification/              LULC rasters and transition products
|-- change/                      Change scores, masks, gain/loss products
|-- statistics/                  CSV/JSON/Excel summaries
|-- maps/                        300/600 DPI PNG, JPEG, PDF, and SVG maps
|-- reports/
|   |-- dashboard.html           Portable offline interactive dashboard
|   |-- report.html              Static HTML report
|   |-- report.pdf               Publication report
|   `-- interpretation.md        Rule-based scientific interpretation
|-- validation/                  Quality score and component explanations
|-- exports/                     GeoTIFF and vector/table deliverables
|-- portfolio_exports/           Shareable infographic, brief, maps, metadata
|-- logs/                        Project processing logs
`-- cache/                       Resumable local intermediates
```

The dashboard works locally without a web server. Keep its neighboring output
folders together so relative map and download links remain valid.

## Data, Methods, and Providers

### Boundaries

The wizard can search OpenStreetMap Nominatim for polygonal administrative
boundaries or accept a local GeoJSON, Shapefile, or GeoPackage. It checks
geometry validity, CRS, coordinate ranges, plausible area, and provenance, then
creates a preview for human confirmation. Boundary names and administrative
levels vary between countries; the user remains responsible for choosing the
intended legal/statistical boundary.

### Imagery

- Sentinel-2 Level-2A (available from 2015; typically 10-20 m analytical bands).
- Landsat 5 TM Collection 2 Level 2 (1984-2012).
- Landsat 7 ETM+ Collection 2 Level 2 (1999-present; SLC-off after May 2003).
- Landsat 8 OLI Collection 2 Level 2 (2013-present).
- Landsat 9 OLI-2 Collection 2 Level 2 (2021-present).

The professional workflow uses Microsoft Planetary Computer STAC by default.
Provider connectors also exist for Copernicus Data Space, USGS, NASA Earthdata,
and generic STAC workflows, but authenticated services require valid accounts,
current endpoints, and credentials. GeoWatch attempts to keep one mission and
the same seasonal window across comparison years. Any cloud/season fallback is
reported for approval.

Credentials are read only from environment variables, never committed project
files. Supported prefixes are `GEOWATCH_COPERNICUS`,
`GEOWATCH_PLANETARY_COMPUTER`, `GEOWATCH_NASA_EARTHDATA`, and `GEOWATCH_USGS`,
with `_USERNAME`, `_PASSWORD`, or `_TOKEN` suffixes as applicable.

### Analysis

Surface-reflectance scale/offset and QA rules are sensor-specific. Compared
rasters must share a CRS, transform, extent, resolution, nodata policy, and AOI
mask. Valid-pixel coverage is calculated relative to the approved polygon.

Unsupervised K-Means/ISODATA classifications are exploratory. Supervised Random
Forest, XGBoost, or SVM requires labeled training data. Classification accuracy
is reported only when independent validation labels are available; otherwise
the reports clearly state that no accuracy claim can be made.

## Maps, Reports, and Quality

Publication maps use a shared theme system and include a crisp AOI boundary,
title/subtitle, relevant legend or colorbar, scale bar, north arrow, projection,
source/date credits, locator context, and balanced map furniture. Nodata and
outside-AOI pixels remain masked.

The offline dashboard includes a draggable before/after viewer, summary cards,
map gallery, scene table, statistics, provenance, limitations, and relative
download links. Missing optional products are reported gracefully.

The deterministic interpretation engine explains vegetation, water, built-up,
bare-soil, LULC, hotspot, and uncertainty signals from the actual run metadata
and statistics. It does not require an online AI service.

The `GeoWatch Quality Score` (0-100) summarizes run quality, not classification
accuracy. Its weighted components cover boundary confidence, imagery
availability, cloud/nodata coverage, sensor consistency, seasonal consistency,
processing completeness, and classification reliability. Component scores,
reasons, and warnings are exported as JSON, Markdown, and CSV.

The portfolio folder reuses available final products to create a one-page
infographic, selected maps, a short PDF brief, key-statistics CSV, GitHub README
snippet, dashboard launcher, and machine-readable metadata.

## Limitations

- Administrative boundaries from collaborative or public sources may not match
  an official legal boundary. Verify important projects against authoritative
  local data.
- Imagery availability, clouds, haze, shadows, seasonal phenology, coastal
  conditions, and sensor history can limit comparable coverage.
- Landsat 7 scenes after May 2003 contain SLC-off gaps; multi-scene compositing
  reduces but may not eliminate them.
- Cross-sensor comparisons require harmonization and should be interpreted more
  cautiously than same-mission comparisons.
- Spectral change indicates changed reflectance, not automatically a known
  cause. Field checks or high-resolution reference imagery are recommended.
- Unsupervised LULC class labels are exploratory and may confuse spectrally
  similar surfaces.
- Large AOIs can require substantial memory, disk, processing time, and network
  bandwidth. Start with a small boundary when learning the workflow.
- Live provider APIs, authentication methods, and catalogs can change outside
  the application's release cycle.

## Troubleshooting

### `rasterio` or `osgeo.gdal` is missing

Confirm that the command is running inside the project-local environment:

```powershell
& $mm --root-prefix $root run -n geowatch python --version
& $mm --root-prefix $root run -n geowatch geowatch doctor --strict
```

Do not use a system Python 3.14 installation for the production workflow. If
the environment is damaged, recreate it:

```powershell
.\setup-micromamba.ps1 -Recreate
```

### No common imagery policy is available

Try a smaller AOI, a wider but identical seasonal window, a higher cloud limit,
or Landsat instead of Sentinel-2. GeoWatch intentionally stops when it cannot
find a scientifically consistent policy for every requested year.

### The wrong boundary is proposed

Reject it in the wizard and provide a trusted local boundary file. Similar city,
district, division, metropolitan, and province names often represent very
different areas.

### A run was interrupted

Use `geowatch status` first, then `geowatch resume`. Verified downloads and
completed stages are reused when possible.

### Development checks fail because coverage is below 80%

Run the complete test suite. Small targeted test selections can pass their own
tests while still failing the repository-wide coverage threshold.

## Development

Install the development environment, then run:

```powershell
& $mm --root-prefix $root run -n geowatch ruff check src tests scripts
& $mm --root-prefix $root run -n geowatch mypy src
& $mm --root-prefix $root run -n geowatch pytest
& $mm --root-prefix $root run -n geowatch python scripts\verify_imports.py
```

Regenerate the JSON configuration schema after model changes:

```powershell
& $mm --root-prefix $root run -n geowatch python scripts\regenerate_docs.py
```

Generated imagery, outputs, logs, local environments, credentials, and caches
must remain untracked. Review `git status` before committing. Never commit API
tokens, passwords, downloaded satellite scenes, or client AOIs without explicit
permission.

## License and Credits

GeoWatch is released under the MIT License. See [LICENSE](LICENSE).

Application and publication credit: **GeoWatch Project**. Satellite data and
boundary products remain subject to their respective provider licenses and
attribution requirements, which are recorded in each run's provenance files.
