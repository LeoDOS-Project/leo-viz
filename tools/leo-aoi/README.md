# AOI generation

A data pipeline for synthetic spatial workloads:

**GeoTIFF layers → normalized demand field → sampled anchors → rectangles or connected rectangular mosaics.**

AOIs have no application labels, DAGs, deadlines, observation cadence or resource
profiles. Those parameters can be attached later using rules and correlations.
The examples in the paper are motivation, not a catalogue of simulated jobs.

This Python package sits above the `leo-sim` infrastructure crate. Rasterio/GDAL
handles geospatial ingestion; NumPy handles reproducible sampling; Shapely exports
valid dissolved polygons. The output is versioned JSON for the Rust simulator or
other consumers. An AOI is not an acquisition action: `leo-sim::config::Target`
describes a circular sensing target and cannot faithfully represent these mosaics.
Converting AOI requirements to feasible pushbroom acquisitions is a later stage.

## Run with a real GDP dataset

From `tools/leo-aoi` (Python 3.11 or newer):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/fetch_gdp.py
.venv/bin/python -m leo_aoi ingest examples/gdp.json data/gdp-demand.npz
.venv/bin/python -m leo_aoi generate data/gdp-demand.npz examples/mosaics.json runs/gdp-10000 --geojson
.venv/bin/python -m unittest discover -s tests -v
```

The downloader fetches only the 9.5 MB, 30 arc-minute **total GDP** raster from
[Kummu et al., version 4](https://doi.org/10.5281/zenodo.18429133), verifying a pinned
SHA-256 checksum. Band 35 is 2024. The dataset contains GDP estimates, not measured
EO demand. The coarse file is convenient for a runnable example; substitute the
5 arc-minute or 30 arc-second total-GDP raster for finer location structure.
Check the band mapping: the 30 arc-second file uses a different sequence of years.
Dataset attribution and the paper citation are provided on the linked record.
For GDP, nothing downloads during ingestion or generation; paths are local and relative
to the ingestion configuration file. Data and generated runs are git-ignored.

A network-free uniform reference case is also available:

```sh
.venv/bin/python -m leo_aoi ingest examples/uniform.json data/uniform-demand.npz
.venv/bin/python -m leo_aoi generate data/uniform-demand.npz examples/mosaics.json runs/uniform-10000
```

Create `data/` first if no dataset has been downloaded. Outputs are never overwritten.

## Visualizer panel

In the native visualizer, open **Earth Observation → AOI Generator** in the right sidebar. Set
the **GDP rectangles** page's number of AOIs, average area (km²), size ratio (maximum/minimum), and location
source, then click **Generate**. Forests, Rivers and Coasts have independent pages
and overlays, described below. Generation runs in the background and replaces
the previous overlay on success. Closing the panel keeps the overlay; use its
visibility checkbox or Clear overlay to hide or remove it.

The arithmetic expectation of a log-uniform distribution is the requested average:
for size ratio `r > 1`, `min = mean × ln(r)/(r−1)` and `max = min × r`.
At `r = 1` every requested area equals the mean. The panel reports the actual
sample mean. Ground rectangles have independent random rotation and a maximum
elongation control; legacy grid rectangles retain tile rounding. Seed and
clustering controls are under **Clustering and reproducibility**.

Complete the Python setup and GDP ingestion above before using the default
**Demand dataset** option. **Data and runtime** accepts any ingested demand `.npz`
and a Python executable. The app prefers this package's `.venv`, or the executable
in `LEO_AOI_PYTHON`, then `python3`. **Uniform globe** needs no raster download.
The GDP page generates single rectangular AOIs, allowing overlaps, and invokes the same
Python sampling, geometry and polygon-export functions as
batch runs; it does not install packages. Environmental pages can download missing
geographic sources on first generation.

Mosaic outlines are attached to Earth in every 3D globe view and clipped at the
horizon. Subpixel AOIs use dots until zoomed in; outlines use a cached spherical
approximation with at most 0.5° between vertices. Generated AOIs are separate from
the circular routing-demo targets. Assign jobs separately in the **Jobs** pane;
acquisition actions remain a later stage.
The overlay is kept for this app session; native generation is not available in
the browser build. Interactive generation is limited to 50,000 AOIs per source.

## Environmental sources

The wider **Earth Observation → AOI Generator** panel has separate **GDP rectangles**,
**Forests**, **Rivers**, and **Coasts** pages. Each retains its own settings and overlay
for the current app session. Generating one source replaces only that source's
overlay; the **Visible overlays** checkboxes allow all four layers to be compared.
Forest outlines are green, rivers cyan, and coasts amber. Source preparation and
generation run in a worker, with progress shown in the panel.

| Source | Sampling and footprint | Main controls |
| --- | --- | --- |
| [ESA WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221) | Tree-cover cells sampled by spherical ground area; connected growth favours forest with varied elongation and coherent lobes, stopping at a sampled target area. Clearings remain as holes. | Count, seed, mean area, area ratio, elongation, outline variation, forest adherence |
| [HydroRIVERS v1.0](https://www.hydrosheds.org/products/hydrorivers) | Sample a location by eligible river length; follow `NEXT_DOWN` and buffer both banks in a local metric projection. | Count, seed, mean path length, length ratio, bank buffer, minimum discharge |
| [OpenStreetMap coastlines](https://osmdata.openstreetmap.de/data/coastlines.html) | Sample a location by eligible coastline length; join directed source chunks and follow the shore. Land is on the left of the directed coastline. | Count, seed, mean path length, length ratio, inland and offshore buffers |

These are generic spatial requirements, without application/DAG labels. Overlap
is allowed. Job counts and bounded area/length distributions determine how much
geography receives jobs; the source map is not converted wholesale into AOIs.
Area and length distributions are log-uniform, parameterized by arithmetic mean
and max/min ratio. Location and shape RNG streams are independent and seeded.
Record IDs include the source kind, so layers can be combined without collisions.

Each environmental source defaults to 1,000 AOIs, seed 42, anchor latitude ±70°,
and a 100 m projected output tile edge. Forests default to 1,000 km² mean area,
100× area range, maximum elongation 3:1, outline variation 0.7 and forest adherence
0.8. River and coastal paths default to a 100 km
mean and 20× length range. River buffers extend 2 km on each bank, with minimum
discharge 10 m³/s. Coastal buffers extend 2 km inland and 1 km offshore.
Footprints may extend outside the anchor latitude band. Counts are limited to
50,000 per source; very large footprints also have explicit geometry memory limits.

Forest growth adds only edge-adjacent cells, so each footprint remains connected.
Each AOI samples a log-uniform growth aspect ratio from 1 to **Elongation**, a
uniform direction over 0–180°, and a smooth preference field producing broad
lobes and indentations. **Outline variation** sets its strength; **Forest
adherence** makes nonforest frontier cells less attractive. At zero adherence,
only the anchor is selected from forest cover. At one, crossing clearings remains
possible to fulfil the area target: the footprint is a monitoring region, not a
claim that every enclosed pixel is forest. The final aspect ratio also depends
on the preference field and source geography.

Growth stops at the requested area with at most one source-cell overshoot, before
the existing all-touched output tiling adds boundary cells. No holes are filled.
A four-radius ellipse bounds growth; generation reports an error if that bound
or the existing cell/memory limits prevent fulfilling the request. Coarse cells
limit the variety of very small AOIs. Shape controls do not change sampled anchors
or requested areas. Each record includes the realized growth direction, elongation,
variation and coarse tree-cover area fraction. These are synthetic monitoring
footprints, not fire-spread or ecological boundary models.

Explicit old `compactness` configurations select the original growth and hole-
filling behaviour for replay. New requests omit it (or set it to null); the UI
uses only the three new controls. River/coast geometry is unchanged.

WorldCover's categorical COG overviews are read by HTTP range requests into a
global **0.025° sampling grid** (about 2.8 km north–south). Class 10 is a coarse
tree-cover proxy, not a fractional forest-cover estimate or a native-resolution
forest boundary. This avoids downloading the full 117 GB 10 m product. Narrow
forests can be lost at this scale; changing the simulated camera GSD does not
improve this geographic proxy. The 100 m output tiles describe the monitoring
footprint, not additional forest evidence. The river cache retains reaches with
discharge ≥1 m³/s; the panel can impose a higher threshold. River and coastal
source lines are simplified by 0.0005° before indexing; metric footprint buffers
are constructed using local azimuthal equidistant projections.

Rivers stop at outlets or missing/inconsistent links. Coasts stop at unresolved
junctions, gaps, or after completing one circuit of a closed coastline. Therefore
actual path lengths can be smaller than requested. Both lengths and a truncation
flag are recorded, and the panel reports the actual mean and truncated count.
A river buffer is a monitoring corridor, not a physical flood-inundation model.

After installing the Python package, first generation can prepare a missing source.
For explicit preparation from `tools/leo-aoi`:

```sh
.venv/bin/python -m pip install -e .
.venv/bin/python -m leo_aoi.environment_data all
# Or prepare just forest, river, or coast. Existing caches are reused.
```

Files live under ignored `tools/leo-aoi/data/environment/`. The global river and
coastline archives are approximately 544 MB and 925 MB; extraction
require several GB. Compressed sampling caches on this machine are approximately
2 MiB (forest), 73 MiB (river), and 54 MiB (coast). WorldCover tile reads are cached
individually and can resume. Downloads, extraction and prepared-cache publication
use temporary files. A completed cache allows network-free generation.

Prepared caches and generated viewer JSON retain source names, citation/license
information, processing parameters and SHA-256 hashes. The OSM download URL is a
changing extract: preserve the cache and its source hash for experiment replay.
WorldCover: CC BY 4.0, © ESA WorldCover project 2021 / Contains modified Copernicus
Sentinel data (2021) processed by ESA WorldCover consortium. HydroRIVERS: HydroSHEDS
license, Lehner & Grill (2013), doi:10.1002/hyp.9740. Coastlines: © OpenStreetMap
contributors, ODbL 1.0. Source attributions remain necessary when publishing data.

For a headless run, pass `{"schema_version":1,"environment":{"kind":"river",
"count":1000,"seed":42}}` on stdin to `.venv/bin/python -m leo_aoi.viewer`; stdout
is the same versioned viewer JSON and stderr reports progress. Optional fields
match `environment.Config`. Each result retains requested/actual areas, compact
tile rectangles, dissolved Polygon/MultiPolygon geometry and source-specific
sampling metadata. Narrow corridors use all-touched rasterization; actual tile
area includes this quantization. Dateline-crossing footprints remain one AOI.

## AOI rendering

An Earth-fixed bounding-sphere hierarchy rejects off-screen and hidden groups
before individual footprints are projected. Visibility bounds include complete
footprints, rather than only anchors, so large AOIs and horizon crossings survive.
Rings have precomputed levels of detail selected for a maximum 0.7-pixel
simplification tolerance. Small footprints become dots; overlapping dots share a
3-pixel screen bin. Camera changes below a conservative 0.25-pixel displacement
reuse a cached mesh. Polygon fills default to 10% opacity, independently adjustable
from outlines. They are triangulated once on the loading worker, preserve holes,
follow the globe surface and use the same visibility culling. Full geometry,
AOI counts and simulated resource demand
are unchanged by these display approximations.

The moving-camera benchmark measures the current filled render path on the
bundled 50,000 AOIs, alongside the original brute-force outline-only path. Run from the repository root:

```sh
cargo test --release --offline --locked benchmark_default_jobs_render -- --ignored --nocapture
```

Timing measures CPU mesh construction, not whole-app FPS or GPU time. Tests also
check visible-boundary retention, dateline handling, detail restoration on zoom,
source sampling, corridor connectivity and deterministic replay.


## Frozen Default Jobs preset

The visualizer's **AOI Generator → Default Jobs → Load preset** reads the bundled
`assets/aois/default-jobs-oriented.json.gz`. Loading works without Python or a
local GDP raster. The preset restores 50,000 AOIs, 100 km² expected mean area,
1,000× size ratio, seed 42, 600 hotspots, clustered fraction 0.4 and 100 km hotspot
spread. Bearings are uniform over 0–180° and aspect ratios log-uniform over 1/3–3.
Both the panel and reproducible build use `assets/aois/default-jobs-oriented.settings.json`.

Requested areas are approximately 0.691–691.467 km²; this frozen sample's actual
mean is 99.434 km². Each footprint is a rectangle in a local equal-area projection,
with sampled geographic edges and dateline splitting. This preset provides spatial
input only: use the **Jobs** pane to assign pipelines and temporal parameters.

The menu also offers **Legacy rectangles (1,000 km²)**, backed by the unchanged
`assets/aois/default-jobs.json.gz` and original settings/manifest. Load it when
using a workload saved against that geometry; job identities include geometry.

From the repository root, after the GDP setup above, reproduce the new preset:

```sh
tools/leo-aoi/.venv/bin/python -m leo_aoi.preset \
  assets/aois/default-jobs-oriented.settings.json \
  tools/leo-aoi/data/gdp-demand.npz /tmp/default-jobs-oriented.json.gz --name "Default Jobs"
```

Use a destination that does not already exist. The archive is deterministic gzip
(timestamp zero, no embedded filename) of the versioned viewer JSON. It records
resolved generation settings, demand provenance, stable AOI IDs, anchors, hotspot
IDs, areas, local width/height, rotation and Polygon/MultiPolygon geometry.
Oriented rectangles have `tile_count: null`: they have no global-grid tile
assignment. Observation-unit tiling is a separate future stage. Loading and fill
triangulation run in the background on desktop.

The adjacent `default-jobs-oriented.json.manifest.json` records source and archive
SHA-256, library versions and realized statistics. Reproduction requires the
recorded input and compatible library versions. The source is the GDP example's
Kummu et al. v4 2024 PPP GDP field, with 60% direct demand-field sampling and 40%
displaced hotspot sampling. Frozen geometry and its manifest are repository assets;
downloaded rasters and intermediate demand fields remain git-ignored.

## Ingestion and composition

`examples/mixed.json` shows how to combine additional local rasters; its population
and mask filenames are placeholders. Every raster needs a CRS. Inputs can have
different resolutions and projections; GDAL reprojects them onto a shared,
configurable equal-area demand grid. Raster bands are one-based.

| Layer kind | Input values | Resampling and interpretation |
| --- | --- | --- |
| `mass` | Total GDP, inhabitants, built-up m² **per source cell** | Weighted sum of contributing source cells |
| `density` | Nonnegative amount **per km²** | Average, then multiply by destination ground area |
| `mask` | Eligible fraction from 0 to 1 | Average, then apply `mask_threshold` |

Declared nodata contributes zero. Nonfinite or negative remaining values, missing
CRS, empty layers after masking and unknown config fields fail explicitly. Scaled
integer rasters must have their scale/offset applied before ingestion. Categorical
rasters should be converted to a numeric weight or fractional mask upstream.
GDAL reads source blocks without loading the whole source raster into Python.
The destination demand grid is capped at 20 million cells.

Each demand layer is normalized independently after masking, then combined using
its `weight`. Thus a 70/30 GDP/population mixture means 70/30 sampling mass,
independent of the layers' units. `concentration` raises the combined nonzero
field to a power and renormalizes (1 preserves it). `background_fraction` mixes
in uniform ground-area sampling over eligible demand cells. With no layers it
must be 1. These are experiment parameters, not fitted demand estimates.

The optional mask restricts **anchor locations at demand-grid resolution**. It
does not clip the generated AOI to land, buildings, or another mask. Nor does
upsampling create new spatial evidence: anchor positions within a demand cell
are sampled uniformly in ground area. Demand can be zero in places where an AOI
extends. A coverage mask over an entire irregular AOI requires another stage.

Ingestion produces `demand.npz`: normalized probabilities, eligible cells, grid,
and provenance including file hashes, bands, CRS and configuration. That artifact
can be reused with many shape and seed configurations.

## Geometry and size

The GUI and current Default Jobs use `shape: "oriented_rectangle"`. Dimensions
and rotation are defined in a Lambert azimuthal equal-area plane centred on each
sampled anchor, using the simulator's 6371 km Earth radius. This avoids systematic
north–south stretching at high latitudes. Aspect and area are sampled independently;
`random_rotation` defaults to true and uses its own seeded stream. GeoJSON edges
are sampled at no more than 5 km intervals, refined near the poles and split at
the dateline. Projection area equals the requested area; sampled boundaries
approximate it. These rectangles store vector geometry, not grid memberships.

Legacy rectangles and mosaics use spherical cylindrical equal-area coordinates:
`x = R longitude`, `y = R sin(latitude)`, with the same 6371 km Earth radius as
`leo-sim::geometry`. Columns wrap at the dateline; rows do not wrap at the poles.
Cells have equal ground area in this spherical model. Rectangles are aligned in
this projection; their physical aspect ratio varies with latitude, increasingly
near the poles. `tile_edge_km` sets the approximate projected edge length, not
sensor resolution or a promise of square ground pixels. The rounded grid dimensions
and exact resulting cell area are recorded in the manifest.

The example draws requested areas log-uniformly from 1 to 5,000,000 km², then
rounds to the nearest number of tiles. Actual area differs by at most half a tile.
Aspect ratios are also log-uniform; both ranges are configurable. These broad
synthetic distributions are intended for experiments, not commercial calibration.

The legacy preset uses `shape: "rectangle"`: one rectangle
centered on the sampled anchor, shifted away from a pole when needed. Height is
rounded from the requested area/aspect ratio, then width is rounded to match area.
No remainder strips are added. Actual area can therefore differ by more than half
a tile; both requested and actual areas are recorded. A footprint crossing the
dateline has two storage/GeoJSON pieces but remains one AOI. Overlaps between AOIs
are retained, with no spacing or separation constraint.

For backwards compatibility, batch configurations without `shape` retain the
original `"mosaic"` mode (including `examples/mosaics.json`). In that mode each AOI
is a union of vertically adjacent rectangular bands sharing an anchor
column. Varying band widths produces compact, irregular outlines. Integer remainders
are represented by additional rectangles, preserving the exact tile count. Every
mosaic is edge-connected on the globe, contains its sampled anchor, and has no
duplicate or overlapping tiles internally. Different AOIs may overlap arbitrarily.
GDP shapes are contiguous; their outlines are independent of forest, harbor or
national boundaries. Environmental sources use the separate strategies below.

A continent-sized AOI may contain hundreds of millions of small tiles, but it is
stored as a few integer rectangles. Memory and generation cost depend on the number
of bands, not tile count. `Aoi.tiles()` is a lazy iterator; `Aoi.shared_tiles(other)`
computes exact pairwise overlap on the same grid without expanding it.

## Location and overlap controls

With `hotspot_count: 0`, each anchor is independently drawn from the demand field.
Otherwise, hotspot centres are sampled once from that field. A configurable fraction
of AOIs select those centres with replacement and receive a Gaussian displacement
in kilometres, applied on the sphere; the remainder sample the field directly.
Hotspot displacements must land in an eligible demand cell. An impossible jitter
configuration errors after bounded retries. Smaller spreads and fewer hotspots
increase local clustering. No cities or regions are hard-coded.

Locations, hotspot parents, shapes and rotations use separate PCG64 streams
derived from `seed`. Changing
shape parameters does not change anchors, and increasing `count` preserves the
existing output prefix. Retain the recorded library versions for exact replay.
Overlap is geometry supplied to the later job generator, not a prescribed sharing
policy. The generator neither merges nor deduplicates different AOIs.

## Output contract

Each run writes a new directory:

- `manifest.json`: schema version, resolved generation configuration, tile grid,
  RNG/library versions, demand/source provenance and output hashes.
- `aois.jsonl`: one AOI per line, with stable ID, anchor, hotspot ID (or null),
  requested and actual area, plus rectangular tile ranges/counts for grid shapes or
  vector geometry, dimensions, bearing and `tile_count: null` for ground rectangles.
- `aois.geojson` (optional): one dissolved Polygon/MultiPolygon per AOI in lon/lat.
  Dateline-crossing AOIs are split at ±180°, not drawn across the map.
- `coverage.npz`: number of distinct AOIs intersecting each coarse demand cell.
- `summary.json`: area quantiles, total tile memberships (null for vector AOIs) and coarse coverage/overlap.

A rectangle `{col, row, width, height}` denotes half-open integer column/row
ranges, with origin at the north-west of the grid. Tile identity is the grid
definition plus `(col, row)`, consistent across every AOI in a run. Use integers
wide enough for continent-scale counts (e.g. Rust `u64`). JSON geometry is compact
and independent of future execution or state-placement decisions.

The coverage statistics count an AOI once per demand cell even when several of
its rectangles touch that cell. They are explicitly **coarse intersection
statistics**, not exact fine-tile overlap or a measure of reusable observations.
Large AOIs count their full area in `sum_aoi_area_km2`, including overlap with
other AOIs. This sum is not the area of their geographic union.

Tests cover mass conservation when resizing rasters, density/area weighting,
mixtures and masks, weighted anchor sampling, deterministic replay, topology at
the poles/dateline, valid GeoJSON, exact overlap, compact continental geometry,
and complete output generation.
