# AOI Data in Rust

The visualizer generates GDP, forest, river and coast AOIs in Rust on desktop
and WebAssembly, without a Python environment or subprocess. Open **Earth
Observation -> AOI Generator**, select a source and generate. Each source has
independent settings and an overlay. **Default Jobs -> Load preset** loads the
unchanged saved set of 50,000 AOIs.

## Default Datasets

Compact snapshots live in `assets/aois/` and are tracked by Git LFS. Fetch them
for desktop builds or a local web deployment:

```sh
git lfs pull --include="assets/aois/*.aoi-data.json.gz"
```

The browser fetches only the selected snapshot when Generate is pressed, then
caches decoded data for the session. Uniform globe and the saved Default Jobs
preset do not fetch a source dataset. Snapshots are not embedded in the executable
or WebAssembly. Trunk copies them into the web distribution; Pages CI resolves
LFS files before building. Browser downloads come from the deployed site, not
the Git LFS API. Do not deploy LFS pointer text as a dataset.

| Source | Compressed Size | Detail |
| --- | ---: | --- |
| GDP 2024 | 1.7 MiB | 1,440 x 720 equal-area demand grid; 264,199 positive cells |
| Forests | 0.41 MiB | WorldCover 2021 tree-cover proxy sampled at 0.05 degrees |
| Rivers | 22.6 MiB | RiverATLAS v1.0; all 10 global regions; 857,210 reaches with flow >= 10 m3/s |
| Coasts | 1.2 MiB | Natural Earth land v5.1.1, 1:10 million; 6,847 shoreline segments |

Forests are area-weighted connected patches, not fire detections. Rivers follow
connected reaches downstream and buffer both banks. Coasts follow shoreline
segments, with independent inland and offshore buffers. Paths can end early at
outlets, missing connections, map edges or after one island circuit. Lines are
simplified to 0.01 degrees. Forest sampling is a coarse proxy, not a forest
fraction measurement or sensor resolution. Natural Earth coasts omit inland
lakes and do not include OSM-level detail. The compact river snapshot cannot
recover smaller streams; custom snapshots may contain lower-discharge reaches.

See [dataset notes](../../assets/aois/DATASETS.md) for attribution and processing.
Metadata and raw source hashes are retained in the snapshots. The default set
is about 26 MiB compressed; decoded network data requires more memory.

## Custom Data

The Dataset field accepts a prepared `.aoi-data.json.gz` file or URL. Clear the
field to select the default snapshot. Desktop relative custom paths resolve
from the repository root; default snapshots can also live alongside the binary
under `assets/aois/`. Browser paths resolve from the deployment's base URL.
Dropping a prepared file onto a source page works on both platforms. Cross-origin
URLs must permit CORS.

This directory contains a standalone Rust exporter for existing prepared `.npz`
caches, with no NumPy runtime. From the repository root:

```sh
cargo run --release --manifest-path tools/leo-aoi/Cargo.toml -- tools/leo-aoi/data/gdp-demand.npz target/custom-demand.aoi-data.json.gz
cargo test --manifest-path tools/leo-aoi/Cargo.toml
```

The exporter preserves metadata, validates arrays, writes little-endian network
arrays and uses a fixed gzip timestamp. It refuses to overwrite existing files.
It supports prepared demand, forest, river and coast caches; it is not a raw
GeoTIFF/COG/shapefile ingestion tool. The former Python AOI package has been
removed. Its ignored local caches and environments are left untouched. Existing
example JSON files are retained for reference, not as inputs to this exporter.

## Portable Format

Uncompressed JSON or gzip-compressed JSON is accepted, with a 512 MiB input and
decoded JSON limit. Every document has `schema_version: 1` and one source kind.

Demand uses `kind: "demand"`, `columns`, `rows` and sorted, positive
`cells: [[index, weight], ...]`. Its optional `eligible` field is a base64 mask.
Forest uses `kind: "forest"`, `columns`, `rows` and `mask`. Both masks are
row-major, LSB-first bit-packed, with zero padding in the last byte. Forest cells
are equirectangular, west to east and north to south; demand uses the existing
spherical equal-area grid.

River and coast sources use `kind: "river"` or `kind: "coast"`, plus a `network`
object with `points`, `offsets`, `next_index`, `length`, `ids` and `flow`. These
may be JSON arrays or base64 little-endian arrays: points are f32 longitude/
latitude pairs, offsets u64, downstream indices i32, lengths f64 kilometres,
IDs i64 and discharge f32 m3/s. Offsets have one extra terminal entry; -1 means
no downstream connection. Coast segments must have land on their left so that
inland/offshore buffers have the correct sign.

Generation uses a 6,371 km sphere, independent ChaCha8 random streams and local
spherical azimuthal-equidistant buffers. Identical inputs and settings are
deterministic on desktop and WebAssembly. GPU display approximations do not
change stored AOI geometry. The saved preset remains unchanged.
