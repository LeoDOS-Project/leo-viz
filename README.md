# LEO Constellation Visualizer

A satellite constellation visualization tool built with Rust and egui.

## Features

- Walker Delta and Star constellation configurations
- Real-time TLE satellite tracking (Starlink, OneWeb, Iridium, GPS, etc.)
- Multi-planet support (Earth, Mars, Moon, and other solar system bodies)
- 3D globe view with GPU-accelerated rendering
- Torus topology view for constellation analysis
- Ground track visualization
- Inter-satellite link routing visualization
- Coverage analysis

## Satellite view

Choose **Display > View > Satellite**, or **Add satellite view** in the tab menu,
to inspect original NASA models of Sentinel-6, Aqua, Landsat 8, ICESat-2, SMAP,
the MiRaTa CubeSat, GOES-R series, SWOT, GRACE-FO, OCO-2, Aura, GPM Core Observatory
and Suomi NPP. Models load on demand. The individual view has an
orbit/pan/zoom camera, original model materials and textures, selectable source
components, an exploded layout and selection isolation. Material primitives are
grouped into hand-reviewed assemblies, with source mesh names available under
**Source meshes**. Hovering geometry or a component-list entry highlights the assembly without selecting it.
A label beside the cursor identifies the part a click will select. Clicking keeps
that assembly blue until another selectable assembly is hovered. Hover takes
priority over selection, so only one assembly is highlighted at a time. The
isolated part preview retains the original materials, and selection does not animate.
A fixed panel above the
parts list shows the hovered or selected mesh in isolation, with its original
materials and component description. Two-finger scrolling orbits the upright
model; pinching zooms without also rotating it.
Descriptions explain what each identified assembly does. Review caveats and raw
material information are inside **Source meshes**. Aqua's mixed material layers
are separated into whole connected surfaces for educational selections, including
the small communications dish, CERES heads, sensor housings, sounding heads and
fittings. Visually inferred assignments are labelled; remaining unknown hardware
is not given invented identities. The original triangles, winding and materials
are retained without changing the downloaded GLB files.
SMAP's reflector uses a simplified shaded gold surface and perimeter truss instead
of the original asset's dense triangular web. This is a display-only change;
the source geometry, textures and physical scale are retained.
The additional NASA illustrations retain their original materials. Aura includes
the retired HIRDLS and TES instruments, labelled as such. Where source geometry
mixes several instruments by material, the descriptions state the limitation
rather than inventing component boundaries. Suomi NPP's approximate ruler scale
uses its 4.2 m body height, not its much longer deployed solar wing.
Terra remains table-only: NASA's current GLB and FBX package omit two referenced
texture files. PACE is a candidate for a later educational model; NASA's
[integration paper](https://ntrs.nasa.gov/api/citations/20190032036/downloads/20190032036.pdf#page=4)
contains dimensioned drawings and labelled assemblies, but depicts an earlier
design and must be checked against the flight configuration.
The details panel shows a small preview of the whole satellite and published
mission and design statistics when no component is being inspected. These are
reference figures, not live spacecraft telemetry. Hovering or selecting a
component replaces them with its preview and description, without
moving the component list. Click the model background to clear a selection.
Hover **[?]** beside the satellite name for a mission description. Statistics stay
on one line; long values are truncated with the full text available on hover.
The satellite name is also shown in the view heading.
Auto-rotation starts enabled; **Space** pauses or resumes it in the active satellite
view. **Isolate selection** can stay enabled without a selection: the whole model
is shown until a component is selected. Clicking the model background clears the
selection without disabling isolation.
Switching spacecraft preserves the physical ruler scale and camera orientation.
The previous spacecraft stays visible while the new model loads, and previously
loaded models are reused within the tab. **Reset camera** fits the selected model.
Published size references distinguish deployed, stowed and bus dimensions.
**Scale** toggles full-width and full-height rulers labelled **X (m)** and **Y (m)** and a light reference
grid at the camera's target plane. Rulers use the stated size reference: Landsat 8
is calibrated to its reviewed bus mesh, rather than inventing a deployed envelope.
GOES-R uses its magnetometer boom as the size reference; SWOT uses the complete
solar-array span. Splitting an original mesh for selection retains its full scale reference.
These illustration scales are approximate, not CAD measurements; geometry is not
stretched to force every dimension to match.
The inspector is not yet connected
to simulated spacecraft resources or orbital attitude.

Choose **Display > View > Satellite Table**, or **Add satellite table**
in the tab menu, to compare the catalogue in a table with original model thumbnails,
launch dates, mass, orbit, power, payload, dimensions and source links. Mass and
power qualifiers are retained rather than treating bus mass as launch mass or
average demand as solar generation. Thumbnails fit each model individually; their
apparent sizes are not a physical comparison. Click a satellite name or thumbnail
to open its inspector. Only visible rows allocate GPU thumbnail targets.
Columns share the available width, with wrapped text and row heights sized to
their contents. Narrower views group related fields without reducing the text
size or requiring horizontal scrolling.
Click a column header to sort; click it again to reverse the direction. The sort
selector provides all fields even when columns are grouped in a narrow view.
Dates sort chronologically and numeric fields use consistent units. Missing values
remain last in either direction. Mass and power sort by the first published value,
without changing their qualifiers; dimensions sort by the largest published span.

The catalogue is independent of the model library. It bundles an operational-only
snapshot from [WMO OSCAR/Space](https://space.oscar.wmo.int/satellites), dated in the
view, including spacecraft without a 3D model. Search by name, agency, orbit or
instrument. Missing specifications remain unspecified; dry mass, launch mass and
published power figures retain their qualifiers. Active and degraded payloads are
distinguished from inactive instruments. Operator retirement notices override
known stale database flags. Hosted ISS instruments are not separate spacecraft
entries. Some source records represent operating constellations rather than
individual spacecraft. This is a sourced snapshot, not live health telemetry.

The **Satellite** view contains only spacecraft with
downloaded models. Each pane has a satellite dropdown and a **+** button to add
another pane, following the planet-view layout. Comparison panes share a physical
scale and camera: rotating, panning or zooming in any pane updates all models
together. Auto-rotation is also shared. Each pane shows the main model and
satellite statistics, without miniature previews or a component list. The
individual view retains component controls and previews. The view starts with
one pane; use **+** to compare additional satellites.
The arrow buttons beside each dropdown switch to the previous or next satellite
in that dropdown's order, wrapping at either end without changing other panes.
Narrow windows stack the panes. Satellites without models remain in the table.
Opening a model from the table shows **Back** at the top left of its rendered
grid, returning to the same table search, sort and scroll position.

Refresh the bundled metadata using Rust (requires network access):
`cargo test satellite_catalogue::tests::refresh_operational_catalogue -- --ignored --nocapture`.
This regenerates `assets/satellite-catalogue.json`; ordinary native and browser
builds use the bundled snapshot without contacting WMO. The HTML parser is only
a dependency of the native maintenance tests, not the application.

The original GLB files are retained unchanged in Git LFS. Fetch them with
`git lfs pull --include="assets/satellite-*.glb"`. Browser builds fetch the selected
asset on demand. Credits and source links are in the inspector and
[`assets/satellite-models-LICENSE.txt`](assets/satellite-models-LICENSE.txt).

## Infrastructure experiments

The modular [`leo-sim`](crates/leo-sim/README.md) crate runs deterministic orbital,
compute, storage, ISL, downlink and optional energy experiments without the GUI.
Click **Sim** in the visualizer for the same engine's controls, live resource
snapshot and JSON reports. See the crate documentation for batch runs, assumptions
and the boundary between infrastructure actions and the future job layer.

## Spatial workloads

The Rust AOI generator creates reproducible, overlapping AOIs as rectangles
or environmental patches and corridors, using weighted geographic data or uniform sampling.
At the top of the right **Earth Observation** sidebar, open
**AOIs & jobs · presets → Default AOIs (bundled) → Load**. This loads 53,000 AOIs:
50,000 bundled GDP rectangles plus 1,000 each of forests, rivers and coasts.
All four layers load **saved polygon coordinates**, without regenerating AOIs or
loading geographic source datasets. The environmental samples were frozen from
the generator's default settings and seed 42, with the 100-edge-per-AOI cap.
Native loading prepares geometry on the bounded CPU worker pool; loading the
same preset again reuses any layers still in memory, including their display
buffers. This cache holds weak references, so clearing/replacing the layers can
release their memory. Loading runs in the background, reports each layer and
preparation stage, and replaces all four layers together only after success;
an error preserves the previous layers. Job settings are assigned separately.
GDP generation defaults are an
expected mean area of 100 km², a 1,000× max/min area ratio, seed 42, 600 hotspots,
0.4 clustered fraction and 100 km hotspot spread. Rectangles have random bearings
and log-uniform aspect ratios up to 3:1, defined in a local equal-area projection
to avoid the old grid's north–south stretching at high latitudes. The frozen
sample averages 99.4 km² and lives in
[`assets/aois/default-jobs-oriented.json.gz`](assets/aois/default-jobs-oriented.json.gz).
The adjacent manifest records provenance, library versions, statistics and hashes.
The preset menu also retains **Legacy rectangles (1,000 km²)** for existing saved
workloads. Geometry changes require assigning a new set of jobs.

The frozen environmental files are `assets/aois/default-{forest,river,coast}.json.gz`.
`assets/aois/default-environment.manifest.json` records their generator settings,
source and output hashes, geometry identities and complexity. To deliberately
rebuild these assets after changing defaults (requires the geographic snapshots):

```sh
cargo test --release rebuild_environment_presets -- --ignored --nocapture
```

This command checks that saved coordinates and workload identities exactly match
the generated sample. Ordinary **Load** never runs this command. For load
timings, use `cargo test --release benchmark_preset_loading -- --ignored --nocapture`.
An optimized native run with eight workers measured about 6.5 seconds for the
previous GDP-load-plus-environment-generation path and 1.22 seconds for a cold
load of all 53,000 saved AOIs. Reusing the loaded layers took under 1 ms (worker
loading time, excluding UI frame scheduling and first GPU upload).

For new samples, tune the parameters and click **Generate**. Generation runs in
Rust on desktop and in the browser, with no Python installation or subprocess.
The 2024 GDP demand field is distributed with Git LFS; **Uniform globe** requires no dataset.
Custom prepared datasets can be loaded from a local file on desktop, a URL, or
by dropping a `.aoi-data.json.gz` file onto the selected AOI source page.
The optional [dataset exporter](tools/leo-aoi/README.md) is also written in Rust.
Job parameters are assigned in the **Jobs** pane described below; acquisition scheduling remains a later stage.

The AOI panel also has independent **Forests**, **Rivers** and **Coasts** pages,
using compact ESA WorldCover, RiverATLAS and Natural Earth snapshots. They generate
connected forest patches with varied elongation, lobes and retained clearings, or buffered paths, with source-specific controls and
independent overlays. Fetch the prepared snapshots for desktop or local web builds:

```sh
git lfs pull --include="assets/aois/*.aoi-data.json.gz"
```

New forest, river and coast AOIs have at most **100 stored boundary edges each**,
counting all holes and separate pieces together. Simplification runs before
triangulation and render preparation; capture simulation and saved geometry use
the same reduced footprint. Smallest holes or pieces are discarded when needed
to meet the cap, and the area is recalculated. The panel reports boundary counts,
removed features and the largest area change. For existing layers, use
**Simplify loaded AOIs to 100 edges** on each environmental source page, then
**Assign jobs to AOIs** again and **Save** to retain the result. Loading a file
preserves its original geometry until you explicitly simplify it.

Desktop AOI generation and simplification use a worker pool sized automatically
from the available logical CPUs, leaving two CPUs free and using at most eight
workers by default (limited to the item count, falling back to one).
Seeded locations and shape parameters are sampled in their original order;
geometry, simplification, triangulation and outline preparation run in parallel.
Results are assembled in input order, so worker count does not change geometry,
IDs or saved workloads. Progress reports the stage, count and number of workers.
The browser yields between short batches on its main thread.
Generated layers accumulate in memory without per-AOI disk writes. **Save** streams the completed layers as
JSON (optionally gzip-compressed) through a buffered temporary file and atomically
replaces the destination, without building a second full JSON copy in memory.

The GDP preset is unchanged: its ground rectangles have sampled curved sides and
dateline splits, so they can use more than four stored edges. GPU geometry counts
also include outline detail levels, spherical subdivision and fill triangles;
the stored-edge cap is not a GPU vertex cap.

In the browser, each source is fetched only when generating AOIs and cached for
the session, rather than embedded in WebAssembly. The four snapshots total about
26 MiB compressed. WorldCover uses a coarse 0.05-degree tree-cover proxy; sensor
GSD is independent. See the [dataset notes](assets/aois/DATASETS.md) and
[Rust workflow](tools/leo-aoi/README.md).

Generated AOIs use persistent GPU buffers by default. Camera transforms, horizon
clipping, outline detail selection, and distant-dot rendering run in shaders;
moving the globe does not rebuild AOI meshes on the CPU. Globe polygon fills also
use persistent GPU triangles, preserve holes, and keep the independent **Fill opacity**
control (10% by default). Buffers are shared
between globe views, with independent camera uniforms per view. Exact picking
and the selection highlight still use the original geographic boundaries.

Rendering uses the GPU whenever a GPU context is available. Display
approximations do not alter stored AOIs. To run the native GPU pixel tests and
the release benchmark with the bundled 50,000 AOIs:

```sh
cargo test --release aoi::gpu::tests -- --ignored --nocapture
```



## Job configurations and steady workload totals

After loading or generating AOIs, open **Earth Observation → Jobs → Job generator**
and click **Assign jobs to AOIs**. All loaded GDP, forest, river and coast AOIs
receive one independent job, including hidden overlays. Geography does not select
an application type. Jobs are independent of the circular routing-demo AOIs.

Desktop job assignment uses the same CPU budget and runs off the UI
thread, with a progress meter. The shared headless generator uses the same pool.
Assignments remain identical for the same seed and AOIs regardless of worker count.
The previous workload stays available until all new assignments succeed; a result
is discarded if its inputs changed while it was running. Browser assignment uses
cooperative batches without shared-memory threads.

The generator has a seed, relative weights for **Map**, **Map → Window**, and
**Window**, joint cadence/window profiles, global cadence and history-span bounds,
and output-size distributions. Initial synthetic profiles are:

| Cadence | Samples in a full window | Oldest-to-newest span |
| --- | --- | --- |
| 2 hours | 13 | 1 day |
| 1 day | 31 | 30 days |
| 1 week | 53 | 364 days |

Map jobs use the chosen cadence without a window. Each window advances one
sample per update. Default output densities have medians 8,192 B/km²/update for
mappers and 2,048 B/km²/update for windows, with a log-uniform 100× maximum/minimum
range. These are editable experimental choices, not measurements of applications.
Assignment is deterministic from seed and AOI identity/geometry, independent of
iteration order. Changing generator settings requires **Reassign all jobs**;
individual inspector edits apply directly when valid.

Click an AOI on the **3D Earth globe** to see its cadence, pipeline, window,
byte profiles and estimates. A yellow outline marks it; overlapping AOIs offer
a selection list. Picking uses the spatial index and exact polygon interiors,
including holes and dateline-separated parts. Distant dots have a small click
radius. The observation sampler takes precedence when armed. Existing place
labels take precedence; otherwise AOIs take precedence over satellite/ISL clicks
at the same screen location. Disable the corresponding AOI overlay to click
through it.

The summary reports **fresh input/day**, **retained window history**, and
**final output/day**, with mean input/output bytes per second and a pipeline
breakdown. Input bytes use the existing **Data** pane's shared capture profile:

```
image bytes = area_km² × 1e6 / GSD_m² × channels × bits/sample / 8 / compression
input/day = image bytes × 24 / cadence_hours
history = k × history-entry bytes
output/day = area_km² × output_bytes_per_km² × 24 / cadence_hours
```

Map has no history; Map → Window retains mapper outputs; Window retains imagery
in the shared image encoding. Image compression does not change the separately
configured record-output densities. Histories are saturated and reused, not
retransmitted or allocated afresh for every overlapping window. Temporary working
memory, mosaic assembly buffers and queued data are excluded from retained state.

Every job is assumed fulfilled without failures, warm-up, time-of-day skew or
resource limits. Overlapping AOIs count separately. Outputs describe the payload
offered for Earth delivery, not measured network traffic. No ISL, downlink,
sharing or execution placement is simulated. The input total is useful cropped
imagery, distinct from the Data pane's continuous full-swath sensor production.
All sizes are linear in area, so these aggregate estimates do not require a
choice of observation-unit subdivision or simultaneous whole-AOI acquisitions.

**Earth Observation → AOIs & jobs · presets** has one **Save / Load** pair above
**AOI Generator**. **Save** writes every loaded GDP, forest, river and coast AOI,
including hidden layers, plus all currently assigned jobs and their shared Data
settings. With no jobs assigned, it writes AOIs only. Jobs that no longer match
the loaded AOIs must be reassigned before saving; the previous file is preserved
if validation fails. Saving is disabled while assignment is running.

The default file is `assets/presets/default-preset.json.gz`. Saving selects
**Saved preset**, and **Load** restores that file's geometry and assignments
together. On startup, the saved preset is selected if the default file exists.
Selecting **Default AOIs (bundled)** or the legacy sample loads only their AOIs.
Loading any AOI-only file/sample clears existing assignments. Shared Data
settings are restored when jobs are included; AOI-only files leave them as-is.

Use `.json.gz` for compressed JSON or `.json` for plain JSON. Geometry includes
original coordinates, holes, dateline parts, IDs, sources, anchors, areas,
source seeds and path metadata. The outer document retains AOI schema version 1
for headless geometry readers, adds `preset_version: 1`, and optionally embeds a
versioned `workload` containing actual jobs, generator parameters and image settings.
Older AOI files and workload-only files also use the same **Load** button;
workload-only imports require matching AOIs already loaded.

Saving and loading run in the background with bounded geometry preparation.
Saves replace the destination atomically. Loads validate geometry and every job
reference before replacing the current setup; failure preserves AOIs, jobs and
Data settings. Loading needs neither Python nor geographic source datasets.
Capture progress and constellation state remain in simulation checkpoints.

The **Satisfiability** section checks the current visible constellation against
all assigned/loaded jobs using **Observation reach**. Click **Check current jobs**
to forecast from the current simulation time. The default horizon is 30 days;
**Fit job cadences (up to 30 days)** covers the longest requested cadence where
possible. Jobs with a cadence longer than the forecast remain **untested**.
Playback does not move an existing forecast.

Every AOI polygon must have full usable coverage, and the longest silence at
any of its locations must be **no greater than its required cadence**. The check
tracks the most recent access interval separately for each observed polygon
fragment; it does not substitute an AOI centre or average observation count.
Unlike fixed cadence windows, this catches observations near the beginning of
one window and the end of the next that leave an excessive gap between them.
Holes, separate pieces, overlaps and each job's solar-time/daylight requirements
are respected. Phase, fulfilment mode, storage capacity and sensor byte settings
belong to scheduling/execution and do not affect this geometric verdict.

Results show satisfied, incomplete-coverage, too-slow, resolution-limited and
untested jobs, grouped by cadence. Job details show coverage percentage,
required cadence and longest-silence bounds. Forecast start/end gaps count.
A crop's acquisition time lies within its sampled track segment; results near
the cadence threshold remain inconclusive when that timing uncertainty can
change the verdict. Halve the track step and polygon edge spacing under
**Accuracy** to check convergence. Coverage uses the existing relative area
tolerance of one part per million.

This is a **finite-forecast geometric assessment**, not an all-time guarantee
or a pointing/collection schedule. No data processing, storage allocation,
downlink, clouds, energy or compute is simulated. The native worker shares CPU,
memory and duty limits and supports cancellation. It retains a spatial partition
of last-access intervals, not a crop history or capture ledger. Job/AOI/orbit/reach
changes cancel an active check and mark an old result outdated; changing the
imaging width alone (within the same reach) or Data settings does not.

Assigned jobs, including individual edits, are stored in the combined preset
above. Loading checks IDs, geometry hashes, areas and anchors, and restores
shared Data settings without reassigning jobs. Adding/replacing/removing AOI
layers hides stale totals until jobs are reassigned or a matching preset is
loaded. Visibility alone has no effect.

The implementation separates pure generation/accounting (`src/jobs/model.rs`),
combined persistence (`src/aoi/storage.rs`, using `leo_capture::storage` for jobs), and UI (`src/jobs/mod.rs`). Totals
are cached until jobs or capture settings change; they are not recomputed each
animation frame.


## Acquisition opportunities and capture baseline

After assigning jobs, open **Earth Observation → Capture simulation** and click
**Start new capture run from setup**. The run freezes the selected Earth's visible
constellations, current playback time, AOIs, jobs and Data settings. Keplerian, J2
and TLE/SGP4 are supported; unsupported propagators fail explicitly. Setup edits
apply to the next run, so they never invalidate the active session.

The optional **Open capture diagnostic map** uses the session's own clock and saved constellation, independent
of the setup preview. It shows AOIs, current satellite ground positions, and
retained crop outlines from the last 10 simulated minutes (latest 2,000 records,
up to 100,000 vertices). Zoom to inspect AOI boundaries (up to 50,000 vertices
per frame, with spatial culling). The complete capture ledger remains in the session even
when its footprints are no longer displayed. Starting or loading a run leaves the
current globe view intact. That globe remains the setup/orbit preview and does not
follow capture time. In the diagnostic map, blue points are AOIs, amber points are
satellite ground positions, and green outlines are recently retained crops—not
cumulative coverage or the full sensor swath. Closing it returns to the globe.

Preparation reports named stages (read/decompress, validate inputs, build AOI
geometry, build spatial index, initialize coverage, validate restored state and
positions), with AOI counts and elapsed time. Its percentage describes only the
current geometry stage; complex environmental boundaries take more work than
rectangular AOIs.

- **Run/Pause** and a speed control advance capture time. A new run starts running;
  **Load checkpoint** opens paused. Run horizon defaults to one day (up to 30 days).
- Live meters show total retained bytes, reuse savings, and constellation/per-satellite
  bytes captured in the trailing **24 simulated hours**. Storage bars compare
  cumulative occupancy with each satellite's configured capacity. Each crop is
  credited at its end time; crop-record counts are strip/atlas pieces, not passes.
- **Unfulfilled acquisition updates** counts cadence deadlines without full AOI coverage,
  including partial coverage. Completed, partial and zero-coverage counts and the
  number of distinct affected jobs are separate. Partial independent crops remain
  usable. These are acquisition shortfalls, not worker crashes, processing failures or job
  drops. All cadence slots count, even those outside the allowed local solar-time
  window. Segment-level daylight/time rejection totals are shown separately;
  individual failed-deadline causes are not yet classified.
- **Baseline coverage by job** separates jobs meeting every elapsed deadline,
  jobs with a shortfall, and jobs with no deadline yet. A one-day run cannot test
  weekly or monthly deadlines. These results assess full AOI coverage at cadence,
  even when independent partial crops are usable; they do not certify other dates.
- **Fast-forward** advances a specified number of hours using the same capture
  calculations, while suspending scene rendering and preview physics. Progress and
  controls remain available. It restores the prior Run/Pause state afterwards;
  targets round up to a track-step boundary and stop at the run horizon.
- **Save checkpoint** writes a versioned gzip JSON file atomically at a committed
  step. It embeds orbit elements/SGP4 constants, epoch, current positions, all AOI
  polygons, jobs, sensor settings, counters, cropped footprints and shared-job
  associations, and partial mosaics. Loading requires no AOI files or TLE download
  and continues from the saved tick without replay. Invalid loads preserve the
  previous session. Save messages identify the simulation timestamp actually saved;
  later progress is not automatically saved. Worker errors include the operation
  and failure text, preserve the last completed display, and disable run/save until
  a session is loaded or restarted. No image pixels are generated or stored.

**Native resource limits** in the capture panel configure a shared background-worker
budget, simulation calculation/rest ratio (75% active by default), and client-memory
pause threshold. The worker budget defaults to `min(8, max(1, CPUs - 2))` and is
shared across AOI generation, job assignment, capture preparation and capture
intersection workers. On macOS these calculation threads use Utility QoS so
interactive work takes precedence. Settings apply for the current app session;
CPU budget changes take effect as admitted work batches finish.

The memory pause threshold defaults to 16 GiB and can be adjusted in the panel.
The monitor measures the whole process (including compressed memory on macOS;
RSS plus swap on Linux). Other platforms explicitly report when monitoring is
unavailable. A run reaching the threshold pauses at a completed tick, remains
saveable, and can resume after the limit is raised or memory is freed. Preparation,
checkpoint decoding and validation can cancel before retaining more state. This
is a **soft pause threshold**, not an OS allocation cap: one in-flight operation
can overshoot it, and GPU/other applications' memory is not separately controlled.
No captured data is deleted to meet the threshold. **Unload session** explicitly
stops capture and releases the session; save first because unsaved progress is
lost. This lets you release an old session before loading another large checkpoint.
Simulated satellite storage
capacity is a separate quantity and remains diagnostic in this unconstrained baseline.

Preparation clips independent AOIs in bounded parallel batches. Capture intersection
work is parallel for large workloads, then updates commit in the original
satellite/patch/crop order so reuse and checkpoints stay deterministic. Dense atlas
cells use an additional spatial index. UI snapshots copy only the displayed jobs,
share immutable inputs, incrementally maintain 24-hour capture totals and replace
stale snapshots in a single-slot mailbox. Paused sessions do not rebuild unchanged
views, and a closed diagnostic map generates no satellite/footprint display geometry.
The complete retained ledger still grows with the run; it is needed for checkpoints.

For a repeatable 50,000-AOI / 169-satellite performance check:

```sh
cargo test --manifest-path crates/leo-capture/Cargo.toml benchmark_capture_pipeline -- --ignored --nocapture
```

The job generator and individual inspector now include acquisition requirements:

- **Independent crops** become ready incrementally; **complete mosaics** require
  cumulative coverage of the entire AOI within one cadence interval. Default mix:
  50% each, independent of Map/Window pipeline and geography.
- Each interval starts at `run start + phase + n × cadence`; the default phase is
  zero. Captures do not carry over between intervals. Incomplete final intervals
  are reported separately from due updates. This is not a maximum-revisit promise.
- Default cadence profiles explicitly distinguish **every 2 hours**, **daily**,
  **weekly**, and **monthly (fixed 30 days)**. Subdaily jobs use **any-time** capture:
  both solar-time and elevation filters are disabled, assuming these jobs can use
  night acquisitions. Daily and slower generated jobs require daylight.
- Daylight **local solar time** follows a noon-centred normal (default σ = 4 h),
  truncated to ±min(tolerance, 6 h) around noon. Thus the default ±1 h tolerance
  draws preferences between 11:00 and 13:00, with noon in every acceptance window.
  This replaces the contradictory midnight/daylight draws. The minimum solar
  elevation defaults to 0°. Time is evaluated at the AOI anchor; illumination is
  checked across cropped geometry. Polar night, higher minimum elevations and
  orbital coverage can still prevent acquisition. Clouds are not modeled.
- **Regenerate capture timing** repairs an existing workload using the current
  generator seed and solar settings, preserving AOIs, cadences, phases, priorities,
  fulfilment modes, pipelines and byte profiles. Save the workload and start a new
  capture run to use these changes. Loading an old workload/checkpoint preserves
  its original rules; new GUI/CLI runs reject daylight cadences below 24 h and
  zero-width daylight windows. Historical checkpoint replay remains available.
- This is a timing-consistency check, not a constellation coverage certificate.
  The unconstrained baseline tests actual coverage over the chosen dates. A job
  that fails there needs its requirements or constellation changed before a
  resource-constrained scheduler could meet all its deadlines.
- Integer priorities default to uniform 1–5; larger values win. The specified
  future overload policy drops whole jobs strictly from lowest priority, with job
  identity breaking ties. This unconstrained baseline does **not** drop jobs.

Only newly useful cropped imagery is retained. The same physical crop can satisfy
multiple overlapping jobs when their timing constraints permit it. The report
separates physical bytes, logical bytes credited to jobs, bytes saved by reuse,
and physical bytes serving multiple jobs. Repeated passes do not inflate a job's
coverage within its current interval. All retained crops remain until the run
ends; no processing, deletion, downlink or historical-window execution is implied.

Storage defaults to **32 decimal TB per satellite**. The baseline measures retained
bytes and when each satellite would exceed that capacity, without enforcing it.
Sensor write throughput, energy and compute capacity are not constraints. The
existing infrastructure simulation remains a separate experiment.

Coverage uses polygon unions and differences, preserving holes, partial overlap
and dateline-split geometry. Local equal-area charts partition computation rather
than replacing areas with point samples. Default track segments are 5 seconds and
polygon-edge spacing is 1 km; mosaics allow 1 ppm area tolerance for clipping
roundoff. This is a numerical baseline: halve segment duration and edge spacing
to check convergence. Solar/time boundaries can conservatively lose sub-step
slivers. Geometric spherical areas, reported alongside source areas, determine
captured byte volumes.

Version-2 workload files preserve acquisition fields; loading a version-1 workload
assigns them deterministically from the saved seed, preserving its original
pipelines, cadences and byte profiles. Workload files describe demand; checkpoints
preserve a running capture session. Steady workload totals remain estimates,
separate from measured capture fulfilment.

The renderer-independent [`leo-capture`](crates/leo-capture/README.md) crate also
provides a headless CLI and optional streaming JSONL acquisition opportunities for
the later scheduler. No execution engine or constrained scheduler is introduced.

## WorldView orbit presets

The Earth constellation **Preset** dropdown includes **WorldView-3** (one
satellite), **WorldView-30**, and **WorldView-60**. The latter two are hypothetical
fleets using WorldView-3's nominal **617 km altitude and 97.9° retrograde
inclination**, propagated with secular J2. The presets replace orbital overrides,
clear stale tracks, and set the view's **Observation reach** to **450 km** and
**Observation width** to **13.1 km**. Both overlays are enabled: a faint cyan band with
dashed edges shows the observable corridor, and a solid strip in each orbital
plane's color shows the imaging swath. Their visibility toggles are independent.
They preserve resource profiles. RAAN and phase are design coordinates, not a
live ephemeris or a fixed 10:30 local solar time for every plane.

| Preset | Planes × satellites | Walker Delta phasing F | Largest sampled gap across ±84° |
| --- | --- | --- | --- |
| WorldView-3 | 1 × 1 | 0 | 6.50 days |
| WorldView-30 | 5 × 6 | 3 | 5.01 hours |
| WorldView-60 | 20 × 3 | 14 | 2.37 hours |

The 450 km width represents an **idealized accessible corridor**. WorldView-3's
native camera swath is **13.1 km**; its operator quotes 4.5-day revisit at 40° N
with viewing angles within 20° of nadir. That is not a worst-case whole-band
guarantee, and dividing it by satellite count does not determine a fleet's
maximum revisit. Coverage/revisit calculations and the unconstrained capture baseline treat the
full observation reach as accessible. Data rates, the solid swath overlay and
image previews use the narrower observation width. Steering schedules,
simultaneous collection capacity, and daylight are excluded from geometric
revisit. See the [operator datasheet](https://pacgeo.com/wp-content/uploads/2025/12/Vantor_WorldView-3_25AUG2025_Datasheet_PacGeo.pdf)
and [USGS orbit characterization](https://pubs.usgs.gov/of/2021/1030/i/ofr20211030i_ver1.1.pdf).

These results cover 2,040 jittered geographic points, including ±84°, over
seven days with 10-second interpolated track segments. Every sampled point was
seen; the worst 20 points per preset were rechecked at five seconds. The earlier
three-day screen gave 4.74 hours for WorldView-30; the longer run raised that
measurement to 5.01 hours. These are measured gaps, **not all-time bounds or a
continuous geographic coverage certificate**. The single-satellite result is
also a model observation, not a claim about commercial service performance.

Use **Earth Observation → Sampled latitude forecast…** to measure the active
view. The sampler also shows rows near the geometric reach of ±84.12°, slightly
beyond this comparison's ±84° boundary; those rows can have larger gaps.
The recorded inputs and validation summary are in
[`docs/worldview-presets.json`](docs/worldview-presets.json). Full search tooling,
per-point audits, and optimization experiments remain on the
`codex/constellation-optimizer` branch.

## ADAM orbit preset

Open a constellation's basic **Preset** dropdown (beside Starlink, OneWeb, etc.)
and select **ADAM** on Earth. This replaces that constellation's orbital settings:

- 169 satellites: 13 planes × 13 satellites, Walker Delta, F = 0, inclination 70°.
- Circular Keplerian orbit with exactly 15 revolutions per sidereal day;
  altitude is computed from the shared repeat-orbit model (about 561.386 km).
- Four ISL neighbors (+Grid); custom geometry, SSO, drag and propagation state reset.
- The view's Earth Observation pushbroom width becomes 20 km. Sweep visibility
  remains independently controlled by its toggle.

The individual tracks repeat every sidereal day (23 h 56 min); the constellation
position set repeats every 1/13 of that day (110.47 min), permuting satellites.
The 20 km sensor does not retain the earlier 210 km full-band coverage
certificate: the one-plane ascending-track spacing is about 205.28 km. Repeat
symmetry alone does not prove full coverage or a 110.47-minute revisit guarantee.
The Earth Observation panel recomputes its coverage result from the actual swath
and orbits; failed sufficient conditions are labelled unverified. Illumination,
clouds and acquisition scheduling still govern usable observations. Terminal and
compute profiles remain editable.


## Pushbroom observation preview

Enable **Display > Constellations > Pushbroom sweeps**, or use **Observation model**
in the right **Earth Observation** sidebar. Each displayed satellite
leaves a fading, ground-fixed strip with a bright current cross-track scanline.
The default swath is 20 km and the trail covers the last 60 simulation seconds;
both are adjustable. **Show sensor fan** adds faint rays from the satellite to
the scanline. Pause freezes the trail; hiding both overlays clears it.

Enable **Separate observation reach** to model and display a wider accessible
corridor alongside the imaging swath. **Show observation reach** toggles its faint
cyan fill and dashed edges independently of **Pushbroom sweeps**. Both widths
remain physically to scale, share one bounded history per satellite, and use the
same recent-trail duration. The WorldView presets configure these widths and
enable both overlays automatically. Coverage, revisit, satisfiability and the unconstrained capture baseline use
observation reach even when its overlay is hidden. Data rates always use the
smaller observation width. Turning off **Separate observation reach** makes
reach equal to width.

This is a continuous nadir-imaging geometry preview on Earth globes, using the
satellite positions already shown (including TLE/numerical positions). It does
not claim scheduled observations or apply daylight, weather, resource or AOI
acceptance constraints. Animate to accumulate the first two position samples.
Trails reset on reverse time, large time jumps and position discontinuities.
The physical swath stays to scale; line strokes remain visible at small zoom.

On desktop, enable **Pushbroom images** and click a satellite to open its image
window. The image follows its recorded swept ground, clipped to **Image length**
(50 km by default) and the available trail history. The newest scanline is at
the top. Width and along-track length retain their physical proportions.
**Freeze image** holds the current image; closing the window deselects that
satellite. Up to four image windows can be open at once.

The preview is textured on the GPU every frame, independently of tile loading.
Source detail follows the window size and display density rather than a fixed
320-pixel bitmap. It has a separate 512-tile RAM cache and a persistent
`leo-viz/pushbroom-tiles` cache in the system cache directory, so moving the globe
does not evict its imagery or compete for the same download queue. It uses
cached parents while sharper tiles load, keeps overlapping atlas regions on the
GPU, and stops fetching when disabled. Requests are limited to 256 tiles per
strip, with five-second retries on failures and 24-hour caching of missing tiles.
This is an ESRI basemap preview, not a live satellite feed or a sensor-quality
product. Imagery is unavailable beyond Web Mercator's polar limit. Like the
existing ESRI globe overlay, the image windows are currently desktop-only.

The existing **Rendering → Texture → 21K** option uses
`textures/earth/Earth_Diffuse_21k.jpg`. Large textures are Git LFS assets; fetching
and checking out an existing LFS object restores its local pixels without
changing the repository's tracked pointer or uploading anything. The GPU may
reduce the texture to its supported maximum dimension.

## Earth Observation

The resizable right **Earth Observation** sidebar contains **AOI Generator**,
**Observation model** (observation width/reach, overlay visibility, coverage and a point sampler),
and **Data**. With **Display** set to **Planet**, select **Application → Earth Observation** to open it;
select **None** or close it with **−** to restore the full view width. The application
is available only when the active planet view contains exactly one Earth.
Its sections scroll independently of
the constellation settings on the left. Width remains editable when the sweep
preview is hidden. Choose an Earth view, click **Sampler**, then click its 3D
globe. A yellow marker and coordinates identify the point; the calculation runs
in the background. Dragging still rotates the globe; Escape cancels picking.

With **Display** set to **Planet**, **Rendering** contains Projection, Texture and ESRI Tiles.
**Clock** is nested under **Display** for Planet and Solar System views only;
it contains simulation speed, date, time and synchronization controls.
ESRI imagery is enabled by default in native and browser builds and applies only to
Earth; other planets and missing imagery use the selected base texture. Both builds
use the same GPU tile renderer, parent fallbacks and fading across projections.
The browser downloads at most four tiles concurrently, keeps up to 1,024 decoded
tiles in memory and uses normal HTTP caching; native builds retain their disk cache.
Browser cache retention is controlled by the browser, not an offline download guarantee.
The sun/moon button beside **?** changes the global theme.
Sidebar icons hide and reopen the settings. When collapsed, the reopen icon sits at
the top-left with space in the view header, leaving the canvas at full width.
The presentation icon hides controls. **Sim** in the Earth Observation pane opens
the separate infrastructure simulation window.
The tabs menu contains **Auto cycle tabs**, immediately above **Cycle interval**.

`leo-sim::observation` uses the shared orbital elements for visible circular
Keplerian constellations. It expresses the ground-point dot product as three
harmonics and isolates scanline-crossing events using derivative bounds and
numerical root finding (0.1 ms time tolerance), without fixed-step propagation.
The assumptions are spherical Earth, constant spin and continuous nadir imaging,
without daylight, clouds or scheduling/resource constraints.

The pane reports **ground-track repeat cycle T**, the selected **coverage region**
(full Earth or a symmetric latitude band), and a separate **coverage guarantee**.
A repeat cycle is not a coverage guarantee, and the displayed common cycle need
not be the shortest repeat of an interchangeable-satellite pattern. Coverage is
labelled guaranteed, impossible beyond latitude reach, or unverified. An unverified
result is not evidence of holes. A sufficient interval is explicitly an upper
bound; exact minimum intervals remain available for common-rate equatorial cases.

Expand **Repeat-orbit designer** to choose inclination, approximate altitude,
altitude tolerance and maximum repeat cycle. `leo-sim::repeat_ground_track`
enumerates coprime revolution/day ratios and their circular Keplerian altitudes,
ranked by proximity to the desired altitude. **Apply orbit to selected
constellation** retains its satellite/plane counts, phasing and communication
configuration. The selected altitude is applied at full floating-point precision;
manually rounding it can break repetition. J2 repeat-orbit design is not included.

**Daily polar preset → Add guaranteed baseline** creates 15 revolutions per
sidereal day at approximately 561.386 km, inclination 90°, and 134 evenly spaced
satellites in one plane for a 20 km swath. This guarantees geometric full-Earth
coverage in every sidereal-day interval (23 h 56 min), without claiming a minimum
count or shortest interval. The count adjusts to swath and repeat ratio. Other
configurations are retained; the checked **Hide other constellations when adding**
option hides them and deselects their TLE overlays to isolate the baseline.

The sufficient single-plane construction uses N >= ceil(2πR/(m w)), increased
until gcd(N,d)=1 for multi-day repeats. This produces mN distinct ascending
crossing longitudes. The shared observation model also checks arbitrary plane and
phase layouts: within each common-rate, common-inclination repeating subset, the
largest ascending-track longitude gap must be <= w/R. This proves coverage of
the central latitude band ±asin(|sin i|), including the full Earth for polar orbits.
It does not certify the extra band reached only by the swath edge. Counts alone
are insufficient: coincident tracks add no coverage. The guarantee is recomputed
from orbital elements, never assumed from the preset label. It may be supplied
by a subset even if other visible satellites do not share its repeat cycle.
Expand **Equations and assumptions** for the derivation.

Common-rate equatorial constellations also have a synodic repeat cycle,
even at altitudes without an integer-day resonance. For general points, a
ground-track repeat within 30 sidereal days (phase residual
at most 1e-9 radians) enables a cyclic maximum including the boundary gap. Otherwise
the sampler reports the largest observed gap and longest silence in a configurable
1–30 day horizon, with an explicit warning that the all-time maximum is unknown.
Results are snapshots from the playback time at selection/recalculation; changing
orbits, width or epoch clears them and cancels an outdated worker. Eccentric,
J2, numerical and TLE models are never silently treated as circular Keplerian:
use the separate sampled forecast where supported.

### Acquisition data estimates

The **Data** section configures channel count, native resolution per channel,
packed sample bit depth and compression ratio. Defaults are **4 channels, 0.3 m,
12 bits/sample and 1:1 (uncompressed)**. Settings belong to the active tab, like
swath width. All channels share that resolution; this is a hypothetical payload,
not WorldView-3's native multispectral specification.

Rates are shown in decimal **bytes**, per satellite (mean and range), per visible
constellation and across the selected Earth view, with per-hour and per-24-hour
volumes extrapolated from the current rate. The reusable `leo-sim::acquisition`
model computes `width × ground speed / GSD² × channels × bits/sample / 8`.
Ground-subpoint speed uses the existing Kepler/J2 or TLE/SGP4 propagators and
Earth rotation over a centred 0.5-second interval; summaries refresh up to four
times per wall-clock second during playback. Numerical/42 configurations report
an unsupported estimate rather than silently substituting another propagator.

These are continuous acquisition estimates across **Observation width**, the
actual sensor imaging swath. **Observation reach** is used for access and revisit
and does not multiply data volume. For WorldView, the rates therefore use
13.1 km rather than 450 km (about 34.35 times less). Estimates are independent of playback
speed and sweep-overlay visibility. Hidden constellations and debris are excluded;
overlapping captures count separately. They do not integrate an observation
schedule or apply AOI cropping, illumination, clouds, imaging duty limits or
protocol overhead. Compression divides the selected encoded byte volume, and
16 bits/sample can represent a 16-bit storage container. At a rounded ground
speed of 7 km/s, the default 20 km swath produces **9.33 GB/s per satellite**, or
**33.6 TB per imaging hour** before compression; actual orbit-based rates vary.

### Literature presets

Open **Earth Observation → Observation model → Literature presets**, choose a design, then click
**Add literature preset**. Each entry includes a source link, published orbital
parameters, objective and sensor assumptions. The new constellations remain
editable; existing ones are kept and optionally hidden.

| Preset / source | Published Walker T/P/F | Altitude / inclination | Role |
| --- | --- | --- | --- |
| [He et al. (2020), §4.2 / Table 4](https://doi.org/10.1155/2020/8813685) | 5/5/3 | 597.766 km / 90° | Ideal Walker counterpart of the optimized initial disaster-monitoring constellation; balances transfer cost and reconfiguration time. |
| [Zong & Kohani (2019), Tables 1–2](https://doi.org/10.1155/2019/4373749) | 21/3/1 + 28/4/0 | 1100 km / 80° + 750 km / 65° | Two-shell study configuration for broad-footprint global coverage. |
| [Crisp et al. (2018), Table 3](https://arxiv.org/pdf/1807.02021#page=7) | 3/3/0; 3/3/0; 3/3/1 | 700 km / 90°; 1100 km / 86°; 1500 km / 96° | Three revisit benchmarks, not claimed optima. |

These are starting configurations for comparison, not certified optimal designs
for the current pushbroom workload. J2 is selected by default; the existing
**Sampled latitude forecast** handles these orbits. Uncheck **Use J2 propagation**
for a Keplerian comparison with the analytical tools. Neither option retunes a
published altitude to our repeat condition, and published revisit/repeat values
are not adopted as guarantees. Earth's constants, the epoch and sensor geometry
must match before comparing predictions with the papers.

Only He supplies a directly applicable swath here (60 km); its optional checkbox
changes the shared swath for Earth views in the current tab. For the other entries,
the existing pushbroom width is retained, without converting circular visibility
footprints into scanlines. This imports He's initial ideal Walker layout, not its
numerical phase deviations or target-specific transfer sequence. Zong's unspecified
shell offsets use zero.

The catalog is in `src/earth_observation/literature.rs` and uses existing
`PlanetConfig`, `ConstellationConfig` and Walker propagation. Published Walker
phasing uses increasing physical RAAN. This renderer numbers planes in the opposite
direction, so imported Delta layouts use `F_editor = (P - F_paper) mod P`.
Position-set tests check the conversion against the standard rotation equations;
the source table in the pane always shows the published F.

## Observation revisit

Open **Earth Observation → Observation model → Sampled latitude forecast…** and click **Calculate revisit**.
When a common repeat cycle is known, the button is **Sample one repeat cycle…**
and initializes the forecast with that exact cycle and Earth view. This remains a
sampled regional estimate, separate from the analytical coverage guarantee.
The background forecast uses visible Earth constellations, the observation reach
(equal to imaging width unless a wider reach is configured), and the current
simulation time. Choose 0.01–30 days, longitude sample count,
and an optional specific latitude/longitude (initially Gothenburg). Results show
observations per day, successive-observation gap ranges, never-seen locations,
and the longest observation-free span including the forecast's boundary gaps.
Locations with fewer than two observations have no established revisit interval.

Latitude rows are spaced ten degrees apart with extra samples near the nominal
reach limits. Longitudes are reproducibly stratified with jitter to reduce
resonance with Walker plane spacing. Ranges are sampled extrema, not guarantees
for every location or times outside the run. Increase duration/spatial samples
and use **Half time step** to check sensitivity. Changing constellation, swath,
or sampling settings marks results stale; playback does not change the snapshot.

The calculation reuses the visualizer's Kepler/J2 elements and TLE/SGP4 data.
Numerical/42 forecasts are explicitly unsupported rather than silently replaced
with another propagator. Earth-fixed unit-vector arcs interpolate short ground
track segments (at most 20 s, shorter for narrow swaths), testing perpendicular
scanline crossings rather than checking only discrete satellite positions.
The reusable spatial index and statistics live in `leo-sim::revisit`.
This is spherical-Earth, continuous-nadir geometry: it excludes daylight,
clouds, scheduling, resource limits, maneuvers and debris, and does not estimate
complete AOI/mosaic coverage. Simultaneous events within 1 ms are coalesced.

## Location Search

Press **Cmd+F** or **Ctrl+F** to reveal a search field at the top left of each
Earth view. Enter a place or address, press Enter, and select a result to navigate
that view. Globe and flat map projections are supported; other planets do not
show location search. Escape closes the focused search field.

The first result is highlighted automatically. Press Enter again to select it.
Ctrl+J / Ctrl+K, Tab / Shift+Tab, or the down/up arrows cycle through results
in the focused search field without changing the query. The highlighted result
scrolls into view. Provider links and attribution are in the **[?]** Info window.

Selecting a result also fetches its actual mapped boundary through Nominatim's
OSM-object lookup. Polygons retain holes and separate islands and use the same
cached GPU renderer as other AOIs, in every projection. Bounding rectangles are
used only to fit the camera, never as a substitute for a mapped area. Results
mapped only as points or lines retain just their location marker. For example,
a city label may be a point even when a separately named municipality has a
mapped boundary. The selected highlight remains when the search field is closed
and is cleared when a new query is typed. Hovering previews the location and any
boundary already cached, without making geometry requests.

Searches use [Photon](https://github.com/komoot/photon), based on OpenStreetMap.
An internet connection is required, but no API key is needed. Addresses are sent
only when submitted, not while typing. Successful queries are cached in memory;
requests across views are limited to one per second. The public service has no
availability guarantee and may throttle heavy use.

Boundary lookups use public Nominatim by default, after explicit selection only.
Requests are deduplicated across views, serialized with at least 1.1 seconds
between starts, identified on desktop, and cached in memory (up to 32 results).
Large boundaries use topology-preserving simplification with a 0.00001-degree
tolerance. Loading and errors are shown in the search panel; failed requests can
be retried. A completed lookup fits the boundary unless the user has already
moved the camera while it was loading.

The [public Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/)
requires attribution, application identification, caching, and a maximum of one
request per second **across all users of the application**, not per installation.
The client limiter covers one running app only. Distributed or busier deployments
must use a shared limiting/caching proxy or another provider; the public endpoint
has no availability guarantee. It can be changed without rebuilding: set
`LEO_VIZ_NOMINATIM_URL` on desktop, supply `?nominatim_url=...` in the browser, or
edit **Boundary Service** in the **[?]** Info window.
The in-app endpoint setting is persisted; an existing setting takes precedence
over the environment variable or URL parameter. Browser endpoints must allow CORS.

## Building

```bash
cargo build --release
```

### ESRI Diagnostics

Desktop ESRI imagery is supported in every Earth projection, with detail selected
from the visible map area and display resolution. All projections share the same
tile cache, parent fallbacks, and fades. Imagery supports levels up to 23 where
coverage exists. The globe
can zoom down to a displayed altitude of 0.01 km. Missing tiles fall back to the
nearest available parent without replacing imagery with a placeholder. Missing
tile responses are cached for 24 hours, including across app restarts. Existing
tile reuse and fades also apply to the higher levels and pushbroom image windows.

Run an automatic native replay without manually moving the camera:

```bash
LEO_VIZ_TILE_DIAGNOSTICS=1 cargo run --release
```

It exercises 3000/1000/2000 km transitions, repeats them with warm caches,
and moves to a nearby location. It saves `target/esri-diagnostics.json` and
`target/esri-diagnostics-*.png`, then closes the diagnostic window automatically.
The existing disk cache is not cleared. Optional
`LEO_VIZ_TILE_DIAGNOSTICS_LAT` and `LEO_VIZ_TILE_DIAGNOSTICS_LON` override the
default location of 48.7 N, 25.5 E.

Set `LEO_VIZ_TILE_DIAGNOSTICS_PROJECTIONS=1` along with
`LEO_VIZ_TILE_DIAGNOSTICS=1` to automatically render all twelve flat projections
with Esri imagery. This writes `target/esri-projections-diagnostics.json` and
matching screenshots.

AOI overlays also render in every projection. Generated layers retain their colors,
opacity, holes, and selection; manual AOI circles retain their fills and labels.
Projected geometry is cached and drawn on the GPU, with preparation off the desktop
UI thread. Add `LEO_VIZ_TILE_DIAGNOSTICS_AOIS=1` to the projection replay to include
the bundled 50,000 AOIs and a manual circle. Results and screenshots are written
to `target/aoi-projections-diagnostics*`.

The report separates RAM reuse, disk hits, actual HTTP requests, cancellations,
queue delays, RAM/GPU evictions, GPU tile reuse, and atlas upload frames. The native
GPU cache retains up to 512 geographic tiles (128 MiB, allocated as needed),
independently of the displayed atlas, so returning to a level or nearby area does
not require uploading those tiles again. Cached parent tiles can also fill a new
detail level on the GPU while sharper tiles load. It replaces the least recently used
tiles outside the requested area when full. Atlas CPU timings include
conversion and queue submission, not GPU execution. Settle time includes the
existing 150 ms fade and a 250 ms quiet-period check. Diagnostic HTTP requests
have an eight-second timeout; each view has a fifteen-second deadline.

To exercise a moving satellite and the live image window in the same automatic
replay, also set `LEO_VIZ_PUSHBROOM_DIAGNOSTICS=1`. This writes
`target/pushbroom-diagnostics.json` and matching screenshots, including image
preview resolution, rendered frames, tile uploads, and CPU timings in the trace.
The regular GPU tests also compare
maximum-zoom panning with double-precision Mercator coordinates and check the
per-frame budget for parent-tile preparation.
Set `LEO_VIZ_TILE_DIAGNOSTICS_CLOSEUP=1` to append three real 0.5 km views with
small north/south movements to the native replay.
Set `LEO_VIZ_TILE_DIAGNOSTICS_HIGH_RESOLUTION=1` instead to exercise levels 19-23
in Ukraine, Stockholm, and Los Angeles, including missing coverage and returns
to cached views. This saves `target/esri-high-resolution-diagnostics.json` and
matching screenshots. Each step reports the source levels used for fallback.

## Ground Terminals

Choose **Display > View > Ground Terminals**, or **Add ground terminals** in the
tab menu. The sortable table contains twelve antenna/station configurations;
its checkboxes select traces in the downlink-rate chart. Published G/T, receive
bands and system rate limits are separate from the calculated useful rates and
decimal GB per pass. Each configuration links to its specification source.

The Rust calculation uses a nominal overhead circular LEO pass, without Earth
rotation, site weather, scheduling or acquisition delays. Reference G/T stays
constant. Editable assumptions cover satellite EIRP, RF bandwidth, spectral
efficiency, required Eb/N0, losses, margin, framing overhead and backhaul.
The adaptive rate is bounded by the link budget, Shannon capacity, bandwidth,
satellite, receiver and any published station ceiling. Unknown ceilings are not
verified unlimited throughput. VHR 3200 is limited to 4,050 Mbit/s for the one
channel modelled here, not its two-channel 8,100 Mbit/s aggregate. This view does
not change simulation ground stations or guarantee a real modem operating point.

Sources, checked 2026-10-30: [Safran catalogue](https://www.safran-group.com/download/media/2010),
[VISION datasheet](https://www.safran-group.com/sites/default/files/2022-02/DS-VISION-A4.pdf),
[NASA Table 11-8](https://www.nasa.gov/smallsat-institute/sst-soa/ground-data-systems-and-mission-operations/),
[TUBITAK UZAY](https://uzay.tubitak.gov.tr/en/satellite-ground-station-and-operations/ground-station/),
and [Viasat receiver datasheet](https://www.viasat.com/content/dam/us-site/government/antenna-systems/documents/High_Rate_Receiver_VHR3200_Datasheet_3_30_22.pdf).

## TODO

### Features

- Ability to see other planets around the one currently being viewed. When
zoomed out, we should be able to see the solar system.
- Clicking another planet should make that planet be focused instead.
- Camera tilt (Ctrl+drag) with perspective projection for viewing satellites from the side
- Lambert solver for orbit transfer visualization (satellite maneuvers between orbits)
- Two-impulse rendezvous planning (engine burns to move a satellite to meet another)

### Realism

- Euler-Hill relative motion for formation flying (drift modeling within tight clusters)
- WGS84 ellipsoid model for precise ground track lat/lon
- Per-satellite NRLMSISE-00 atmosphere drag (varies by position, time, solar activity) via leodos-lib42
- EGM96 spherical harmonic gravity model via leodos-lib42
- NASA 42 numerical integration propagator via leodos-lib42 (full force models: drag, solar pressure, higher-order gravity)
- Lunar gravitational perturbations on satellite orbits
- Precession/nutation frame transforms for long-term coordinate accuracy
- Lagrange point visualization for cislunar orbits

## Image Credits

Planet textures are from:

**[Solar System Scope](https://www.solarsystemscope.com/textures/)** (CC BY 4.0)
- Earth, Moon, Mars, Mercury, Venus, Jupiter, Saturn, Sun (2K, 8K)
- Stars, Stars + Milky Way (2K, 8K)

**[Solar System in 16K Texture Pack](https://www.artstation.com/marketplace/p/5oNl/solar-system-in-16k-texture-pack)** by Textures for Planets
- Earth (21K diffuse, 16K clouds, normal, specular, cities)
- Mercury (16K diffuse, normal, specular, roughness, displacement)
- Venus (16K diffuse, normal, roughness, displacement)

**[Steve Albers' Planetary Maps](https://stevealbers.net/albers/sos/sos.html)**
- Ganymede (4K, Bjorn Jonsson modified)
- Io (2K, Voyager + Galileo composite)
- Europa (4K, Juno data)
- Charon (8K)
- Enceladus (8K, Cassini)

**[Askaniy on DeviantArt](https://www.deviantart.com/askaniy)** (CC BY 3.0)
- Titan True Color Map (2K, Cassini infrared + Huygens data)
- Triton Texture Map (2K, Voyager 2 calibrated color)
- Phobos Texture Map (2K, Mars Express)

**[Oleg-Pluton on DeviantArt](https://www.deviantart.com/oleg-pluton)**
- Mimas Texture Map (2K, Cassini)

**[FarGetaNik on DeviantArt](https://www.deviantart.com/fargetanik)**
- Iapetus Truecolor Texture Map (2K, Cassini)

**[Bjorn Jonsson's Planetary Maps](https://bjj.mmedia.is/)**
- Callisto (4K)

**[Celestia Project](https://github.com/CelestiaProject/CelestiaContent)** (GPL)
- Vesta (4K, Dawn mission mosaic)

**[NASA Science Assets](https://assets.science.nasa.gov/)**
- Additional planetary imagery

## TLE Data

Real-time satellite tracking data is fetched from
[CelesTrak](https://celestrak.org/) maintained by Dr. T.S. Kelso.
- ISS, Starlink, OneWeb, Kuiper
- Iridium, Iridium NEXT, Globalstar, Orbcomm
- GPS, Galileo, GLONASS, Beidou
- Molniya, Planet
- ...
