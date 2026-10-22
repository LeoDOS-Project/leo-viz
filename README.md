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

## Infrastructure experiments

The modular [`leo-sim`](crates/leo-sim/README.md) crate runs deterministic orbital,
compute, storage, ISL, downlink and optional energy experiments without the GUI.
Click **Sim** in the visualizer for the same engine's controls, live resource
snapshot and JSON reports. See the crate documentation for batch runs, assumptions
and the boundary between infrastructure actions and the future job layer.

## Spatial workloads

The Rust AOI generator creates reproducible, overlapping AOIs as rectangles
or connected rectangular mosaics, using weighted GDP demand or uniform sampling.
In the right **Earth Observation** sidebar, open **AOI Generator → Default Jobs → Load preset** to display the bundled
50,000 AOIs without Python or external data files. Generation defaults are an
expected mean area of 1,000 km², a 1,000× max/min area ratio, seed 42, 600 hotspots,
0.4 clustered fraction and 100 km hotspot spread. The saved dataset uses the
existing GDP demand field, with one rectangle per AOI (split only at the dateline),
and retains tile ranges alongside its
GeoJSON boundaries in [`assets/aois/default-jobs.json.gz`](assets/aois/default-jobs.json.gz).
The adjacent manifest records provenance, library versions, statistics and hashes.

For new samples, tune the parameters and click **Generate**. Generation runs in
Rust on desktop and in the browser, with no Python installation or subprocess.
The 2024 GDP demand field is distributed with Git LFS; **Uniform globe** requires no dataset.
Custom prepared datasets can be loaded from a local file on desktop, a URL, or
by dropping a `.aoi-data.json.gz` file onto the selected AOI source page.
The optional [dataset exporter](tools/leo-aoi/README.md) is also written in Rust.
Job parameters are assigned in the **Jobs** pane described below; acquisition scheduling remains a later stage.

The AOI panel also has independent **Forests**, **Rivers** and **Coasts** pages,
using compact ESA WorldCover, RiverATLAS and Natural Earth snapshots. They generate
bounded patches or connected buffered paths, with source-specific controls and
independent overlays. Fetch the prepared snapshots for desktop or local web builds:

```sh
git lfs pull --include="assets/aois/*.aoi-data.json.gz"
```

In the browser, each source is fetched only when generating AOIs and cached for
the session, rather than embedded in WebAssembly. The four snapshots total about
26 MiB compressed. WorldCover uses a coarse 0.05-degree tree-cover proxy; sensor
GSD is independent. See the [dataset notes](assets/aois/DATASETS.md) and
[Rust workflow](tools/leo-aoi/README.md).

Generated AOIs use persistent GPU buffers by default. Camera transforms, horizon
clipping, outline detail selection, and distant-dot rendering run in shaders;
moving the globe does not rebuild AOI meshes on the CPU. Buffers are shared
between globe views, with independent camera uniforms per view. Exact picking
and the selection highlight still use the original geographic boundaries.

Disabling **GPU rendering** uses the CPU fallback with its spatial visibility
hierarchy, zoom-dependent outlines, and grouped distant dots. Display
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

**Save / load workload** writes versioned compressed JSON containing actual
assignments, generator parameters and shared image settings. The file references
AOIs rather than duplicating geometry. Load the same AOI datasets first; loading
verifies IDs, geometry hashes, areas and anchors, and restores the shared Data
settings. Adding/replacing/removing an AOI dataset hides stale totals until jobs
are reassigned or a matching workload is loaded. Visibility alone has no effect.

The implementation separates pure generation/accounting (`src/jobs/model.rs`),
compressed persistence (`src/jobs/storage.rs`), and UI (`src/jobs/mod.rs`). Totals
are cached until jobs or capture settings change; they are not recomputed each
animation frame.


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
the scanline. Pause freezes the trail; disabling the overlay clears it.

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
**Observation model** (swath width, sweep visibility, coverage and a point sampler),
and **Data**. Collapse it with **−** and reopen it with **EO** on the right edge or
the **Earth Observation** toggle on the left. Its sections scroll independently of
the constellation settings on the left. Width remains editable when the sweep
preview is hidden. Choose an Earth view, click **Sampler**, then click its 3D
globe. A yellow marker and coordinates identify the point; the calculation runs
in the background. Dragging still rotates the globe; Escape cancels picking.

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

These are continuous full-swath acquisition estimates, independent of playback
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
The background forecast uses visible Earth constellations, the pushbroom swath
width, and the current simulation time. Choose 0.01–30 days, longitude sample count,
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
