# Acquisition opportunities and unconstrained capture

Application-level workload/capture library above `leo-sim`. No renderer, GPU,
processing kernels, execution actions, admission controller or rolling-horizon
scheduler. The GUI and CLI share this implementation and job/persistence models.

```sh
cargo test --manifest-path crates/leo-capture/Cargo.toml
cargo run --release --manifest-path crates/leo-capture/Cargo.toml -- \
  crates/leo-capture/scenarios/overlap.json /tmp/capture-report.json /tmp/opportunities.jsonl
cargo run --release --manifest-path crates/leo-capture/Cargo.toml -- \
  crates/leo-capture/scenarios/default-jobs.json /tmp/default-capture-report.json
```

The overlap fixture has three jobs over two partially overlapping rectangles.
One rectangle appears twice. A completed update of all three requires half as
many physical crop bytes as their independently counted logical bytes. The default
fixture uses the bundled 50,000 AOIs, a 13×13 ADAM-style constellation, 20 km swath,
one day, noon-centred job preferences and daylight gating.

## Inputs and reproducibility

The first argument is JSON with `epoch_unix_s`, optional `start_s`, `swath_km`,
`shells` (the shared `leo-sim::orbit::WalkerShell` schema), and `aoi_files`.
Paths in the configuration resolve relative to that configuration file. Each
AOI file is a plain or gzipped `leo-aoi` document; holes and dateline-separated
MultiPolygons are preserved. IDs, areas, anchors and geometry hashes are checked
against a loaded workload. Invalid polygon topology is rejected explicitly.

Set `workload` to an existing gzipped workload, or omit it to generate assignments
using `generator`. Optional `capture` overrides the workload's image encoding;
otherwise its settings (or CaptureProfile defaults for a generated workload) apply.
`options` supplies duration, track-step size, geographic chart size, edge spacing,
storage capacity and mosaic roundoff tolerance. Omitted values are recorded in
resolved form in the report. All output sizes are decimal bytes. The default
storage is 32,000,000,000,000 bytes per satellite.

The report contains resolved configuration, actual job assignments, source hashes,
capture settings, per-job fulfilment, per-satellite retained bytes, opportunity
counters and overlap-reuse measurements. `LEO_SIM_REVISION` may be set at build
time to record a source revision; archive source AOI files and lockfiles too.
Outputs are written through temporary files. The optional third argument streams
opportunity records to JSONL without holding the full opportunity graph in memory.
Times in events and reports are seconds from `epoch_unix_s + start_s`.

## Contract

- A job has `independent` or `mosaic` acquisition fulfilment, independently of its
  Map/Window pipeline. Mosaics accumulate the union of captured polygons, never
  the sum of strip areas. Independent crops become ready immediately; mosaics
  become ready only after complete coverage (default numerical tolerance 1 ppm).
- Acquisition slots are `[phase + n×cadence, phase + (n+1)×cadence)`, starting at
  n=0. No prehistory is seeded. Slot boundaries split strips spatially; captures
  cannot be reused as fresh observations in another slot. The unfinished final
  slot is censored, not reported as a missed deadline.
- All pieces of a mosaic must come from its slot. Thus the initial maximum mosaic
  collection span equals the cadence. There is no separate tighter coherence bound.
- Job preferred local solar times use a seeded truncated normal with mean 12 h
  and σ=4 h, bounded to [0,24). The default circular tolerance is ±1 h and minimum
  solar elevation is 0°. Both are editable. Timing is evaluated at the AOI anchor;
  solar elevation is checked at crop vertices at interval start, middle and end.
  No cloud simulation or random availability is applied.
- Only newly needed crop polygons are retained. Overlapping eligible jobs share
  one physical capture; histories and application outputs remain separate future
  concerns. The model assumes lossless identity of overlapping pixels from the
  same strip and common capture encoding, with negligible cropping cost.
- `fulfilled_logical_bytes` counts new imagery credited to jobs (including partial
  mosaics). `retained_physical_bytes` counts the union per physical capture.
  Their difference is `reuse_saved_bytes`. `reused_physical_bytes` counts physical
  regions used by at least two jobs once, even if three or more jobs share them.
- All useful crops remain stored through the horizon. No deletion, downlink,
  processing or replication is simulated. `first_capacity_exceeded_s` diagnoses
  this retention scenario against finite storage; it does not reject a capture.
  Write throughput, energy and compute capacity are unconstrained.
- Priorities are uniform integers 1–5 by default, with larger values strictly
  preferred. `lowest_priority_first` defines the future whole-job drop order;
  it is not a weighted-profit objective. Ties are deterministic by job key.
  This baseline accepts all eligible captures and always reports zero drops.

`due_updates`, `complete_due_updates`, `partial_due_updates` and
`missed_due_updates` describe respectively all closed slots, fully covered slots,
partially covered slots and slots with no captured area. `ready_area_km2` counts
all area made available for independent processing or completed mosaics, including
the open final slot. Nothing here executes Map/Window operators.

## Geometry, time and scale

`Atlas` splits the original polygons at a fixed geographic grid (default 2°) and
projects each piece into a common spherical Lambert azimuthal equal-area chart.
This permits exact planar unions across different jobs in a patch. GeoJSON edges
are densified (default 1 km); spherical geometric areas can differ slightly from
the source's ellipsoidal area. Reports retain both. Polygon holes are not sampled
away. The grid is a computational partition, not a tiling approximation to demand.

`opportunities` streams swept scanline intersections in time-step/satellite/patch
order. A conservative sphere hierarchy excludes distant patches. Propagation
comes from a callback: the GUI uses its existing Kepler/J2/SGP4 adapters; the CLI
uses shared Walker elements. Heading comes from a centred 0.5-second ground
velocity sample, including samples 0.25 s outside the run's ends. Adjacent strips
share identical boundary scanlines to avoid artificial wedge gaps.

Default track intervals are 5 seconds. Temporal eligibility conservatively checks
the whole short interval, so usable sub-step slivers near a time/daylight boundary
may be rejected. The footprint between samples and curved polygon boundaries are
numerically approximated; this is an unconstrained upper baseline within that
numerical model, not an exact continuous-time coverage certificate. Repeat
representative runs with half the step and edge spacing before interpreting small
differences. Events within one step use deterministic satellite order; potential
first-capture attribution differences are bounded by that step.

The baseline retains coverage geometry only for active request slots, freeing it
at deadlines. Completed requests retain a completion flag until their deadline.
The atlas, per-job counters and optional output stream remain; no images are
allocated. Cancellation is checked during atlas construction and each time step.

Tests cover independent readiness versus mosaics, cumulative capture, repeated
passes, partial overlaps, timing-dependent sharing, daylight, midnight wrapping,
holes, dateline/polar geometry, temporal convergence, finite-capacity diagnostics,
priority order, deterministic normal draws and v1→v2 workload migration.


## Resumable sessions

`session::Session` wraps the same opportunity generator and baseline with an owned
state and retained crop ledger. `Inputs` embeds the constellation (analytical
orbital elements or SGP4 constants), orbit epoch and run offset, full jobs and AOI
polygons, sensor profile, swath and fixed run options. `advance(ticks)` advances
on the original fixed tick grid; live playback and fast-forward differ only in
pacing and rendering. The final tick may be shorter at the configured horizon.

`meters()` reports physical bytes in `(elapsed − 86400, elapsed]`, per satellite,
with captures credited at their end time. Occupancy is cumulative and never
reduced by expiry of that reporting window. A failed update is a closed cadence
slot lacking full AOI coverage; partial independent captures remain ready. Counts
include slots with no opportunities, and exclude incomplete final slots.

`save(path)` streams schema-1 gzip JSON to a unique sibling temporary file, syncs
and renames it. `load(path, progress)` validates schema, inputs, geometry hashes,
coverage state, ledger accounting and propagated saved positions. It rebuilds
only the spatial atlas, never the capture history. All job/slot coverage polygons,
expiry deadlines, completed/partial counters, physical crops and their job
associations survive. Source datasets and network access are unnecessary. A
checkpoint can be large: every retained crop footprint and allocation is kept;
no pixels are allocated. The batch CLI still omits this ledger unless using the
session API, and remains suitable for lightweight opportunity/report generation.

The native GUI drives the session on a worker, publishing bounded UI snapshots.
It never drops simulation steps when it skips a display update. Fast-forward
skips the main app's scene rendering and preview physics; pause/save commands are
handled between chunks of at most 12 ticks. A loaded session is initially paused.
The constrained scheduler, storage enforcement, downlink and processing remain
future work.
