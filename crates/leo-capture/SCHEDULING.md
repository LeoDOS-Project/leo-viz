# Acquisition graphs

This generates imaging choices and their transitions. It does not select paths,
execute jobs, or certify mosaic completion. The GUI capture baseline is unchanged.

## Run the saved workload with WorldView-60

From the repository root, after saving an AOI/jobs preset in the EO pane:

```sh
cargo run --release --manifest-path crates/leo-capture/Cargo.toml -j 2 \
  --bin leo-scheduling-graph -- \
  crates/leo-capture/scenarios/worldview60-scheduling.json
```

The example reads `assets/presets/default-preset.json.gz` without regenerating
geometry or assignments. It uses WorldView-60: 20 planes × 3 satellites, F=14,
617 km, 97.9°, J2. The orbital epoch and planning start are both
**2026-11-13 00:00 UTC**, with a 24-hour horizon. These are design orbits, not
live spacecraft ephemerides. The output directory must not already exist.
Paths resolve relative to the config file. `first_satellite` and `satellite_count`
allow scaling tests on a contiguous subset.

Generated files are saved under `artifacts/scheduling-graphs/`, outside Cargo's
build directory and excluded from Git. The completed first run is recorded in
[`scenarios/worldview60-scheduling-result.json`](scenarios/worldview60-scheduling-result.json).
It contains 973,789 acquisition choices and 8,094,888 edges across all 60 satellites.
These are graph-generation results, not a solved observation schedule.

## Finite imaging model

- Acquisitions occupy one 30-second time step. The swept-scanline polygon model
  uses 2° geographic charts and 2 km polygon-edge spacing. These are configurable.
- Imaging width is 13.1 km, with constant acceptable off-nadir quality.
- Reach is a circle 450 km in diameter. Across-track offsets are 11.79 km apart;
  along-track offsets are −150, 0 and +150 km. Centres reserve half the imaging
  width inside the reach boundary. This produces **91 pointing modes**.
- Modes follow the moving ground track at fixed offsets. This is a restricted
  family of forward-scanning strips, not every possible agile imaging trajectory.
- Maneuvers use a nadir-relative quaternion surrogate. WorldView-3's published
  200 km / 12 s retargeting example is interpreted as nadir-to-target at 617 km.
  An **assumed 2-second settling allowance** gives an effective angular rate near
  1.79°/s. This is not a reconstructed spacecraft controller. Tracking the moving
  reference frame has no additional cost. No imaging occurs during maneuver edges.
- Transition arrival times round up to acquisition start times. On the default
  grid every change of mode consumes one empty 30-second step. Continuing in
  the same mode has no additional settling delay. Small rapid acquisitions are
  therefore deliberately coarsely represented.
- Saved phase/cadence and illumination rules apply. Crops crossing cadence
  boundaries are split spatially and temporally. Mosaic span equals cadence.
  Initial imagery is empty and initial pointing is nadir.
- Weekly/monthly opportunities are included even without a deadline in this
  horizon. Such jobs are not counted as having due observations.
- No storage, bandwidth, processing, energy or cloud constraints.

Reference: [WorldView-3 manufacturer datasheet](https://www.edpmedia.com/assets/maxar_brochure/WorldView3.pdf).
Finer time/pointing sampling or a different maneuver law can change feasibility.
An optimum on these graphs would apply to this finite model only.

## Sparse paths and optional acquisition

Action `a` has start node `2*a` and end node `2*a+1`, connected by two parallel
edges: **capture** earns its crop coverage; **skip** earns nothing. Both occupy
the interval in the same pointing mode. A waiting edge joins each action end
to the next action start in that mode, allowing observations to be skipped.

From an end node we need only the earliest reachable start in each other mode.
The wait chain reaches all later starts without acquiring intermediate images.
This preserves the acquisition subsequences of the dense graph for this finite
model: replace a dense transition with an earlier valid transition and wait/skip
forward. This relies on equal-length, nonoverlapping slots within a mode and the
time-invariant maneuver law. It is not a generic transitive reduction of a graph
whose vertices are mandatory observations.

When all mode changes have the same rounded lag, transition edges share time
nodes: an end connects to a hub at its ready time, and that hub connects to each
mode's earliest available start. Returning to the original mode this way is merely
an alternative to its wait chain. No coverage is earned at a hub.

This factoring preserves feasibility, but **exact physical slew time is not an
additive edge cost through a shared hub**. A later secondary objective minimizing
actual slew time must inspect consecutive selected acquisitions or use direct
transition edges. Small tests exhaustively compare dense/sparse capture sequences
and compare every reachable next acquisition with/without hubs.

## Files and interpretation

Each satellite has two compressed JSONL files:

- `satellite-NNN.actions.jsonl.gz`: `id`, `tick`, `mode`, and eligible `crops`.
  Crops carry job index, cadence slot, start/end times, patch ID, area and polygons.
- `satellite-NNN.graph.jsonl.gz`: a header, action topology records, and optional
  hub records. `next_same` and `maneuver_to` reference action IDs. `maneuver_hub`
  references a graph node ID. Hub records provide `hub`, `ready_tick`, and
  `targets` (action IDs). The header supplies source/sink IDs and `source_to`.

The source represents nadir at time zero. Source transitions allow the initial
maneuver. Source-to-sink idle is always allowed; each mode's last end connects to
the sink. Unreachable actions are counted in the manifest. A solver must require
source-connected paths before crediting acquisitions.

Crop coordinates are kilometres in spherical Lambert azimuthal equal-area
charts, not longitude/latitude. `charts.json` supplies chart centres as Earth-fixed
unit vectors, with radius 6371 km and the `geometry::Chart` coordinate convention.
Job indices refer to the archived preset's `workload.jobs` order. An atlas patch
is a computational chart, not a mosaic demand cell.

One capture edge may serve several jobs. Preserve those associations and use
polygon unions for coverage/reuse; summing crop areas double-counts overlap.
The later optimizer needs coverage constraints or adaptive spatial refinement
with a polygon verifier. The union of all candidates is only an upper bound on
scheduled coverage within this candidate set, not a feasible schedule.

The manifest records configuration, constellation, source checksum, deadlines,
sizes, memory and runtime. A copy of the original preset is archived. Inputs are
never edited. Job/geometry mismatches and a preset changed during preparation
are rejected. Files use temporary paths and atomic rename. Only `manifest.json`
with `status: complete` marks a completed run; failures leave `INCOMPLETE`.

An existing complete direct graph can be compacted without recomputing crops:

```sh
crates/leo-capture/target/release/leo-scheduling-graph --compact \
  artifacts/scheduling-graphs/worldview60-24h-20261113
```

Compaction marks the manifest while working and supports retry. It does not
change candidate footprints or the archived preset.

## Resources and subsequent optimization

The example uses eight workers, bounded by the shared CPU limit. Geometry work
uses Utility QoS on macOS and 75% calculation duty. Eligibility masks are shared,
satellites are generated independently, and crop records stream directly to gzip.
The existing 16 GiB process-memory threshold applies. On macOS generation also
stops below 1 GiB free disk. These are cooperative checks, not hard OS limits.

Paths must ultimately be selected jointly across satellites: mosaics can require
several paths, while captures may serve multiple jobs. Independent per-satellite
score maximization would not solve the constellation problem. Successful 24-hour
graph generation does not imply a full-day MILP will solve efficiently. A shorter
optimization horizon carrying partial mosaics forward remains an option.
