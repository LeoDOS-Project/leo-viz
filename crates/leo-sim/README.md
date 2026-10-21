# Infrastructure simulation

`leo-sim` is a standalone, deterministic infrastructure library and CLI. It has no
GUI, GPU, LeoDOS, TLE-download or native C dependency. The visualizer depends on this
crate; the crate does not depend on the visualizer. Application graphs, rolling
windows, admission policies and job generation belong above this layer.

## Run

From the repository root:

```sh
cargo run --release --manifest-path crates/leo-sim/Cargo.toml -- \
  crates/leo-sim/scenarios/smoke.json /tmp/report.json /tmp/events.jsonl
cargo test --manifest-path crates/leo-sim/Cargo.toml
cargo run --release --manifest-path crates/leo-sim/Cargo.toml --example scale
```

The optional third filename streams events after each step. The report contains
the resolved scenario (including all defaults), simulator version, optional
`LEO_SIM_REVISION` compile-time environment value, wall time, resource counters,
remaining operations, object locations and a bounded event tail. Set
`LEO_SIM_REVISION` when building archived experiment binaries and retain the source
and lockfile. `seed` is recorded for future stochastic environment adapters; the
built-in environment is deterministic and does not consume it. Reports are
observations of state, not resumable checkpoints.

In the existing GUI, click **Sim**, then **Apply / reset**, **Step**, or **Run**.
Edit/paste scenario JSON or load a file. The panel plots the experiment's own
positions and clock and exports reports. Importing the active Earth view copies
its initial Walker orbits and ground-station coordinates. It deliberately uses
explicit scenario defaults for resources, sensing and network configuration;
review these before running an experiment. TLEs, drag, numerical propagation,
partial Walker layouts and fractional phasing are not imported. Existing 3D
routing and physics animations remain visual diagnostics; use the experiment
panel or CLI for measured resource accounting.

## Modules and reuse

| Module | Responsibility |
| --- | --- |
| `orbit` | Pure Keplerian / secular J2 propagation, Walker Delta / Star shells |
| `geometry` | Earth rotation, line of sight, elevation, nadir footprint, sunlight |
| `network` | Geometry-dependent links/contact plans and shared transfer budgets |
| `resources` | Rolling compute duty and optional shared battery/load budgets |
| `storage` | Immutable object identities, copies, capacity reservations |
| `config` | Versioned JSON parameters, defaults and validation |
| `engine` | Fixed-step clock, action execution, metrics and bounded events |

The existing `WalkerConstellation` and `SatelliteState` remain the viewer's types.
Single and bulk analytical Walker queries now call `orbit::Elements::sample`.
Ground geometry, GMST, eclipse detection and net battery integration reuse the
same functions. This also removes the old difference between single-satellite
and bulk J2 propagation. The numerical and lib42 propagators remain in the viewer.

## Simulation contract

All times are seconds from an explicit Unix epoch. Inputs use km, degrees,
**bits/s** for configured network rates, bytes for objects, joules and watts for
energy. Compute uses abstract work units: duration at full utilization is
`work / compute_units_per_s`. Rates can be calibrated with a measured processor
profile later; a work unit is not assumed to be a FLOP.

Each action has a unique `id`, an earliest start `at_s`, and a `type`:

- `compute`: satellite, positive work, resident input IDs, one immutable output.
- `acquire`: satellite, duration, circular target area, output. Requires one sensor
  and full target containment in the nadir footprint throughout the sampled
  exposure. Optional minimum solar elevation bounds illumination over the area.
- `transfer`: object ID and explicit path. Each hop waits for contact/capacity;
  bytes already sent survive contact gaps. Ground stations may only end a path.
- `delete`: explicitly release a resident object. Inputs of running computations
  and active transfer sources are protected.

`Simulator<E>` accepts an `Environment` implementation. `OrbitEnvironment` provides
orbital geometry; the same interface can supply recorded contacts, observation
usability/cloud traces or other controlled fixtures without changing resource
accounting. `submit_actions` accepts validated batches at or after the current
boundary; `step`, `run_until`, `report`, `has_object` and `drain_events` form the
policy-facing API. Batches are atomic on validation failure. Actions can wait for
objects produced by another action in the same batch. Placement, routes, deadlines,
retry/rescheduling decisions and action cancellation are left to a future policy
layer; this first core executes submitted attempts and reports their outcomes.

The scheduler in this crate is only deterministic arbitration: earliest submitted
`at_s`, then submission order. One computation owns each satellite's compute
resource until completion, although duty/energy limits may suspend progress. One
acquisition owns its sensor; losing visibility or power after starting rejects
that attempt and releases its output reservation. Other blocked operations queue.
No operation is assumed successful merely because the simulation horizon ends.

## Communications and data residency

The built-in topology has up to two along-plane and two cross-plane neighbors per
satellite, no inter-shell links, a Walker-phased Delta seam and an open Star seam.
Earth obstruction, maximum range and cross-plane latitude cutoff determine ISL
availability. Rates are configured constants while a link is available. This is
an explicit capacity model, independent of the viewer's illustrative optical
Shannon-capacity tooltip. Replace `Environment::link` to supply measured or
range-dependent rates.

ISLs have independently shared directed capacities (full duplex); both endpoints
also enforce their terminal count. Downlinks have a ground elevation mask,
per-link rate, per-satellite aggregate rate/terminal count, and a station-wide
aggregate rate/contact count. All simultaneous transfers share those budgets.
Terminal assignments last for the entire step. Sensor field of view does not
control ground contact. `OrbitEnvironment::contacts` uses the execution predicate
and reports sampled intervals and unshared capacity; it does not promise that
other transfers will leave that capacity available.

Transfers reserve the whole destination object's space before sending. Partial
objects cannot be consumed or forwarded. A hop finishes serialization at a step
boundary, then propagation/ground-delivery latency elapses before the next hop
can start. Relay copies remain until explicitly deleted. Existing copies are
reused without sending the same object again. Concurrent duplicate deliveries
share the resulting copy. Output space is reserved before compute or sensing
starts; exceeding capacity stalls an operation. Ground storage is unbounded and
ground processing is outside this infrastructure version.

## Compute duty and energy

The default is a rolling duty constraint, not a physical power guarantee:

`integral of compute utilization over (t - window_s, t] <= max_fraction * window_s`.

The default window is 86,400 s and fraction is 0.2. There is no midnight reset.
`initial_utilization` specifies a uniform prehistory, avoiding an unstated initial
budget. Setting it to zero allows an initial burst. Set `duty` to `null` to remove
the constraint. Satellite overrides can model different compute, storage, duty,
sensor, downlink and battery resources.

Optional `battery` parameters enforce per-step energy conservation and a reserve.
Solar generation uses a midpoint cylindrical Earth-eclipse test; idle, compute,
sensing, ISL transmit/receive and downlink loads draw from one budget. Net energy
is integrated **before** capacity clipping. A baseline idle load that alone
breaches the reserve stops the run as infeasible. The configured
`max_average_load_w` bounds average load within a step, not sub-step electrical
peaks. Battery and duty constraints can be used independently or together. This
model does not include attitude-dependent generation, thermal limits or battery
aging, and should be used for sensitivity analysis rather than hardware sizing.

## Discretization, validation and scale

`step_s` is fixed for a run. Action times, acquisition durations, run boundaries
and duty windows must align to it. Calling `step` from a slow GUI, pausing rendering,
or calling `run_until` in different chunks gives the same physical result.
Utilization and rates are fluid within a step; completed outputs become available
only at its end. Link availability is sampled at the step start; acquisition
visibility is checked at start, midpoint and end. Propagation arrivals are rounded
up to a boundary. Contact crossings can therefore shift by a step, and sub-step
visibility gaps can be missed. Always repeat representative experiments at half
the step and compare admission-independent quantities such as contact capacity,
transferred bytes, energy minima and completion times before interpreting results.

Regression fixtures cover orbital closure, shared query propagation, geometry,
contact-plan consistency/convergence, shared links/stations/terminals, propagation,
partial transfers, copy reuse, storage pressure, input lifetime, compute dependency
ordering, rolling budgets/prehistory, shared energy, JSON validation and replay.
The smoke scenario processes a local object then sends its smaller result through
an ISL to ground. It is an infrastructure check, not an application model.

The `scale` fixture runs 1,024 satellites with competing per-satellite compute
activity over one day, at 10 s steps, and asserts the duty-bound work count. It
measures the core, not orbit/network-heavy job scheduling. Positions are computed
on demand and topology is sparse. The core skips idle periods without batteries;
active/battery periods require per-step resource accounting. Event history is
bounded by `event_capacity`; streamed traces still lose events if more than that
many occur within a single step. Check `events_dropped`, or increase capacity.
Memory also includes submitted actions, objects/copies and rolling duty history.

Fault tolerance is intentionally outside this implementation.
