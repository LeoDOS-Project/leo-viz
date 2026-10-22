//! Renderer-independent, deterministic LEO infrastructure simulation.
//!
//! Time is quantized to a scenario's fixed step. Renderers only inspect snapshots;
//! scheduling policies submit actions. No application/window semantics live here.
pub mod acquisition;
pub mod config;
pub mod engine;
pub mod geometry;
pub mod network;
pub mod orbit;
pub mod observation;
pub mod resources;
pub mod repeat_ground_track;
pub mod revisit;
pub mod storage;
pub use config::*;
pub use engine::{Event, OperationStatus, Report, Simulator};
pub use network::{Environment, OrbitEnvironment};
pub type Result<T> = std::result::Result<T, String>;

/// Version recorded alongside reproducible scenario outputs.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
