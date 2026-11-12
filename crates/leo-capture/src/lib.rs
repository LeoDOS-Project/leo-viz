//! Application-level capture planning, independent of rendering and execution.
pub mod baseline;
pub mod dataset;
pub mod geometry;
pub mod parallel;
pub mod resources;
pub mod session;
mod spatial;
pub mod storage;
#[cfg(test)]
mod tests;
mod validation;
pub mod workload;
