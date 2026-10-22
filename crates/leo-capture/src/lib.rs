//! Application-level capture planning, independent of rendering and execution.
pub mod baseline;
pub mod dataset;
pub mod geometry;
mod spatial;
pub mod storage;
#[cfg(test)]
mod tests;
pub mod workload;
pub mod session;
