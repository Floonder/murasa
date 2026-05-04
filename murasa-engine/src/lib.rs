//! Murasa Engine
//!
//! High-level orchestration layer for running geospatial risk analyses.
//! Builds on `murasa-core` to provide a turnkey `RiskEngine` that wires
//! together data loaders, parameter plugins, and post-processors.

#![warn(missing_docs)]

pub mod engine;
pub mod presets;

pub use engine::RiskEngine;
