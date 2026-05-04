//! Murasa Core
//!
//! Core types, traits, and utilities for the Murasa geospatial risk assessment
//! engine. This crate provides the foundational abstractions upon which
//! plugins and engine implementations are built.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod analysis;
pub mod base;
pub mod config;
pub mod data_source;
pub mod hydro;
pub mod logger;
pub mod normalize;
pub mod processing;
pub mod resampler;
pub mod terrain;

pub use config::{EngineConfig, FactorConfig, SpatialConfig, OutputConfig, ClassificationConfig, PathsConfig, WeightsConfig};
pub use data_source::{DataSource, DataRegistry, SourceType};
pub use base::{ParameterPlugin, RasterParameterPlugin, PostProcessorPlugin, DataLoaderPlugin, PluginRegistry, LoaderRegistry};
pub use logger::{LoggerSetup, log_section, log_subsection, log_success, log_error, log_warning, log_progress};
