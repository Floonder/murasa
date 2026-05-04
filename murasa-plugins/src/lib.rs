//! Murasa Plugins
//!
//! Built-in parameter plugins and data loaders for the Murasa risk engine.

#![warn(missing_docs)]

pub mod curvature;
pub mod elevation;
pub mod land_use;
pub mod proximity;
pub mod rainfall;
pub mod reporting;
pub mod slope;
pub mod twi;
pub mod water_exclusion;

pub mod loaders;

pub use elevation::ElevationPlugin;
pub use slope::SlopePlugin;
pub use rainfall::RainfallPlugin;
pub use twi::TWIPlugin;
pub use proximity::ProximityPlugin;
pub use land_use::LandUsePlugin;
pub use water_exclusion::WaterExclusionPlugin;
pub use reporting::VectorExplainabilityPlugin;
