//! Data loader plugins for ingesting external datasets.

pub mod admin_boundaries;
pub mod landuse;
pub mod rivers;

pub use admin_boundaries::AdminBoundariesLoader;
pub use landuse::LandUseLoader;
pub use rivers::RiversLoader;
