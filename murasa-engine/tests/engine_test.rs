use murasa_engine::RiskEngine;
use murasa_core::config::EngineConfig;

#[test]
fn test_engine_creation() {
    let config = EngineConfig::default();
    let engine = RiskEngine::new(config);
    assert_eq!(engine.crs, "EPSG:4326");
    assert!(engine.grid_shape.is_none());
}

#[test]
fn test_preset_bandung() {
    let config = murasa_engine::presets::Presets::flood_bandung();
    assert_eq!(config.name, "bandung_flood_susceptibility");
    assert_eq!(config.spatial.resolution, 10.0);
}
