use murasa_core::config::{EngineConfig, WeightsConfig};
use std::collections::HashMap;

#[test]
fn test_config_defaults() {
    let config = EngineConfig::default();
    assert_eq!(config.name, "unnamed_analysis");
    assert_eq!(config.spatial.target_crs, "EPSG:4326");
    assert!(config.factors.is_empty());
}

#[test]
fn test_weights_total() {
    let weights = WeightsConfig {
        rainfall: 0.25,
        elevation: 0.20,
        slope: 0.15,
        twi: 0.15,
        proximity: 0.15,
        land_use: 0.10,
    };
    assert!((weights.total() - 1.0).abs() < 0.001);
}

#[test]
fn test_normalize_weights() {
    let mut config = EngineConfig::default();
    config.factors.insert("elevation".to_string(), murasa_core::config::FactorConfig {
        name: "elevation".to_string(),
        weight: 0.5,
        source_path: None,
        source_type: "raster".to_string(),
        processor: None,
        parameters: HashMap::new(),
    });
    config.factors.insert("slope".to_string(), murasa_core::config::FactorConfig {
        name: "slope".to_string(),
        weight: 0.5,
        source_path: None,
        source_type: "raster".to_string(),
        processor: None,
        parameters: HashMap::new(),
    });
    config.normalize_weights();
    let total: f64 = config.factors.values().map(|f| f.weight).sum();
    assert!((total - 1.0).abs() < 0.001);
}
