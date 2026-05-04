use murasa_core::config::{EngineConfig, WeightsConfig, SpatialConfig, OutputConfig, ClassificationConfig};

/// Pre-built engine presets for common analysis types.
pub struct Presets;

impl Presets {
    /// Flood susceptibility preset for the Bandung region.
    pub fn flood_bandung() -> EngineConfig {
        let mut config = EngineConfig::default();
        config.name = "bandung_flood_susceptibility".to_string();
        config.description = "Flood susceptibility analysis for Bandung, Indonesia".to_string();
        config.analysis_type = "susceptibility".to_string();
        config.spatial = SpatialConfig {
            target_crs: "EPSG:4326".to_string(),
            metric_crs: "EPSG:3857".to_string(),
            resolution: 10.0,
            study_area_key: "admin".to_string(),
            filter_province: Some(vec!["Jawa Barat".to_string()]),
            filter_city: None,
            filter_district: None,
        };
        config.output = OutputConfig {
            output_dir: std::path::PathBuf::from("./output/bandung"),
            formats: vec!["geojson".to_string(), "geopackage".to_string()],
            generate_report: true,
        };
        config.classification = ClassificationConfig {
            method: "quantile".to_string(),
            num_classes: 5,
            class_names: vec![
                "Very Low".to_string(),
                "Low".to_string(),
                "Moderate".to_string(),
                "High".to_string(),
                "Very High".to_string(),
            ],
            class_colors: vec![],
            thresholds: None,
        };
        config.weights = Some(WeightsConfig {
            rainfall: 0.25,
            elevation: 0.20,
            slope: 0.15,
            twi: 0.15,
            proximity: 0.15,
            land_use: 0.10,
        });
        config
    }

    /// Landslide susceptibility preset.
    pub fn landslide_default() -> EngineConfig {
        let mut config = EngineConfig::default();
        config.name = "landslide_susceptibility".to_string();
        config.analysis_type = "susceptibility".to_string();
        config.spatial.resolution = 12.5;
        config.weights = Some(WeightsConfig {
            rainfall: 0.20,
            elevation: 0.15,
            slope: 0.30,
            twi: 0.10,
            proximity: 0.15,
            land_use: 0.10,
        });
        config
    }
}
