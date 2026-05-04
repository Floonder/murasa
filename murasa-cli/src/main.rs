use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;
use murasa_core::{
    config::EngineConfig,
    logger::{LoggerSetup, log_section, log_subsection, log_success, log_error},
    base::LoaderRegistry,
};
use murasa_engine::RiskEngine;
use murasa_plugins::{
    ElevationPlugin, SlopePlugin, RainfallPlugin,
    TWIPlugin, ProximityPlugin, LandUsePlugin,
    WaterExclusionPlugin, VectorExplainabilityPlugin,
    loaders::{AdminBoundariesLoader, RiversLoader, LandUseLoader},
};

#[derive(Parser, Debug)]
#[command(name = "murasa", about = "Geospatial Risk Assessment Engine", version)]
struct Cli {
    /// Path to YAML configuration file.
    #[arg(short, long, default_value = "flood_bandung_config.yaml")]
    config: PathBuf,
}

fn main() -> Result<()> {
    LoggerSetup::init("murasa", None)?;
    let args = Cli::parse();

    log_section("Flood susceptibility pipeline", 70);
    log::info!("Config: {}", args.config.display());

    let mut config = EngineConfig::from_yaml(&args.config)?;
    log_subsection("Loading configuration", 70);
    log::info!("  Analysis : {}", config.name);
    log::info!("  CRS      : {}", config.spatial.target_crs);
    log::info!("  Resolution: {}m", config.spatial.resolution);
    if !config.factors.is_empty() {
        log::info!("  Factors  : {:?}", config.factors.keys().collect::<Vec<_>>());
    }

    log_subsection("Init engine", 70);
    let mut engine = RiskEngine::new(config.clone());
    engine.auto_register_from_config();

    log_subsection("Loading data", 70);
    let mut loader_registry = LoaderRegistry::new();
    loader_registry.register(Box::new(AdminBoundariesLoader::from_config(&config)));
    if let Some(loader) = RiversLoader::from_config(&config) {
        loader_registry.register(Box::new(loader));
    }
    if let Some(loader) = LandUseLoader::from_config(&config) {
        loader_registry.register(Box::new(loader));
    }
    loader_registry.load_all(&config, &mut engine.registry)?;

    log_subsection("Registering risk parameters", 70);
    let mut registered = 0usize;

    if let Some(ref weights) = config.weights {
        let params_cfg = &config.plugin_parameters;
        macro_rules! register_if_weighted {
            ($factor:ident, $plugin:expr) => {
                let w = weights.$factor;
                if w > 0.0 {
                    if engine.register_parameter($plugin) {
                        registered += 1;
                    }
                }
            };
        }
        register_if_weighted!(rainfall, Box::new(RainfallPlugin::new(
            weights.rainfall,
            params_cfg.get("rainfall")
                .and_then(|v| v.get("column_name")).and_then(|v| v.as_str())
                .unwrap_or("gridcode"),
            None,
        )));
        register_if_weighted!(elevation, Box::new(ElevationPlugin::new(
            weights.elevation,
            params_cfg.get("elevation")
                .and_then(|v| v.get("invert")).and_then(|v| v.as_bool())
                .unwrap_or(true),
        )));
        register_if_weighted!(slope, Box::new(SlopePlugin::new(
            weights.slope,
            params_cfg.get("slope")
                .and_then(|v| v.get("invert")).and_then(|v| v.as_bool())
                .unwrap_or(true),
        )));
        register_if_weighted!(twi, Box::new(TWIPlugin::new(weights.twi)));
        register_if_weighted!(proximity, Box::new(ProximityPlugin::new(
            weights.proximity,
            params_cfg.get("proximity")
                .and_then(|v| v.get("feature")).and_then(|v| v.as_str())
                .unwrap_or("river"),
            params_cfg.get("proximity")
                .and_then(|v| v.get("max_distance")).and_then(|v| v.as_f64())
                .unwrap_or(500.0),
        )));
        register_if_weighted!(land_use, Box::new(LandUsePlugin::new(
            weights.land_use,
            std::collections::HashMap::new(),
            None,
        )));
    } else if !config.factors.is_empty() {
        for (factor_name, factor_cfg) in &config.factors {
            let plugin: Option<Box<dyn murasa_core::base::ParameterPlugin>> = match factor_name.as_str() {
                "elevation" => Some(Box::new(ElevationPlugin::new(factor_cfg.weight, true))),
                "slope" => Some(Box::new(SlopePlugin::new(factor_cfg.weight, true))),
                "rainfall" => Some(Box::new(RainfallPlugin::new(factor_cfg.weight, "gridcode", None))),
                "twi" => Some(Box::new(TWIPlugin::new(factor_cfg.weight))),
                "proximity" => Some(Box::new(ProximityPlugin::new(factor_cfg.weight, "river", 500.0))),
                "land_use" => Some(Box::new(LandUsePlugin::new(factor_cfg.weight, std::collections::HashMap::new(), None))),
                _ => None,
            };
            if let Some(p) = plugin {
                if engine.register_parameter(p) {
                    registered += 1;
                }
            }
        }
    }

    engine.register_parameter(Box::new(WaterExclusionPlugin::new()));
    registered += 1;
    engine.register_post_processor(Box::new(VectorExplainabilityPlugin::new()));

    log::info!("  Total: {} parameter(s) + 1 post-processor registered", registered);

    engine.print_summary();

    log_subsection("Run complete analysis", 70);
    match engine.run() {
        Ok(_) => {
            log_success(&format!("Results saved to: {}", engine.config.output.output_dir.display()));
            log_section("Completed", 70);
        }
        Err(e) => {
            log_error(&format!("Pipeline failed: {}", e));
            std::process::exit(1);
        }
    }

    Ok(())
}
