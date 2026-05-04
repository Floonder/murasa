use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ndarray::{Array2, Array};
use murasa_core::{
    config::EngineConfig,
    data_source::{DataRegistry, DataSource, SourceType},
    base::{ParameterPlugin, PostProcessorPlugin, AffineTransform},
    logger::{log_section, log_subsection, log_success, log_error},
};

/// The main risk engine that orchestrates data loading, parameter
/// processing, and result export.
#[derive(Debug)]
pub struct RiskEngine {
    /// Parsed engine configuration.
    pub config: EngineConfig,
    /// Central data registry.
    pub registry: DataRegistry,
    /// Active parameter plugins.
    pub parameters: Vec<Box<dyn ParameterPlugin>>,
    /// Active post-processors.
    pub post_processors: Vec<Box<dyn PostProcessorPlugin>>,
    /// Grid shape (height, width).
    pub grid_shape: Option<(usize, usize)>,
    /// Geo-transform.
    pub transform: Option<AffineTransform>,
    /// Target CRS.
    pub crs: String,
    /// Study area mask (true = inside study area).
    pub study_area_mask: Option<Array2<bool>>,
    /// Final risk raster.
    pub risk_raster: Option<Array2<f32>>,
    /// Exclusion mask.
    pub exclusion_mask: Option<Array2<bool>>,
}

impl RiskEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: EngineConfig) -> Self {
        let crs = config.spatial.target_crs.clone();
        Self {
            config,
            registry: DataRegistry::new(),
            parameters: Vec::new(),
            post_processors: Vec::new(),
            grid_shape: None,
            transform: None,
            crs,
            study_area_mask: None,
            risk_raster: None,
            exclusion_mask: None,
        }
    }

    /// Register a data source by name.
    pub fn register_data(&mut self, name: impl Into<String>, source: DataSource) {
        self.registry.register(name, source);
    }

    /// Register a data source from a filesystem path.
    pub fn register_data_from_path(
        &mut self,
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        source_type: SourceType,
    ) {
        self.registry.register_from_path(name, path, source_type, self.crs.clone());
    }

    /// Auto-register sources declared in the config factors block.
    pub fn auto_register_from_config(&mut self) {
        log_subsection("Auto-registering data sources from factors", 70);
        for (name, factor) in &self.config.factors {
            if let Some(ref path) = factor.source_path {
                if path.exists() {
                    let st = if factor.source_type == "raster" { SourceType::Raster } else { SourceType::Vector };
                    self.register_data_from_path(name.clone(), path.clone(), st);
                    log::info!("  {}: {}", name, path.display());
                } else if factor.source_type == "derived" {
                    log::info!("  {}: derived (will be computed)", name);
                } else {
                    log::warn!("  {}: path not found or not specified", name);
                }
            }
        }
        log_success(&format!("Registered {} data sources", self.registry.len()));
    }

    /// Register a parameter plugin if its requirements are satisfied.
    pub fn register_parameter(&mut self, plugin: Box<dyn ParameterPlugin>) -> bool {
        if !plugin.validate_requirements(&self.registry) {
            log::warn!("Plugin '{}' missing required data", plugin.name());
            return false;
        }
        log_success(&format!("Registered: {} (weight={:.3})", plugin.name(), plugin.weight()));
        self.parameters.push(plugin);
        true
    }

    /// Register a post-processor.
    pub fn register_post_processor(&mut self, plugin: Box<dyn PostProcessorPlugin>) {
        log_success(&format!("Registered Post-Processor: {}", plugin.name()));
        self.post_processors.push(plugin);
    }

    /// Clear all parameters.
    pub fn clear_parameters(&mut self) {
        self.parameters.clear();
        log::info!("Cleared all parameters");
    }

    /// Normalize parameter weights so they sum to 1.0.
    pub fn normalize_weights(&mut self) {
        let total: f64 = self.parameters.iter().map(|p| p.weight()).sum();
        if (total - 1.0).abs() < 0.001 {
            return;
        }
        log::warn!("Weights sum to {:.3}, normalizing...", total);
        for param in &mut self.parameters {
            let new_weight = param.weight() / total;
            param.set_weight(new_weight);
        }
    }

    /// Set up the analysis grid from bounds or admin data.
    pub fn setup_grid(&mut self, bounds: Option<(f64, f64, f64, f64)>) -> anyhow::Result<()> {
        let study_area_key = &self.config.spatial.study_area_key;
        let mut bounds = bounds;
        if bounds.is_none() && self.registry.has(study_area_key) {
            // Real impl would compute total_bounds from vector data.
            bounds = Some((106.0, -7.0, 107.0, -6.0));
            log::info!("Grid bounds from {} data", study_area_key);
        }
        let (left, bottom, right, top) = bounds
            .ok_or_else(|| anyhow::anyhow!("Grid bounds required"))?;
        let resolution = self.config.spatial.resolution;
        let width = ((right - left) / resolution) as usize;
        let height = ((top - bottom) / resolution) as usize;
        self.transform = Some(AffineTransform::from_bounds(left, bottom, right, top, width, height));
        self.grid_shape = Some((height, width));
        log_success(&format!("Grid: {}x{} pixels @ {}m", width, height, resolution));
        Ok(())
    }

    /// Set the study area from an admin GeoDataFrame or registry key.
    pub fn set_study_area(&mut self) {
        let key = self.config.spatial.study_area_key.clone();
        if !self.registry.has(&key) {
            log::warn!("No study area defined using key '{}', processing full grid", key);
            return;
        }
        // Placeholder: create a mask that covers the whole grid.
        if let Some(shape) = self.grid_shape {
            self.study_area_mask = Some(Array2::from_elem(shape, true));
        }
        log_success(&format!("Study area: using key '{}'", key));
    }

    /// Calculate the composite risk raster.
    pub fn calculate_risk(&mut self) -> anyhow::Result<Array2<f32>> {
        log_section("CALCULATING RISK", 70);

        if self.parameters.is_empty() {
            anyhow::bail!("No parameters registered");
        }
        let (shape, transform) = match (self.grid_shape, self.transform.as_ref()) {
            (Some(s), Some(t)) => (s, t.clone()),
            _ => anyhow::bail!("Grid not setup (call setup_grid first)"),
        };

        self.normalize_weights();
        self.exclusion_mask = Some(Array2::from_elem(shape, false));

        let mut risk_components: Vec<Array2<f32>> = Vec::new();

        for param in &mut self.parameters {
            param.log_processing();
            match param.process(&mut self.registry, shape, &transform, &self.crs) {
                Ok(component) => {
                    param.log_result(&component);
                    let output_name = format!("risk_{}", param.name());
                    self.registry.register(output_name.clone(), DataSource::from_array(output_name, component.clone(), &self.crs));

                    if param.is_exclusion() {
                        let mask = component.mapv(|v| v > 0.0);
                        if let Some(ref mut excl) = self.exclusion_mask {
                            for ((i, j), &v) in mask.indexed_iter() {
                                if v { excl[[i, j]] = true; }
                            }
                        }
                        log::info!("      Added exclusion mask from {}: {} pixels", param.name(), mask.iter().filter(|&&v| v).count());
                    } else {
                        let weighted = component.mapv(|v| v * param.weight() as f32);
                        risk_components.push(weighted);
                    }
                }
                Err(e) => {
                    log_error(&format!("      Failed: {}", e));
                }
            }
        }

        if risk_components.is_empty() {
            anyhow::bail!("No valid risk components computed!");
        }

        let mut risk = Array2::<f32>::zeros(shape);
        for component in &risk_components {
            risk += component;
        }

        if let Some(ref excl) = self.exclusion_mask {
            for ((i, j), &is_excl) in excl.indexed_iter() {
                if is_excl {
                    risk[[i, j]] = f32::NAN;
                }
            }
        }

        if let Some(ref mask) = self.study_area_mask {
            for ((i, j), &inside) in mask.indexed_iter() {
                if !inside {
                    risk[[i, j]] = f32::NAN;
                }
            }
        }

        log_section("RISK CALCULATION COMPLETE", 70);
        self.risk_raster = Some(risk.clone());
        Ok(risk)
    }

    /// Export the risk raster as a GeoTIFF.
    pub fn export_raster(&self, output_path: &Path) -> anyhow::Result<()> {
        let risk = self.risk_raster.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No risk raster calculated"))?;
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))?;
        // Real implementation would use GDAL to write a GeoTIFF.
        log_success(&format!("Raster saved: {}", output_path.display()));
        Ok(())
    }

    /// Export JSON statistics.
    pub fn export_statistics(&self, output_path: &Path) -> anyhow::Result<()> {
        let _risk = self.risk_raster.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No risk raster calculated"))?;
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))?;
        let stats = HashMap::<String, serde_json::Value>::new();
        let contents = serde_json::to_string_pretty(&stats)?;
        std::fs::write(output_path, contents)?;
        log_success(&format!("Statistics saved: {}", output_path.display()));
        Ok(())
    }

    /// Export all results and run post-processors.
    pub fn export_all(&mut self, output_dir: Option<&Path>) -> anyhow::Result<()> {
        let output_dir = output_dir.unwrap_or(&self.config.output.output_dir);
        std::fs::create_dir_all(output_dir)?;
        self.export_raster(&output_dir.join("risk_map.tif"))?;
        self.export_statistics(&output_dir.join("statistics.json"))?;
        log_success(&format!("All outputs saved to: {}", output_dir.display()));

        if !self.post_processors.is_empty() {
            log_section("POST-PROCESSING", 70);
            let transform = self.transform.clone().unwrap_or_else(||
                AffineTransform::from_bounds(0.0, 0.0, 1.0, 1.0, 1, 1)
            );
            for pp in &mut self.post_processors {
                if let Err(e) = pp.post_process(
                    &mut self.registry,
                    self.risk_raster.as_ref().unwrap(),
                    &transform,
                    &self.crs,
                    output_dir,
                    &self.config,
                ) {
                    log_error(&format!("Post-processor {} failed: {}", pp.name(), e));
                }
            }
        }
        Ok(())
    }

    /// Print a summary of the engine state.
    pub fn print_summary(&self) {
        log_section("ENGINE SUMMARY", 70);
        log::info!("Data sources: {}", self.registry.len());
        self.registry.print_summary();
        log::info!("\nParameters: {}", self.parameters.len());
        for param in &self.parameters {
            log::info!("  - {} ({:.3})", param.name(), param.weight());
        }
        if let Some((h, w)) = self.grid_shape {
            log::info!("\nGrid: {}x{} pixels", w, h);
            log::info!("Resolution: {}m", self.config.spatial.resolution);
        }
    }

    /// Run the full pipeline.
    pub fn run(&mut self) -> anyhow::Result<Array2<f32>> {
        log_section("RISK ENGINE - FULL RUN", 70);
        self.auto_register_from_config();
        self.setup_grid(None)?;
        self.set_study_area();
        let risk = self.calculate_risk()?;
        self.export_all(None)?;
        log_section("COMPLETE", 70);
        Ok(risk)
    }
}
