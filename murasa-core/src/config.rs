use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result};

/// Configuration for a single risk factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorConfig {
    /// Human-readable name of the factor.
    pub name: String,
    /// Weight in the MCDA calculation (0..1).
    #[serde(default)]
    pub weight: f64,
    /// Path to the source dataset.
    pub source_path: Option<PathBuf>,
    /// Source type: raster, vector, derived, etc.
    #[serde(default = "default_source_type")]
    pub source_type: String,
    /// Optional processor override.
    pub processor: Option<String>,
    /// Free-form processor parameters.
    #[serde(default)]
    pub parameters: HashMap<String, serde_yaml::Value>,
}

fn default_source_type() -> String {
    "raster".to_string()
}

impl FactorConfig {
    /// Create a new factor config with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            weight: 0.0,
            source_path: None,
            source_type: default_source_type(),
            processor: None,
            parameters: HashMap::new(),
        }
    }
}

/// Spatial reference and grid configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConfig {
    /// Target CRS (e.g. EPSG:4326).
    #[serde(default = "default_target_crs")]
    pub target_crs: String,
    /// Metric CRS for distance calculations (e.g. EPSG:3857).
    #[serde(default = "default_metric_crs")]
    pub metric_crs: String,
    /// Target pixel resolution in meters.
    #[serde(default = "default_resolution")]
    pub resolution: f64,
    /// Registry key used for the study area.
    #[serde(default = "default_study_area_key")]
    pub study_area_key: String,
    /// Optional province filter.
    pub filter_province: Option<Vec<String>>,
    /// Optional city filter.
    pub filter_city: Option<Vec<String>>,
    /// Optional district filter.
    pub filter_district: Option<Vec<String>>,
}

fn default_target_crs() -> String { "EPSG:4326".to_string() }
fn default_metric_crs() -> String { "EPSG:3857".to_string() }
fn default_resolution() -> f64 { 10.0 }
fn default_study_area_key() -> String { "admin".to_string() }

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            target_crs: default_target_crs(),
            metric_crs: default_metric_crs(),
            resolution: default_resolution(),
            study_area_key: default_study_area_key(),
            filter_province: None,
            filter_city: None,
            filter_district: None,
        }
    }
}

impl SpatialConfig {
    /// Return the most specific active filter level.
    pub fn get_filter_level(&self) -> &str {
        if self.filter_district.is_some() {
            "district"
        } else if self.filter_city.is_some() {
            "city"
        } else if self.filter_province.is_some() {
            "province"
        } else {
            "full"
        }
    }

    /// Return the active filter values, if any.
    pub fn get_active_filter(&self) -> Option<&Vec<String>> {
        self.filter_district.as_ref()
            .or_else(|| self.filter_city.as_ref())
            .or_else(|| self.filter_province.as_ref())
    }
}

/// Output format and directory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Directory to write results.
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    /// Output formats (geojson, geopackage, csv).
    #[serde(default = "default_formats")]
    pub formats: Vec<String>,
    /// Whether to emit a human-readable report.
    #[serde(default = "default_true")]
    pub generate_report: bool,
}

fn default_output_dir() -> PathBuf { PathBuf::from("./output") }
fn default_formats() -> Vec<String> { vec!["geojson".to_string()] }
fn default_true() -> bool { true }

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_dir: default_output_dir(),
            formats: default_formats(),
            generate_report: true,
        }
    }
}

/// Classification / breaks configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationConfig {
    /// Method: quantile, jenks, equal_interval.
    #[serde(default = "default_class_method")]
    pub method: String,
    /// Number of classes.
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    /// Human-readable class labels.
    #[serde(default = "default_class_names")]
    pub class_names: Vec<String>,
    /// Optional hex color palette.
    #[serde(default)]
    pub class_colors: Vec<String>,
    /// Optional explicit thresholds.
    pub thresholds: Option<Vec<f64>>,
}

fn default_class_method() -> String { "quantile".to_string() }
fn default_num_classes() -> usize { 5 }
fn default_class_names() -> Vec<String> {
    vec![
        "Very Low".to_string(),
        "Low".to_string(),
        "Moderate".to_string(),
        "High".to_string(),
        "Very High".to_string(),
    ]
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            method: default_class_method(),
            num_classes: default_num_classes(),
            class_names: default_class_names(),
            class_colors: Vec::new(),
            thresholds: None,
        }
    }
}

/// Legacy path-centric configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathsConfig {
    /// Digital Elevation Model path.
    pub dem: Option<PathBuf>,
    /// Pre-computed slope path.
    pub slope: Option<PathBuf>,
    /// Topographic Wetness Index path.
    pub twi: Option<PathBuf>,
    /// Rainfall data path.
    pub rainfall: Option<PathBuf>,
    /// River network path.
    pub river: Option<PathBuf>,
    /// Land-use data path.
    pub land_use: Option<PathBuf>,
    /// Administrative boundary directories.
    #[serde(default)]
    pub admin_dirs: Vec<PathBuf>,
    /// Output directory override.
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

/// Legacy weight-centric configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeightsConfig {
    pub rainfall: f64,
    pub elevation: f64,
    pub slope: f64,
    pub twi: f64,
    pub proximity: f64,
    pub land_use: f64,
}

impl WeightsConfig {
    /// Sum of all weights.
    pub fn total(&self) -> f64 {
        self.rainfall + self.elevation + self.slope
            + self.twi + self.proximity + self.land_use
    }

    /// Convert to a name -> weight map.
    pub fn as_dict(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("rainfall".to_string(), self.rainfall);
        m.insert("elevation".to_string(), self.elevation);
        m.insert("slope".to_string(), self.slope);
        m.insert("twi".to_string(), self.twi);
        m.insert("proximity".to_string(), self.proximity);
        m.insert("land_use".to_string(), self.land_use);
        m
    }
}

/// Top-level engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Analysis name.
    #[serde(default = "default_unnamed")]
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Analysis type: susceptibility, hazard, vulnerability, risk.
    #[serde(default = "default_analysis_type")]
    pub analysis_type: String,
    /// Spatial settings.
    #[serde(default)]
    pub spatial: SpatialConfig,
    /// Output settings.
    #[serde(default)]
    pub output: OutputConfig,
    /// Classification settings.
    #[serde(default)]
    pub classification: ClassificationConfig,
    /// Factor map (new-style config).
    #[serde(default)]
    pub factors: HashMap<String, FactorConfig>,
    /// Administrative directories.
    #[serde(default)]
    pub admin_dirs: Vec<PathBuf>,
    /// Default admin filename pattern.
    #[serde(default = "default_admin_filename")]
    pub admin_filename: String,
    /// Free-form parameters.
    #[serde(default)]
    pub parameters: HashMap<String, serde_yaml::Value>,
    /// Legacy paths block.
    pub paths: Option<PathsConfig>,
    /// Legacy weights block.
    pub weights: Option<WeightsConfig>,
    /// Legacy plugin parameters block.
    #[serde(default)]
    pub plugin_parameters: HashMap<String, serde_yaml::Value>,
}

fn default_unnamed() -> String { "unnamed_analysis".to_string() }
fn default_analysis_type() -> String { "susceptibility".to_string() }
fn default_admin_filename() -> String { "ADMINISTRASIDESA_AR_25K.shp".to_string() }

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            name: default_unnamed(),
            description: String::new(),
            analysis_type: default_analysis_type(),
            spatial: SpatialConfig::default(),
            output: OutputConfig::default(),
            classification: ClassificationConfig::default(),
            factors: HashMap::new(),
            admin_dirs: Vec::new(),
            admin_filename: default_admin_filename(),
            parameters: HashMap::new(),
            paths: None,
            weights: None,
            plugin_parameters: HashMap::new(),
        }
    }
}

impl EngineConfig {
    /// Load configuration from a YAML file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        let mut config: EngineConfig = serde_yaml::from_str(&contents)
            .with_context(|| "Failed to parse YAML configuration")?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a YAML file.
    pub fn to_yaml(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        let contents = serde_yaml::to_string(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Save configuration to a JSON file.
    pub fn to_json(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Return a map of factor names to weights.
    pub fn get_weights(&self) -> HashMap<String, f64> {
        self.factors.iter().map(|(k, v)| (k.clone(), v.weight)).collect()
    }

    /// Resolve a factor source path by name.
    pub fn get_factor_path(&self, factor_name: &str) -> Option<&Path> {
        self.factors.get(factor_name).and_then(|f| f.source_path.as_deref())
    }

    /// Validate that weights sum to 1.0 (if factors are defined).
    pub fn validate_weights(&self) -> Result<()> {
        if self.factors.is_empty() {
            return Ok(());
        }
        let total: f64 = self.factors.values().map(|f| f.weight).sum();
        if (total - 1.0).abs() > 0.001 {
            anyhow::bail!("Weights must sum to 1.0, got {:.3}", total);
        }
        Ok(())
    }

    /// Normalize factor weights so they sum to 1.0.
    pub fn normalize_weights(&mut self) {
        let total: f64 = self.factors.values().map(|f| f.weight).sum();
        if total > 0.0 {
            for factor in self.factors.values_mut() {
                factor.weight /= total;
            }
        }
    }

    /// Run full validation and prepare output directories.
    pub fn validate(&mut self) -> Result<()> {
        self.validate_weights()?;
        std::fs::create_dir_all(&self.output.output_dir)?;
        // Check missing factor paths
        for (name, factor) in &self.factors {
            if let Some(ref p) = factor.source_path {
                if !p.exists() {
                    log::warn!("Factor path not found: {}: {}", name, p.display());
                }
            }
        }
        Ok(())
    }
}
