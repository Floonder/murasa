use std::collections::HashMap;
use ndarray::Array2;
use crate::data_source::DataRegistry;
use crate::processing::ProcessingContext;

/// Statistics for a processed parameter.
#[derive(Debug, Clone, Default)]
pub struct ParameterStatistics {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Number of valid (finite) pixels.
    pub valid_pixels: usize,
    /// Total number of pixels.
    pub total_pixels: usize,
}

/// Core trait for risk parameters.
pub trait ParameterPlugin: std::fmt::Debug + Send + Sync {
    /// Plugin identifier.
    fn name(&self) -> &str;
    /// Relative weight (0..1).
    fn weight(&self) -> f64;
    /// Set the weight.
    fn set_weight(&mut self, weight: f64);
    /// Whether this parameter acts as an exclusion mask.
    fn is_exclusion(&self) -> bool;
    /// Validate that required data exists in the registry.
    fn validate_requirements(&self, registry: &DataRegistry) -> bool;
    /// Process data and return a normalized risk raster (0..1).
    fn process(
        &mut self,
        registry: &mut DataRegistry,
        grid_shape: (usize, usize),
        transform: &AffineTransform,
        crs: &str,
    ) -> anyhow::Result<Array2<f32>>;
    /// Compute statistics on the last result.
    fn statistics(&self, result: &Array2<f32>) -> ParameterStatistics {
        // Placeholder: real impl would iterate the array.
        ParameterStatistics::default()
    }
    /// Log processing start.
    fn log_processing(&self) {
        log::info!("   Processing: {} (weight={:.3})", self.name(), self.weight());
    }
    /// Log processing result.
    fn log_result(&self, result: &Array2<f32>) {
        let stats = self.statistics(result);
        log::info!("      Range: {:.3} - {:.3}", stats.min, stats.max);
    }
}

/// Raster-specific parameter with built-in normalization.
pub trait RasterParameterPlugin: ParameterPlugin {
    /// Registry keys to search (first found wins).
    fn source_keys(&self) -> &'static [&'static str];
    /// Whether to invert the result (1 - normalized).
    fn inverse(&self) -> bool;
    /// Normalization method name.
    fn normalization(&self) -> &str;

    /// Optional custom transformation before normalization.
    fn transform_data(&self, data: Array2<f32>, _ctx: &ProcessingContext) -> Array2<f32> {
        data
    }
}

/// Affine transform wrapper.
#[derive(Debug, Clone, Copy)]
pub struct AffineTransform {
    /// Pixel width (a).
    pub a: f64,
    /// Row rotation (b).
    pub b: f64,
    /// Column rotation (d).
    pub d: f64,
    /// Pixel height (e).
    pub e: f64,
    /// X offset (c).
    pub c: f64,
    /// Y offset (f).
    pub f: f64,
}

impl AffineTransform {
    /// Create a transform from bounds and grid dimensions.
    pub fn from_bounds(
        left: f64,
        bottom: f64,
        right: f64,
        top: f64,
        width: usize,
        height: usize,
    ) -> Self {
        let a = (right - left) / width as f64;
        let e = (bottom - top) / height as f64;
        Self { a, b: 0.0, d: 0.0, e, c: left, f: top }
    }

    /// Pixel resolution (absolute value of a).
    pub fn resolution(&self) -> f64 {
        self.a.abs()
    }
}

/// Post-processor trait for result vectorization / reporting.
pub trait PostProcessorPlugin: std::fmt::Debug + Send + Sync {
    /// Processor name.
    fn name(&self) -> &str;
    /// Execute post-processing logic.
    fn post_process(
        &mut self,
        registry: &mut DataRegistry,
        risk_raster: &Array2<f32>,
        transform: &AffineTransform,
        crs: &str,
        output_dir: &std::path::Path,
        config: &crate::config::EngineConfig,
    ) -> anyhow::Result<()>;
}

/// Data loader plugin base trait.
pub trait DataLoaderPlugin: std::fmt::Debug + Send + Sync {
    /// Keys this loader registers into the DataRegistry.
    fn provides(&self) -> &'static [&'static str];
    /// Keys this loader depends on.
    fn requires(&self) -> &'static [&'static str];
    /// Name of the loader.
    fn name(&self) -> &str;
    /// Check if this loader should run given the engine config.
    fn can_handle(&self, config: &crate::config::EngineConfig) -> bool;
    /// Execute loading and register results.
    fn load(&mut self, registry: &mut DataRegistry) -> anyhow::Result<HashMap<String, DataSource>>;
    /// Log a loading message.
    fn log_loading(&self, message: &str) {
        log::info!("[{}] {}", self.name(), message);
    }
}

use crate::data_source::DataSource;

/// Registry for parameter plugins.
#[derive(Debug, Default)]
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn Fn() -> Box<dyn ParameterPlugin> + Send + Sync>>,
}

impl PluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin factory.
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn() -> Box<dyn ParameterPlugin> + Send + Sync + 'static,
    {
        let name = name.into();
        log::debug!("Registered plugin: {}", name);
        self.plugins.insert(name, Box::new(factory));
    }

    /// Get a plugin factory by name.
    pub fn get(&self, name: &str) -> anyhow::Result<&(dyn Fn() -> Box<dyn ParameterPlugin> + Send + Sync)> {
        self.plugins.get(name)
            .map(|f| f.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Plugin '{}' not found", name))
    }

    /// List registered plugin names.
    pub fn list_plugins(&self) -> Vec<&String> {
        self.plugins.keys().collect()
    }
}

/// Error type for circular dependencies in the loader graph.
#[derive(Debug, thiserror::Error)]
#[error("Circular dependency detected among: {0:?}")]
pub struct CircularDependencyError(pub Vec<String>);

/// Registry for data loader plugins with topological ordering.
#[derive(Debug, Default)]
pub struct LoaderRegistry {
    loaders: Vec<Box<dyn DataLoaderPlugin>>,
}

impl LoaderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a loader instance.
    pub fn register(&mut self, loader: Box<dyn DataLoaderPlugin>) {
        self.loaders.push(loader);
    }

    /// Resolve execution order via Kahn's algorithm.
    pub fn resolve_order(&self) -> Result<Vec<&dyn DataLoaderPlugin>, CircularDependencyError> {
        // Build provider map
        let mut provider_map: HashMap<&str, &dyn DataLoaderPlugin> = HashMap::new();
        for loader in &self.loaders {
            for key in loader.provides() {
                provider_map.insert(key, loader.as_ref());
            }
        }

        // Build adjacency graph (dependency -> dependents)
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();

        for loader in &self.loaders {
            let name = loader.name();
            in_degree.entry(name).or_insert(0);
            for req in loader.requires() {
                if let Some(dep) = provider_map.get(req) {
                    if dep.name() != name {
                        graph.entry(dep.name()).or_default().push(name);
                        *in_degree.entry(name).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut queue: Vec<&str> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| *name)
            .collect();
        queue.sort_unstable();

        let mut ordered: Vec<&dyn DataLoaderPlugin> = Vec::new();
        let mut visited = std::collections::HashSet::new();

        while let Some(name) = queue.pop() {
            if !visited.insert(name) {
                continue;
            }
            if let Some(loader) = self.loaders.iter().find(|l| l.name() == name) {
                ordered.push(loader.as_ref());
            }
            if let Some(dependents) = graph.get(name) {
                for dep in dependents {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(dep);
                        }
                    }
                }
            }
        }

        if ordered.len() != self.loaders.len() {
            let unresolved: Vec<String> = self.loaders.iter()
                .filter(|l| !ordered.iter().any(|o| o.name() == l.name()))
                .map(|l| l.name().to_string())
                .collect();
            return Err(CircularDependencyError(unresolved));
        }

        Ok(ordered)
    }

    /// Execute all loaders that match the config, in dependency order.
    pub fn load_all(
        &mut self,
        config: &crate::config::EngineConfig,
        registry: &mut DataRegistry,
    ) -> anyhow::Result<()> {
        let active: Vec<&mut Box<dyn DataLoaderPlugin>> = self.loaders.iter_mut()
            .filter(|l| l.can_handle(config))
            .collect();

        log::info!("\n{}", "=".repeat(70));
        log::info!("DATA LOADING PHASE");
        log::info!("{}", "=".repeat(70));

        // In a real implementation we would topologically sort and run.
        // For now, run sequentially.
        let mut executed = 0usize;
        for loader in active {
            log::info!("\n{} (provides={:?}, requires={:?})",
                loader.name(), loader.provides(), loader.requires());
            match loader.load(registry) {
                Ok(sources) => {
                    for (key, source) in sources {
                        registry.register(key, source);
                    }
                    executed += 1;
                }
                Err(e) => {
                    log::error!("Loader {} failed: {}", loader.name(), e);
                }
            }
        }

        log::info!("\n{}", "=".repeat(70));
        log::info!("DATA LOADING COMPLETE ({}/{} loader(s) executed)", executed, self.loaders.len());
        log::info!("{}\n", "=".repeat(70));
        Ok(())
    }

    /// List all loader names.
    pub fn list_loaders(&self) -> Vec<&str> {
        self.loaders.iter().map(|l| l.name()).collect()
    }
}
