use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ndarray::Array2;

/// Supported source types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceType {
    /// GeoTIFF / raster grid.
    Raster,
    /// Vector features (Shapefile, GeoJSON, etc.).
    Vector,
    /// Point data.
    Point,
    /// Tabular data (CSV, Excel).
    Table,
    /// Remote API endpoint.
    Api,
    /// LiDAR / point cloud.
    PointCloud,
    /// NetCDF dataset.
    NetCdf,
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceType::Raster => write!(f, "raster"),
            SourceType::Vector => write!(f, "vector"),
            SourceType::Point => write!(f, "point"),
            SourceType::Table => write!(f, "table"),
            SourceType::Api => write!(f, "api"),
            SourceType::PointCloud => write!(f, "point_cloud"),
            SourceType::NetCdf => write!(f, "netcdf"),
        }
    }
}

/// Generic data source with lazy loading.
#[derive(Debug, Clone)]
pub struct DataSource {
    /// Registry key / name.
    pub name: String,
    /// Source type discriminator.
    pub source_type: SourceType,
    /// Optional filesystem path.
    pub path: Option<PathBuf>,
    /// In-memory raster data (if loaded).
    pub raster_data: Option<Array2<f32>>,
    /// CRS as a string (e.g. EPSG:4326).
    pub crs: String,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl DataSource {
    /// Create a new unloaded data source.
    pub fn new(name: impl Into<String>, source_type: SourceType) -> Self {
        Self {
            name: name.into(),
            source_type,
            path: None,
            raster_data: None,
            crs: "EPSG:4326".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Create a source from a path.
    pub fn from_path(
        name: impl Into<String>,
        source_type: SourceType,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            source_type,
            path: Some(path.into()),
            raster_data: None,
            crs: "EPSG:4326".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Create a source wrapping an in-memory array.
    pub fn from_array(
        name: impl Into<String>,
        array: Array2<f32>,
        crs: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            source_type: SourceType::Raster,
            path: None,
            raster_data: Some(array),
            crs: crs.into(),
            metadata: HashMap::new(),
        }
    }

    /// Whether raster data has been loaded.
    pub fn is_loaded(&self) -> bool {
        self.raster_data.is_some()
    }

    /// Load data from disk if not already loaded.
    pub fn load(&mut self) -> anyhow::Result<()> {
        if self.is_loaded() {
            log::debug!("Data already loaded: {}", self.name);
            return Ok(());
        }
        // Real implementation would dispatch to GDAL / rasterio here.
        log::info!("Loading: {} ({})", self.name, self.source_type);
        Ok(())
    }

    /// Unload in-memory data.
    pub fn unload(&mut self) {
        self.raster_data = None;
    }

    /// Get bounds from metadata if available.
    pub fn get_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        // Placeholder: real impl would parse metadata.
        None
    }
}

/// Centralized registry for all data sources.
#[derive(Debug, Default)]
pub struct DataRegistry {
    sources: HashMap<String, DataSource>,
}

impl DataRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a data source.
    pub fn register(&mut self, name: impl Into<String>, source: DataSource) {
        let name = name.into();
        if self.sources.contains_key(&name) {
            log::warn!("Overwriting existing source: {}", name);
        }
        log::info!("Registered: {}", name);
        self.sources.insert(name, source);
    }

    /// Convenience: register from a path.
    pub fn register_from_path(
        &mut self,
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        source_type: SourceType,
        crs: impl Into<String>,
    ) {
        let mut source = DataSource::from_path(name, source_type, path);
        source.crs = crs.into();
        self.register(source.name.clone(), source);
    }

    /// Get a source, optionally auto-loading it.
    pub fn get(&mut self, name: &str, auto_load: bool) -> anyhow::Result<&DataSource> {
        let source = self.sources.get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("Data source '{}' not registered", name))?;
        if auto_load && !source.is_loaded() {
            source.load()?;
        }
        Ok(source)
    }

    /// Mutable access to a source.
    pub fn get_mut(&mut self, name: &str) -> anyhow::Result<&mut DataSource> {
        self.sources.get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("Data source '{}' not registered", name))
    }

    /// Check if a source is registered.
    pub fn has(&self, name: &str) -> bool {
        self.sources.contains_key(name)
    }

    /// List all registered keys.
    pub fn list_sources(&self) -> Vec<&String> {
        self.sources.keys().collect()
    }

    /// List keys filtered by type.
    pub fn list_by_type(&self, source_type: SourceType) -> Vec<&String> {
        self.sources.iter()
            .filter(|(_, s)| s.source_type == source_type)
            .map(|(k, _)| k)
            .collect()
    }

    /// Unload all in-memory data.
    pub fn unload_all(&mut self) {
        for source in self.sources.values_mut() {
            source.unload();
        }
    }

    /// Print a summary of registered sources.
    pub fn print_summary(&self) {
        log::info!("\n{}", "=".repeat(70));
        log::info!("DATA REGISTRY SUMMARY");
        log::info!("{}", "=".repeat(70));
        for (name, source) in &self.sources {
            let status = if source.is_loaded() { "Loaded" } else { "Not loaded" };
            log::info!("{} | {:8} | {}", status, source.source_type.to_string(), name);
        }
        log::info!("{}", "=".repeat(70));
    }
}

impl std::ops::Deref for DataRegistry {
    type Target = HashMap<String, DataSource>;
    fn deref(&self) -> &Self::Target {
        &self.sources
    }
}
