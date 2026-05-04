use std::collections::HashMap;
use murasa_core::{
    config::EngineConfig,
    data_source::{DataRegistry, DataSource, SourceType},
    base::DataLoaderPlugin,
};

/// Loader for land-use vector data.
#[derive(Debug)]
pub struct LandUseLoader {
    name: String,
    path: Option<std::path::PathBuf>,
}

impl LandUseLoader {
    /// Create from config if a land-use path is available.
    pub fn from_config(config: &EngineConfig) -> Option<Self> {
        let path = config.paths.as_ref()?.land_use.as_ref()?.clone();
        Some(Self { name: "LandUseLoader".to_string(), path: Some(path) })
    }
}

const LU_PROVIDES: &[&str] = &["land_use"];
const LU_REQUIRES: &[&str] = &[];

impl DataLoaderPlugin for LandUseLoader {
    fn provides(&self) -> &'static [&'static str] { LU_PROVIDES }
    fn requires(&self) -> &'static [&'static str] { LU_REQUIRES }
    fn name(&self) -> &str { &self.name }

    fn can_handle(&self, _config: &EngineConfig) -> bool {
        self.path.is_some()
    }

    fn load(&mut self, _registry: &mut DataRegistry) -> anyhow::Result<HashMap<String, DataSource>> {
        let mut sources = HashMap::new();
        if let Some(ref path) = self.path {
            self.log_loading(&format!("Loading land-use from {}", path.display()));
            sources.insert(
                "land_use".to_string(),
                DataSource::from_path("land_use", SourceType::Vector, path.clone()),
            );
        }
        Ok(sources)
    }
}
