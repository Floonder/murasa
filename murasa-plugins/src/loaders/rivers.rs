use std::collections::HashMap;
use murasa_core::{
    config::EngineConfig,
    data_source::{DataRegistry, DataSource, SourceType},
    base::DataLoaderPlugin,
};

/// Loader for river network vector data.
#[derive(Debug)]
pub struct RiversLoader {
    name: String,
    path: Option<std::path::PathBuf>,
}

impl RiversLoader {
    /// Create from config if a river path is available.
    pub fn from_config(config: &EngineConfig) -> Option<Self> {
        let path = config.paths.as_ref()?.river.as_ref()?.clone();
        Some(Self { name: "RiversLoader".to_string(), path: Some(path) })
    }
}

const RIVERS_PROVIDES: &[&str] = &["river"];
const RIVERS_REQUIRES: &[&str] = &[];

impl DataLoaderPlugin for RiversLoader {
    fn provides(&self) -> &'static [&'static str] { RIVERS_PROVIDES }
    fn requires(&self) -> &'static [&'static str] { RIVERS_REQUIRES }
    fn name(&self) -> &str { &self.name }

    fn can_handle(&self, _config: &EngineConfig) -> bool {
        self.path.is_some()
    }

    fn load(&mut self, _registry: &mut DataRegistry) -> anyhow::Result<HashMap<String, DataSource>> {
        let mut sources = HashMap::new();
        if let Some(ref path) = self.path {
            self.log_loading(&format!("Loading rivers from {}", path.display()));
            sources.insert(
                "river".to_string(),
                DataSource::from_path("river", SourceType::Vector, path.clone()),
            );
        }
        Ok(sources)
    }
}
