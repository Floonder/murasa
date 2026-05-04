use std::collections::HashMap;
use murasa_core::{
    config::EngineConfig,
    data_source::{DataRegistry, DataSource, SourceType},
    base::DataLoaderPlugin,
};

/// Loader for administrative boundary vector data.
#[derive(Debug)]
pub struct AdminBoundariesLoader {
    name: String,
    dirs: Vec<std::path::PathBuf>,
    filename: String,
}

impl AdminBoundariesLoader {
    /// Create a new admin boundaries loader from config.
    pub fn from_config(config: &EngineConfig) -> Self {
        Self {
            name: "AdminBoundariesLoader".to_string(),
            dirs: config.admin_dirs.clone(),
            filename: config.admin_filename.clone(),
        }
    }
}

const ADMIN_PROVIDES: &[&str] = &["admin"];
const ADMIN_REQUIRES: &[&str] = &[];

impl DataLoaderPlugin for AdminBoundariesLoader {
    fn provides(&self) -> &'static [&'static str] { ADMIN_PROVIDES }
    fn requires(&self) -> &'static [&'static str] { ADMIN_REQUIRES }
    fn name(&self) -> &str { &self.name }

    fn can_handle(&self, config: &EngineConfig) -> bool {
        !config.admin_dirs.is_empty()
    }

    fn load(&mut self, _registry: &mut DataRegistry) -> anyhow::Result<HashMap<String, DataSource>> {
        self.log_loading("Scanning admin directories...");
        let mut sources = HashMap::new();
        for dir in &self.dirs {
            let path = dir.join(&self.filename);
            if path.exists() {
                sources.insert(
                    "admin".to_string(),
                    DataSource::from_path("admin", SourceType::Vector, path),
                );
                self.log_loading(&format!("Loaded admin from {}", dir.display()));
            }
        }
        Ok(sources)
    }
}
