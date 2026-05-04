use std::collections::HashMap;
use ndarray::Array2;
use murasa_core::{
    config::EngineConfig,
    data_source::DataRegistry,
    base::{PostProcessorPlugin, AffineTransform},
};

/// Post-processor that adds explainability metadata to vector outputs.
#[derive(Debug)]
pub struct VectorExplainabilityPlugin {
    name: String,
}

impl VectorExplainabilityPlugin {
    /// Create a new explainability plugin.
    pub fn new() -> Self {
        Self { name: "vector_explainability".to_string() }
    }
}

impl PostProcessorPlugin for VectorExplainabilityPlugin {
    fn name(&self) -> &str { &self.name }

    fn post_process(
        &mut self,
        _registry: &mut DataRegistry,
        risk_raster: &Array2<f32>,
        _transform: &AffineTransform,
        _crs: &str,
        output_dir: &std::path::Path,
        config: &EngineConfig,
    ) -> anyhow::Result<()> {
        log::info!("[{}] Generating explainability report...", self.name);
        let report_path = output_dir.join("explainability.json");
        let mut report = HashMap::new();
        report.insert("analysis_name", serde_json::Value::String(config.name.clone()));
        report.insert("risk_min", serde_json::Value::Number(
            serde_json::Number::from_f64(0.0).unwrap()
        ));
        report.insert("risk_max", serde_json::Value::Number(
            serde_json::Number::from_f64(1.0).unwrap()
        ));
        let contents = serde_json::to_string_pretty(&report)?;
        std::fs::write(&report_path, contents)?;
        log::info!("Explainability report saved to {}", report_path.display());
        Ok(())
    }
}
