use std::collections::HashMap;
use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Rainfall-based risk parameter.
///
/// Expects a vector source with a rainfall intensity column.
#[derive(Debug)]
pub struct RainfallPlugin {
    name: String,
    weight: f64,
    column_name: String,
    value_mapping: Option<HashMap<i32, f32>>,
}

impl RainfallPlugin {
    /// Create a new rainfall plugin.
    pub fn new(weight: f64, column_name: impl Into<String>, value_mapping: Option<HashMap<i32, f32>>) -> Self {
        Self {
            name: "rainfall".to_string(),
            weight,
            column_name: column_name.into(),
            value_mapping,
        }
    }
}

impl ParameterPlugin for RainfallPlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("rainfall")
    }

    fn process(
        &mut self,
        registry: &mut DataRegistry,
        grid_shape: (usize, usize),
        transform: &AffineTransform,
        crs: &str,
    ) -> anyhow::Result<Array2<f32>> {
        self.log_processing();
        use murasa_core::processing::ProcessingContext;
        use murasa_core::normalize::minmax;
        let ctx = ProcessingContext::new(grid_shape, transform.clone(), crs);
        let source = registry.get("rainfall", true)?;
        let data = ctx.rasterize_vector(source, Some(&self.column_name), true, 0.0)?;
        let mapped = if let Some(ref mapping) = self.value_mapping {
            data.mapv(|v| {
                let key = v as i32;
                mapping.get(&key).copied().unwrap_or(v)
            })
        } else {
            data
        };
        let normalized = minmax(&mapped);
        self.log_result(&normalized);
        Ok(normalized)
    }
}
