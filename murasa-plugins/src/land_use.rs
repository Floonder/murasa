use std::collections::HashMap;
use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Land-use-based risk parameter.
///
/// Uses a coefficient map to translate land-use codes into risk scores.
#[derive(Debug)]
pub struct LandUsePlugin {
    name: String,
    weight: f64,
    coefficients: HashMap<i32, f32>,
    priorities: Option<HashMap<i32, u8>>,
}

impl LandUsePlugin {
    /// Create a new land-use plugin.
    pub fn new(weight: f64, coefficients: HashMap<i32, f32>, priorities: Option<HashMap<i32, u8>>) -> Self {
        Self { name: "land_use".to_string(), weight, coefficients, priorities }
    }
}

impl ParameterPlugin for LandUsePlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("land_use") || registry.has("landuse")
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
        let source_key = if registry.has("land_use") { "land_use" } else { "landuse" };
        let data = ctx.rasterize_vector(registry.get(source_key, true)?, None, true, 0.0)?;
        let mapped = data.mapv(|v| {
            let key = v as i32;
            self.coefficients.get(&key).copied().unwrap_or(0.5)
        });
        let normalized = minmax(&mapped);
        self.log_result(&normalized);
        Ok(normalized)
    }
}
