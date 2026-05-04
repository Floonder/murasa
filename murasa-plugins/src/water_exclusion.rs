use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Water exclusion mask parameter.
///
/// Pixels classified as water receive NaN (excluded from risk calculation).
#[derive(Debug)]
pub struct WaterExclusionPlugin {
    name: String,
    weight: f64,
}

impl WaterExclusionPlugin {
    /// Create a new water exclusion plugin.
    pub fn new() -> Self {
        Self { name: "water_exclusion".to_string(), weight: 0.0 }
    }
}

impl ParameterPlugin for WaterExclusionPlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { true }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("water") || registry.has("land_use")
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
        let ctx = ProcessingContext::new(grid_shape, transform.clone(), crs);
        let source_key = if registry.has("water") { "water" } else { "land_use" };
        let data = ctx.rasterize_vector(registry.get(source_key, true)?, None, true, 0.0)?;
        // Mark water pixels (value == 1) as exclusion
        let mask = data.mapv(|v| if (v - 1.0).abs() < 0.1 { 1.0f32 } else { 0.0f32 });
        self.log_result(&mask);
        Ok(mask)
    }
}
