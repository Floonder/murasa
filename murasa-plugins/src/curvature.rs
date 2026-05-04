use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Curvature-based risk parameter.
#[derive(Debug)]
pub struct CurvaturePlugin {
    name: String,
    weight: f64,
    method: String,
}

impl CurvaturePlugin {
    /// Create a new curvature plugin.
    pub fn new(weight: f64, method: impl Into<String>) -> Self {
        Self { name: "curvature".to_string(), weight, method: method.into() }
    }
}

impl ParameterPlugin for CurvaturePlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("dem")
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
        let dem = ctx.resample_raster(registry.get("dem", true)?, "bilinear")?;
        let curvature = murasa_core::terrain::calculate_curvature(&dem, ctx.resolution as f32, &self.method);
        let normalized = minmax(&curvature);
        self.log_result(&normalized);
        Ok(normalized)
    }
}
