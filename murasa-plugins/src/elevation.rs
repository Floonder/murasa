use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Elevation-based risk parameter.
///
/// `inverse=true`: low elevation = high risk (flood).
/// `inverse=false`: high elevation = high risk (landslide on steep areas).
#[derive(Debug)]
pub struct ElevationPlugin {
    name: String,
    weight: f64,
    inverse: bool,
}

impl ElevationPlugin {
    /// Create a new elevation plugin.
    pub fn new(weight: f64, inverse: bool) -> Self {
        Self { name: "elevation".to_string(), weight, inverse }
    }
}

impl ParameterPlugin for ElevationPlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("dem") || registry.has("elevation")
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
        use murasa_core::normalize::percentile;
        let ctx = ProcessingContext::new(grid_shape, transform.clone(), crs);
        let source = if registry.has("dem") { "dem" } else { "elevation" };
        let data = ctx.resample_raster(registry.get(source, true)?, "bilinear")?;
        let mut normalized = percentile(&data, 5.0, 95.0);
        if self.inverse {
            normalized = murasa_core::normalize::invert(&normalized);
        }
        self.log_result(&normalized);
        Ok(normalized)
    }
}
