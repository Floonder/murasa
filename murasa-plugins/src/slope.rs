use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Slope-based risk parameter.
///
/// `inverse=true`: flat areas = high risk (flood - water accumulates).
/// `inverse=false`: steep areas = high risk (landslide).
#[derive(Debug)]
pub struct SlopePlugin {
    name: String,
    weight: f64,
    inverse: bool,
}

impl SlopePlugin {
    /// Create a new slope plugin.
    pub fn new(weight: f64, inverse: bool) -> Self {
        Self { name: "slope".to_string(), weight, inverse }
    }
}

impl ParameterPlugin for SlopePlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("slope") || registry.has("dem")
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
        let slope = if registry.has("slope") {
            ctx.resample_raster(registry.get("slope", true)?, "bilinear")?
        } else {
            let dem = ctx.resample_raster(registry.get("dem", true)?, "bilinear")?;
            murasa_core::terrain::calculate_slope(&dem, ctx.resolution as f32, true)
        };
        let mut normalized = percentile(&slope, 5.0, 95.0);
        if self.inverse {
            normalized = murasa_core::normalize::invert(&normalized);
        }
        self.log_result(&normalized);
        Ok(normalized)
    }
}
