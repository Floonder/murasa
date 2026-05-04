use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Topographic Wetness Index (TWI) risk parameter.
#[derive(Debug)]
pub struct TWIPlugin {
    name: String,
    weight: f64,
}

impl TWIPlugin {
    /// Create a new TWI plugin.
    pub fn new(weight: f64) -> Self {
        Self { name: "twi".to_string(), weight }
    }
}

impl ParameterPlugin for TWIPlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has("twi") || registry.has("dem")
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
        let twi = if registry.has("twi") {
            ctx.resample_raster(registry.get("twi", true)?, "bilinear")?
        } else {
            let dem = ctx.resample_raster(registry.get("dem", true)?, "bilinear")?;
            // Simplified TWI proxy = ln(flow_accumulation / slope)
            let slope = murasa_core::terrain::calculate_slope(&dem, ctx.resolution as f32, false);
            let flow = murasa_core::hydro::flow_accumulation(&dem);
            Array2::from_shape_fn(dem.raw_dim(), |(i, j)| {
                let s = slope[[i, j]].max(0.001);
                let f = flow[[i, j]].max(1.0);
                (f / s).ln().max(0.0) as f32
            })
        };
        let normalized = percentile(&twi, 5.0, 95.0);
        self.log_result(&normalized);
        Ok(normalized)
    }
}
