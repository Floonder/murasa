use ndarray::Array2;
use murasa_core::{
    data_source::DataRegistry,
    base::{ParameterPlugin, AffineTransform},
};

/// Proximity to features (rivers, faults, etc.).
///
/// Closer to feature = higher risk.
#[derive(Debug)]
pub struct ProximityPlugin {
    name: String,
    weight: f64,
    feature_name: String,
    max_distance: f64,
}

impl ProximityPlugin {
    /// Create a new proximity plugin.
    pub fn new(weight: f64, feature_name: impl Into<String>, max_distance: f64) -> Self {
        Self {
            name: "proximity".to_string(),
            weight,
            feature_name: feature_name.into(),
            max_distance,
        }
    }
}

impl ParameterPlugin for ProximityPlugin {
    fn name(&self) -> &str { &self.name }
    fn weight(&self) -> f64 { self.weight }
    fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    fn is_exclusion(&self) -> bool { false }

    fn validate_requirements(&self, registry: &DataRegistry) -> bool {
        registry.has(&self.feature_name)
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
        let feature_source = registry.get(&self.feature_name, true)?;
        let result = ctx.distance_to_features(feature_source, Some(self.max_distance))?;
        self.log_result(&result);
        Ok(result)
    }
}
