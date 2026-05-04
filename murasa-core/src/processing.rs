use ndarray::{Array2, Array};
use crate::data_source::DataSource;
use crate::base::AffineTransform;
use crate::resampler::{RasterResampler, VectorResampler};

/// Processing context tied to a target grid.
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Grid shape (height, width).
    pub grid_shape: (usize, usize),
    /// Raster geo-transform.
    pub transform: AffineTransform,
    /// Target CRS.
    pub crs: String,
    /// Pixel resolution.
    pub resolution: f64,
}

impl ProcessingContext {
    /// Create a new processing context.
    pub fn new(grid_shape: (usize, usize), transform: AffineTransform, crs: impl Into<String>) -> Self {
        let resolution = transform.resolution();
        Self { grid_shape, transform, crs: crs.into(), resolution }
    }

    /// Resample a raster DataSource to the analysis grid.
    pub fn resample_raster(&self, source: &DataSource, method: &str) -> anyhow::Result<Array2<f32>> {
        let resampler = RasterResampler::new(self.grid_shape, self.transform.clone(), &self.crs, method);
        resampler.resample(source)
    }

    /// Rasterize a vector DataSource to the analysis grid.
    pub fn rasterize_vector(
        &self,
        source: &DataSource,
        column: Option<&str>,
        all_touched: bool,
        fill_value: f32,
    ) -> anyhow::Result<Array2<f32>> {
        let resampler = VectorResampler::new(self.grid_shape, self.transform.clone(), &self.crs);
        resampler.resample(source, column, all_touched, fill_value)
    }

    /// Create an empty grid filled with a value.
    pub fn empty_grid(&self, fill: f32) -> Array2<f32> {
        Array::from_elem(self.grid_shape, fill)
    }

    /// Create a boolean grid filled with a value.
    pub fn boolean_grid(&self, fill: bool) -> Array2<bool> {
        Array::from_elem(self.grid_shape, fill)
    }

    /// Compute normalized distance to nearest vector feature.
    pub fn distance_to_features(
        &self,
        source: &DataSource,
        max_distance: Option<f64>,
    ) -> anyhow::Result<Array2<f32>> {
        let feature_mask = self.rasterize_vector(source, None, true, 0.0)?;
        let mut dist = self.empty_grid(0.0);
        // Placeholder: real impl would use distance transform (e.g. imageproc or ndimage equivalent).
        for ((i, j), &v) in feature_mask.indexed_iter() {
            if v > 0.5 {
                dist[[i, j]] = 0.0;
            } else {
                dist[[i, j]] = self.resolution as f32 * 10.0;
            }
        }
        if let Some(max_dist) = max_distance {
            let max = max_dist as f32;
            dist.mapv_inplace(|v| (1.0f32 - (v / max)).clamp(0.0, 1.0));
        }
        Ok(dist)
    }
}
