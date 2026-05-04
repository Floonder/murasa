use ndarray::{Array2, Array};
use crate::data_source::DataSource;
use crate::base::AffineTransform;

/// Raster resampler to a target grid.
#[derive(Debug, Clone)]
pub struct RasterResampler {
    grid_shape: (usize, usize),
    transform: AffineTransform,
    crs: String,
    method: String,
}

impl RasterResampler {
    /// Create a new resampler.
    pub fn new(
        grid_shape: (usize, usize),
        transform: AffineTransform,
        crs: impl Into<String>,
        method: impl Into<String>,
    ) -> Self {
        Self { grid_shape, transform, crs: crs.into(), method: method.into() }
    }

    /// Resample the source raster to the target grid.
    pub fn resample(&self, source: &DataSource) -> anyhow::Result<Array2<f32>> {
        log::debug!("Resampling {} using {}", source.name, self.method);
        // Placeholder: return an empty grid.
        Ok(Array::zeros(self.grid_shape))
    }
}

/// Vector-to-raster resampler.
#[derive(Debug, Clone)]
pub struct VectorResampler {
    grid_shape: (usize, usize),
    transform: AffineTransform,
    crs: String,
}

impl VectorResampler {
    /// Create a new vector resampler.
    pub fn new(grid_shape: (usize, usize), transform: AffineTransform, crs: impl Into<String>) -> Self {
        Self { grid_shape, transform, crs: crs.into() }
    }

    /// Rasterize vector features onto the target grid.
    pub fn resample(
        &self,
        source: &DataSource,
        value_column: Option<&str>,
        _all_touched: bool,
        fill_value: f32,
    ) -> anyhow::Result<Array2<f32>> {
        log::debug!("Rasterizing {} (column={:?})", source.name, value_column);
        // Placeholder: return a grid filled with fill_value.
        Ok(Array::from_elem(self.grid_shape, fill_value))
    }
}
