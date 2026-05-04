use ndarray::{Array2, Array};

/// Fill local depressions (sinks) in DEM using iterative morphological reconstruction.
pub fn fill_sinks(dem: &Array2<f32>, epsilon: f32) -> Array2<f32> {
    let shape = dem.raw_dim();
    let mut result = Array2::<f32>::from_elem(shape, f32::INFINITY);
    let mut mask_boundary = Array2::<bool>::from_elem(shape, true);
    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            mask_boundary[[i, j]] = false;
        }
    }
    for ((i, j), &v) in dem.indexed_iter() {
        if v.is_nan() {
            mask_boundary[[i, j]] = true;
        }
    }
    for ((i, j), &is_boundary) in mask_boundary.indexed_iter() {
        if is_boundary {
            result[[i, j]] = dem[[i, j]];
        }
    }

    let max_iter = if dem.len() < 1_000_000 { 1000 } else { 100 };
    for _ in 0..max_iter {
        let prev = result.clone();
        for i in 1..shape[0] - 1 {
            for j in 1..shape[1] - 1 {
                let min_n = [
                    prev[[i - 1, j]],
                    prev[[i + 1, j]],
                    prev[[i, j - 1]],
                    prev[[i, j + 1]],
                    prev[[i - 1, j - 1]],
                    prev[[i - 1, j + 1]],
                    prev[[i + 1, j - 1]],
                    prev[[i + 1, j + 1]],
                ].iter().copied().fold(f32::INFINITY, f32::min);
                result[[i, j]] = dem[[i, j]].max(min_n);
            }
        }
        if result == prev {
            break;
        }
    }
    result
}

/// Compute simplified D8 Flow Accumulation proxy based on slope.
pub fn flow_accumulation(dem: &Array2<f32>) -> Array2<f32> {
    let shape = dem.raw_dim();
    let mut slope_rad = Array2::<f32>::zeros(shape);
    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            let dy = (dem[[i + 1, j]] - dem[[i - 1, j]]) / 2.0;
            let dx = (dem[[i, j + 1]] - dem[[i, j - 1]]) / 2.0;
            slope_rad[[i, j]] = (dx.powi(2) + dy.powi(2)).sqrt().atan().max(0.001);
        }
    }
    slope_rad.mapv(|v| 1.0 / v)
}
