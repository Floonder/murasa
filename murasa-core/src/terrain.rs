use ndarray::{Array2, Array};

/// Calculate first derivatives (gradients) of the DEM.
pub fn calculate_gradient(dem: &Array2<f32>, cell_size: f32) -> (Array2<f32>, Array2<f32>) {
    let shape = dem.raw_dim();
    let mut dz_dy = Array2::<f32>::zeros(shape);
    let mut dz_dx = Array2::<f32>::zeros(shape);

    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            dz_dy[[i, j]] = (dem[[i + 1, j]] - dem[[i - 1, j]]) / (2.0 * cell_size);
            dz_dx[[i, j]] = (dem[[i, j + 1]] - dem[[i, j - 1]]) / (2.0 * cell_size);
        }
    }
    (dz_dy, dz_dx)
}

/// Calculate slope from DEM in degrees or radians.
pub fn calculate_slope(dem: &Array2<f32>, cell_size: f32, degrees: bool) -> Array2<f32> {
    let (dz_dy, dz_dx) = calculate_gradient(dem, cell_size);
    let mut slope = Array2::<f32>::zeros(dem.raw_dim());
    for i in 0..dem.nrows() {
        for j in 0..dem.ncols() {
            let rad = (dz_dx[[i, j]].powi(2) + dz_dy[[i, j]].powi(2)).sqrt().atan();
            slope[[i, j]] = if degrees { rad.to_degrees() } else { rad };
        }
    }
    slope
}

/// Calculate aspect in degrees [0, 360]. North = 0, East = 90, South = 180, West = 270.
pub fn calculate_aspect(dem: &Array2<f32>, cell_size: f32) -> Array2<f32> {
    let (dz_dy, dz_dx) = calculate_gradient(dem, cell_size);
    let mut aspect = Array2::<f32>::zeros(dem.raw_dim());
    for i in 0..dem.nrows() {
        for j in 0..dem.ncols() {
            let mut a = (-dz_dx[[i, j]]).atan2(dz_dy[[i, j]]).to_degrees();
            if a < 0.0 {
                a += 360.0;
            }
            aspect[[i, j]] = a;
        }
    }
    aspect
}

/// Compute terrain curvature (profile, plan, total, tangential).
pub fn calculate_curvature(dem: &Array2<f32>, cell_size: f32, method: &str) -> Array2<f32> {
    let (dz_dy, dz_dx) = calculate_gradient(dem, cell_size);
    let shape = dem.raw_dim();
    let mut d2z_dx2 = Array2::<f32>::zeros(shape);
    let mut d2z_dy2 = Array2::<f32>::zeros(shape);
    let mut d2z_dxdy = Array2::<f32>::zeros(shape);

    for i in 1..shape[0] - 1 {
        for j in 1..shape[1] - 1 {
            d2z_dx2[[i, j]] = (dz_dx[[i, j + 1]] - dz_dx[[i, j - 1]]) / (2.0 * cell_size);
            d2z_dy2[[i, j]] = (dz_dy[[i + 1, j]] - dz_dy[[i - 1, j]]) / (2.0 * cell_size);
            d2z_dxdy[[i, j]] = (dz_dx[[i + 1, j]] - dz_dx[[i - 1, j]]) / (2.0 * cell_size);
        }
    }

    let mut result = Array2::<f32>::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let p = dz_dx[[i, j]];
            let q = dz_dy[[i, j]];
            let p2 = p * p;
            let q2 = q * q;
            let denom = match method {
                "profile" => (p2 + q2) * (1.0 + p2 + q2).powf(1.5),
                "plan" => (p2 + q2).powf(1.5),
                "total" => 1.0,
                "tangential" => (p2 + q2) * (1.0 + p2 + q2).sqrt(),
                _ => 1.0,
            };
            let denom = if denom.abs() < f32::EPSILON { 1.0 } else { denom };
            let num = match method {
                "profile" => p2 * d2z_dx2[[i, j]] + 2.0 * p * q * d2z_dxdy[[i, j]] + q2 * d2z_dy2[[i, j]],
                "plan" => q2 * d2z_dx2[[i, j]] - 2.0 * p * q * d2z_dxdy[[i, j]] + p2 * d2z_dy2[[i, j]],
                "total" => d2z_dx2[[i, j]] + d2z_dy2[[i, j]],
                "tangential" => q2 * d2z_dx2[[i, j]] - 2.0 * p * q * d2z_dxdy[[i, j]] + p2 * d2z_dy2[[i, j]],
                _ => 0.0,
            };
            result[[i, j]] = if method == "profile" { -num / denom } else { num / denom };
        }
    }
    result
}
