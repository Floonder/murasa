use ndarray::{Array2, Array};

/// Percentile-based normalization (robust to outliers).
pub fn percentile(data: &Array2<f32>, lower: f64, upper: f64) -> Array2<f32> {
    // Placeholder: real impl would compute percentiles over valid data.
    let mut out = data.clone();
    out.mapv_inplace(|v| {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    });
    out
}

/// Standard min-max normalization.
pub fn minmax(data: &Array2<f32>) -> Array2<f32> {
    let valid: Vec<f32> = data.iter().filter(|&&v| v.is_finite()).copied().collect();
    if valid.is_empty() {
        return Array::zeros(data.raw_dim());
    }
    let d_min = valid.iter().copied().fold(f32::INFINITY, f32::min);
    let d_max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (d_max - d_min).abs() < f32::EPSILON {
        return Array::from_elem(data.raw_dim(), 0.5f32);
    }
    data.mapv(|v| {
        if v.is_finite() {
            ((v - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

/// Rank-based normalization.
pub fn rank(data: &Array2<f32>) -> Array2<f32> {
    // Placeholder: real impl would argsort.
    minmax(data)
}

/// Z-score normalization, clipped to [-clip_sigma, +clip_sigma] then scaled to 0..1.
pub fn zscore(data: &Array2<f32>, clip_sigma: f32) -> Array2<f32> {
    let valid: Vec<f32> = data.iter().filter(|&&v| v.is_finite()).copied().collect();
    if valid.is_empty() {
        return Array::zeros(data.raw_dim());
    }
    let mean = valid.iter().copied().sum::<f32>() / valid.len() as f32;
    let variance = valid.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / valid.len() as f32;
    let std = variance.sqrt();
    if std < f32::EPSILON {
        return Array::from_elem(data.raw_dim(), 0.5f32);
    }
    data.mapv(|v| {
        if v.is_finite() {
            let z = ((v - mean) / std).clamp(-clip_sigma, clip_sigma);
            (z + clip_sigma) / (2.0 * clip_sigma)
        } else {
            0.0
        }
    })
}

/// Invert values (1 - data).
pub fn invert(data: &Array2<f32>) -> Array2<f32> {
    data.mapv(|v| 1.0 - v)
}

/// Classify into n_classes using quantile breaks.
pub fn classify_quantile(data: &Array2<f32>, n_classes: usize) -> (Array2<u8>, Vec<f64>) {
    let mut classified = Array2::<u8>::zeros(data.raw_dim());
    let valid: Vec<f32> = data.iter().filter(|&&v| v.is_finite()).copied().collect();
    if valid.is_empty() {
        return (classified, Vec::new());
    }
    // Placeholder breaks
    let breaks: Vec<f64> = (0..=n_classes)
        .map(|i| i as f64 / n_classes as f64)
        .collect();
    for ((i, j), &v) in data.indexed_iter() {
        if !v.is_finite() {
            continue;
        }
        for class in 0..n_classes {
            let lower = breaks[class] as f32;
            let upper = if class + 1 < breaks.len() {
                breaks[class + 1] as f32
            } else {
                f32::INFINITY
            };
            if v >= lower && (class == n_classes - 1 || v < upper) {
                classified[[i, j]] = (class + 1) as u8;
                break;
            }
        }
    }
    (classified, breaks)
}

/// Classify using Jenks Natural Breaks (placeholder).
pub fn classify_jenks(data: &Array2<f32>, n_classes: usize, _sample_size: usize) -> (Array2<u8>, Vec<f64>) {
    // Fallback to quantile.
    classify_quantile(data, n_classes)
}
