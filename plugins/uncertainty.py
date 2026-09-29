"""Monte Carlo Spatial Uncertainty Analysis for Multi-Criteria Decision Analysis (MCDA).

Quantifies how subjective uncertainty in criteria weights propagates into the
final spatial risk surface (Crosetto et al. 2000).

Research Grounding:
- Crosetto, M., Tarantola, S., & Saltelli, A. (2000). "Sensitivity and uncertainty
  analysis in spatial modelling based on GIS." Agriculture, Ecosystems & Environment, 81(1), 71-79.
- Chen, Y., Yu, J., & Khan, S. (2010). "Spatial sensitivity analysis of multi-criteria
  weights in GIS-based land suitability evaluation." Environmental Modelling & Software, 25(12), 1582-1591.
- Bentivoglio, R. et al. (2022). "Machine Learning for Generalizable Prediction
  of Flood Susceptibility." arXiv:1910.06521.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class UncertaintyResult:
    """Summary dataclass of spatial MCDA uncertainty propagation."""

    mean_risk: np.ndarray
    std_risk: np.ndarray
    coef_variation: np.ndarray
    percentile_05: np.ndarray
    percentile_95: np.ndarray
    n_simulations: int
    sampled_weights: np.ndarray
    criteria_names: List[str]


def sample_dirichlet_weights(
    base_weights: np.ndarray,
    n_samples: int = 100,
    concentration: float = 20.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Sample perturbed weight vectors from a Dirichlet distribution.

    Preserves sum(w) = 1 and non-negativity w_i >= 0. Higher concentration
    corresponds to smaller perturbations around the baseline weight vector.

    Args:
        base_weights: Normalized 1D array of criteria weights summing to 1.
        n_samples: Number of Monte Carlo draws.
        concentration: Scaling parameter alpha0 = concentration.
        random_state: Optional random seed.

    Returns:
        Array of shape (n_samples, len(base_weights)).
    """
    rng = np.random.default_rng(random_state)
    base_weights = np.asarray(base_weights, dtype=float)
    base_weights = base_weights / np.sum(base_weights)

    # Alpha parameters scaled by concentration
    alpha = np.maximum(base_weights * concentration, 0.01)
    samples = rng.dirichlet(alpha, size=n_samples)
    return samples


def compute_mcda_uncertainty(
    component_rasters: Dict[str, np.ndarray],
    weights: Dict[str, float],
    n_simulations: int = 100,
    concentration: float = 25.0,
    exclusion_mask: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
) -> UncertaintyResult:
    """
    Execute Monte Carlo weight perturbation over normalized risk components.

    Args:
        component_rasters: Dictionary mapping criterion name to 2D numpy array [0, 1].
        weights: Dictionary mapping criterion name to base weight.
        n_simulations: Number of Monte Carlo simulations (default 100).
        concentration: Dirichlet concentration parameter (lower = wider uncertainty).
        exclusion_mask: Boolean mask of excluded/water pixels (True = excluded).
        random_state: Random seed for reproducibility.

    Returns:
        UncertaintyResult containing mean, std, CV, and percentile surfaces.
    """
    names = list(weights.keys())
    base_w = np.array([weights[k] for k in names], dtype=float)
    base_w = base_w / np.sum(base_w)

    sampled_w = sample_dirichlet_weights(
        base_w,
        n_samples=n_simulations,
        concentration=concentration,
        random_state=random_state,
    )

    # Stack components into 3D array: (K, H, W)
    first_arr = next(iter(component_rasters.values()))
    h, w = first_arr.shape
    stack = np.empty((len(names), h, w), dtype=np.float32)

    for i, name in enumerate(names):
        arr = component_rasters[name]
        stack[i] = np.nan_to_num(arr, nan=0.0).astype(np.float32)

    # Monte Carlo simulation runs: (n_simulations, H, W)
    # simulated_risk[m] = sum_k sampled_w[m, k] * stack[k]
    simulated_risks = np.tensordot(sampled_w, stack, axes=(1, 0))

    if exclusion_mask is not None:
        simulated_risks[:, exclusion_mask] = 0.0

    mean_risk = np.mean(simulated_risks, axis=0)
    std_risk = np.std(simulated_risks, axis=0)
    coef_variation = std_risk / (mean_risk + 1e-4)

    p05 = np.percentile(simulated_risks, 5, axis=0)
    p95 = np.percentile(simulated_risks, 95, axis=0)

    return UncertaintyResult(
        mean_risk=mean_risk.astype(np.float32),
        std_risk=std_risk.astype(np.float32),
        coef_variation=coef_variation.astype(np.float32),
        percentile_05=p05.astype(np.float32),
        percentile_95=p95.astype(np.float32),
        n_simulations=n_simulations,
        sampled_weights=sampled_w,
        criteria_names=names,
    )
