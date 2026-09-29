"""Unit and integration tests for Monte Carlo spatial MCDA uncertainty propagation."""

import numpy as np
import pytest
from affine import Affine

from murasa.core.config import EngineConfig
from murasa.core.data_source import DataSource, SourceType
from murasa.engine.risk_engine import RiskEngine
from murasa.plugins.spi import SPIPlugin
from murasa.plugins.tpi import TPIPlugin
from murasa.plugins.uncertainty import (
    UncertaintyResult,
    compute_mcda_uncertainty,
    sample_dirichlet_weights,
)


def test_sample_dirichlet_weights():
    base_w = np.array([0.5, 0.3, 0.2])
    n = 200
    samples = sample_dirichlet_weights(base_w, n_samples=n, concentration=50.0, random_state=42)

    assert samples.shape == (n, 3)
    # Weights must be non-negative
    assert np.all(samples >= 0.0)
    # Weights must sum to 1.0
    assert np.allclose(samples.sum(axis=1), 1.0)
    # Sample mean must match base weights closely
    assert np.allclose(samples.mean(axis=0), base_w, atol=0.03)


def test_compute_mcda_uncertainty():
    h, w = 20, 20
    comp1 = np.full((h, w), 0.8, dtype=np.float32)
    comp2 = np.full((h, w), 0.2, dtype=np.float32)

    rasters = {"elevation": comp1, "slope": comp2}
    weights = {"elevation": 0.6, "slope": 0.4}

    res = compute_mcda_uncertainty(
        component_rasters=rasters,
        weights=weights,
        n_simulations=100,
        concentration=30.0,
        random_state=42,
    )

    assert isinstance(res, UncertaintyResult)
    assert res.mean_risk.shape == (h, w)
    assert res.std_risk.shape == (h, w)
    assert res.coef_variation.shape == (h, w)
    assert res.percentile_05.shape == (h, w)
    assert res.percentile_95.shape == (h, w)
    assert res.n_simulations == 100

    # Theoretical base risk = 0.8 * 0.6 + 0.2 * 0.4 = 0.48 + 0.08 = 0.56
    assert np.allclose(res.mean_risk, 0.56, atol=0.03)
    # Standard deviation should be strictly positive
    assert np.all(res.std_risk > 0.0)
    # 5th percentile <= 95th percentile
    assert np.all(res.percentile_05 <= res.percentile_95)


def test_risk_engine_calculate_risk_with_uncertainty():
    config = EngineConfig()
    config.spatial.resolution = 10.0
    config.spatial.target_crs = "EPSG:32648"

    engine = RiskEngine(config)

    # Synthetic grid setup
    dem = np.random.RandomState(42).uniform(10, 100, (25, 25)).astype(np.float32)
    engine.register_data("dem", DataSource("dem", SourceType.RASTER, data=dem, crs="EPSG:32648"))

    engine.grid_shape = dem.shape
    engine.transform = Affine.translation(100000, 9200000) @ Affine.scale(10.0, -10.0)

    # Register multiple plugins
    engine.register_parameter(TPIPlugin(weight=0.5))
    engine.register_parameter(SPIPlugin(weight=0.5))

    uncertainty = engine.calculate_risk_with_uncertainty(n_simulations=50, random_state=42)

    assert isinstance(uncertainty, UncertaintyResult)
    assert uncertainty.mean_risk.shape == dem.shape
    assert uncertainty.std_risk.shape == dem.shape
    assert np.all(np.isfinite(uncertainty.mean_risk))
    assert np.all(np.isfinite(uncertainty.std_risk))
