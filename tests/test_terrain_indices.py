"""Unit and integration tests for Terrain and Hydrologic Risk Indices (TPI, SPI, STI)."""

import numpy as np
import pytest
from affine import Affine

from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.terrain import (
    calculate_slope,
    calculate_spi,
    calculate_sti,
    calculate_tpi,
)
from murasa.plugins.spi import SPIPlugin
from murasa.plugins.sti import STIPlugin
from murasa.plugins.tpi import TPIPlugin


@pytest.fixture
def synthetic_valley_ridge_dem():
    """Create a 30x30 synthetic DEM with a central valley and lateral ridges."""
    x = np.linspace(-10, 10, 30)
    y = np.linspace(-10, 10, 30)
    xx, yy = np.meshgrid(x, y)
    # Parabolic valley along x=0, rising outwards
    dem = (xx**2) + (0.5 * yy) + 100.0
    return dem.astype(np.float32)


def test_calculate_tpi_flat_and_valley(synthetic_valley_ridge_dem):
    dem = synthetic_valley_ridge_dem

    # Flat plane should have TPI approximately zero
    flat_dem = np.full((20, 20), 50.0, dtype=np.float32)
    tpi_flat = calculate_tpi(flat_dem, cell_size=10.0, radius_cells=2)
    assert np.allclose(tpi_flat, 0.0, atol=1e-5)

    # Valley center (x=0) should have negative TPI, ridge edges positive TPI
    tpi_valley = calculate_tpi(dem, cell_size=10.0, radius_cells=3)
    # Central columns should be negative (valley/depression)
    assert np.all(tpi_valley[5:25, 14:16] < 0)
    # Outer columns should be positive (ridges)
    assert np.all(tpi_valley[5:25, 0:3] > 0)


def test_calculate_spi():
    slope_rad = np.array([[0.05, 0.1], [0.2, 0.3]])
    flow_accum = np.array([[10.0, 50.0], [100.0, 500.0]])
    cell_size = 10.0

    spi_raw = calculate_spi(flow_accum, slope_rad, cell_size=cell_size, log_scale=False)
    assert spi_raw.shape == (2, 2)
    # SPI increases monotonically with flow accumulation and slope
    assert spi_raw[0, 0] < spi_raw[0, 1]
    assert spi_raw[0, 1] < spi_raw[1, 0]
    assert spi_raw[1, 0] < spi_raw[1, 1]

    # Log scale output is finite
    spi_log = calculate_spi(flow_accum, slope_rad, cell_size=cell_size, log_scale=True)
    assert np.all(np.isfinite(spi_log))


def test_calculate_sti():
    slope_rad = np.array([[0.05, 0.1], [0.2, 0.3]])
    flow_accum = np.array([[10.0, 50.0], [100.0, 500.0]])
    cell_size = 10.0

    sti_raw = calculate_sti(flow_accum, slope_rad, cell_size=cell_size, log_scale=False)
    assert sti_raw.shape == (2, 2)
    assert sti_raw[0, 0] < sti_raw[1, 1]

    sti_log = calculate_sti(flow_accum, slope_rad, cell_size=cell_size, log_scale=True)
    assert np.all(np.isfinite(sti_log))


def test_tpi_plugin_process(synthetic_valley_ridge_dem):
    dem = synthetic_valley_ridge_dem
    registry = DataRegistry()
    registry.register("dem", DataSource("dem", SourceType.RASTER, data=dem, crs="EPSG:32648"))

    plugin = TPIPlugin(weight=0.2, inverse=True)
    assert plugin.validate_requirements(registry) is True

    transform = Affine.translation(100000, 9200000) @ Affine.scale(10.0, -10.0)
    result = plugin.process(registry, grid_shape=dem.shape, transform=transform, crs="EPSG:32648")

    assert result.shape == dem.shape
    assert np.all(np.isfinite(result))
    # Normalized between 0 and 1
    assert 0.0 <= np.min(result) <= np.max(result) <= 1.0


def test_spi_plugin_process(synthetic_valley_ridge_dem):
    dem = synthetic_valley_ridge_dem
    registry = DataRegistry()
    registry.register("dem", DataSource("dem", SourceType.RASTER, data=dem, crs="EPSG:32648"))

    plugin = SPIPlugin(weight=0.15)
    assert plugin.validate_requirements(registry) is True

    transform = Affine.translation(100000, 9200000) @ Affine.scale(10.0, -10.0)
    result = plugin.process(registry, grid_shape=dem.shape, transform=transform, crs="EPSG:32648")

    assert result.shape == dem.shape
    assert np.all(np.isfinite(result))
    assert 0.0 <= np.min(result) <= np.max(result) <= 1.0


def test_sti_plugin_process(synthetic_valley_ridge_dem):
    dem = synthetic_valley_ridge_dem
    registry = DataRegistry()
    registry.register("dem", DataSource("dem", SourceType.RASTER, data=dem, crs="EPSG:32648"))

    plugin = STIPlugin(weight=0.15)
    assert plugin.validate_requirements(registry) is True

    transform = Affine.translation(100000, 9200000) @ Affine.scale(10.0, -10.0)
    result = plugin.process(registry, grid_shape=dem.shape, transform=transform, crs="EPSG:32648")

    assert result.shape == dem.shape
    assert np.all(np.isfinite(result))
    assert 0.0 <= np.min(result) <= np.max(result) <= 1.0
