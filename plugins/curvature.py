import numpy as np
from scipy.ndimage import generic_filter

from murasa.core.base import ParameterPlugin
from murasa.core.data_source import DataRegistry
from murasa.core.resampler import RasterResampler
from murasa.core.logger import logger


class CurvaturePlugin(ParameterPlugin):
    """
    Terrain curvature risk factor
    
    Can use:
    1. Pre-computed curvature raster
    2. Compute from DEM on-the-fly
    
    For landslide:
    - Concave (negative) = HIGH risk (water accumulates)
    - Convex (positive) = LOW risk (water disperses)
    """

    def __init__(self, name: str = "curvature", weight: float = 0.10,
                 inverse: bool = True, curvature_type: str = "profile"):
        """
        Args:
            name: Plugin identifier
            weight: Weight in risk calculation
            inverse: If True, concave (negative) = high risk
            curvature_type: "profile", "plan", or "total"
        """
        super().__init__(name, weight)
        self.inverse = inverse
        self.curvature_type = curvature_type

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('curvature') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        resampler = RasterResampler(grid_shape, transform, crs)

        if registry.has('curvature'):
            curv_source = registry.get('curvature')
            curvature = resampler.resample(curv_source)
        else:
            logger.info("      Computing curvature from DEM...")
            curvature = self._compute_from_dem(registry, resampler, grid_shape, transform, crs)

        normalized = self._normalize_curvature(curvature)

        if self.inverse:
            normalized = normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result

    def _compute_from_dem(self, registry, resampler, grid_shape, transform, crs):
        dem_source = registry.get('dem')
        dem = resampler.resample(dem_source)
        cell_size = abs(transform[0])

        from murasa.core.terrain import calculate_curvature
        curvature = calculate_curvature(dem, cell_size, method=self.curvature_type)
        return curvature

    def _normalize_curvature(self, curvature):
        """
        Normalize curvature to 0-1 range
        
        Maps: negative (concave) will map to 1 (high risk)
              zero (flat) will map to 0.5
              positive (convex) will map to 0 (low risk)
        """
        p1, p99 = np.nanpercentile(curvature, [1, 99])
        curvature_clipped = np.clip(curvature, p1, p99)

        max_abs = max(abs(p1), abs(p99))
        if max_abs == 0:
            return np.full_like(curvature, 0.5)

        scaled = curvature_clipped / max_abs

        normalized = (1 - scaled) / 2

        return np.clip(normalized, 0, 1)


class SlopeAspectPlugin(ParameterPlugin):
    """
    Slope Aspect (direction) risk factor

    Certain slope orientations may be more susceptible.

    But this is just a what-if plugin for now.

    I still haven't had the good use case for this.
    """

    def __init__(self, name: str = "aspect", weight: float = 0.05,
                 high_risk_directions: list = None):
        super().__init__(name, weight)
        self.high_risk_directions = high_risk_directions or [0, 45, 315]

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('aspect') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        resampler = RasterResampler(grid_shape, transform, crs)

        if registry.has('aspect'):
            aspect_source = registry.get('aspect')
            aspect = resampler.resample(aspect_source, method="nearest")
        else:
            aspect = self._compute_aspect(registry, resampler)

        risk = self._aspect_to_risk(aspect)

        self.result = risk.astype(np.float32)
        self.log_result()
        return self.result

    def _compute_aspect(self, registry, resampler):
        dem_source = registry.get('dem')
        dem = resampler.resample(dem_source)

        from murasa.core.terrain import calculate_aspect

        cell_size = abs(resampler.transform[0])

        return calculate_aspect(dem, cell_size)

    def _aspect_to_risk(self, aspect):
        """Convert aspect to risk based on high-risk directions"""
        risk = np.zeros_like(aspect)

        for direction in self.high_risk_directions:
            diff = np.abs(aspect - direction)
            diff = np.minimum(diff, 360 - diff)

            contribution = np.exp(-(diff ** 2) / (2 * 30 ** 2))
            risk = np.maximum(risk, contribution)

        return risk
