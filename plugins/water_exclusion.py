import numpy as np

from murasa.core.base import ParameterPlugin
from murasa.core.data_source import DataRegistry
from murasa.core.resampler import VectorResampler
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


class WaterExclusionPlugin(ParameterPlugin):
    """
    Excludes water bodies (rivers, reservoirs) from analysis.
    This plugin does NOT return a risk map, but an exclusion mask.
    """

    def __init__(self, name: str = "water_exclusion"):
        super().__init__(name, weight=0.0)
        self.is_exclusion = True

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return True

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        exclusion_mask = np.zeros(grid_shape, dtype=bool)
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('river'):
            river_source = registry.get('river')

            if not river_source.data.empty:
                river_mask = ctx.rasterize_vector(river_source).astype(bool)

                exclusion_mask |= river_mask
                logger.info(f"      Excluded Rivers: {river_mask.sum()} pixels")

        if registry.has('reservoir'):
            reservoir_source = registry.get('reservoir')

            if not reservoir_source.data.empty:
                res_mask = ctx.rasterize_vector(reservoir_source).astype(bool)

                exclusion_mask |= res_mask
                logger.info(f"      Excluded Reservoirs: {res_mask.sum()} pixels")

        self.result = exclusion_mask.astype(np.float32)
        return self.result
