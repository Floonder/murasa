import numpy as np
from murasa.core.base import ParameterPlugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


class ProximityPlugin(ParameterPlugin):
    """
    Proximity to features (rivers, faults, etc.)

    Closer to feature = Higher risk
    """

    def __init__(self, name: str = "proximity", weight: float = 0.10,
                 feature_name: str = "river", max_distance: float = 500):
        super().__init__(name, weight)
        self.feature_name = feature_name
        self.max_distance = max_distance

    def validate_requirements(self, registry):
        return registry.has(self.feature_name)

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        feature_source = registry.get(self.feature_name)
        if feature_source.data.empty:
            logger.warning(f"      No {self.feature_name} features found")
            return ctx.empty_grid()

        self.result = ctx.distance_to_features(
            feature_source,
            max_distance=self.max_distance
        )

        self.log_result()
        return self.result
