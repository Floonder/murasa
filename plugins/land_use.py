import numpy as np
from typing import Optional, Dict

from murasa.core.base import ParameterPlugin
from murasa.core.data_source import DataRegistry
from murasa.core.resampler import VectorResampler
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


class LandUsePlugin(ParameterPlugin):
    """
    Land use runoff coefficient

    Different land covers have different runoff characteristics:
    - Water/Impervious = 1.0 (100% runoff)
    - Forest = 0.2 (80% infiltration)
    """

    def __init__(self, name: str = "land_use", weight: float = 0.20,
                 coefficients: Optional[Dict[str, float]] = None,
                 priorities: Optional[Dict[str, int]] = None):
        super().__init__(name, weight)
        self.coefficients = coefficients or {
            'water': 1.0,
            'settlement': 0.85,
            'building': 0.90,
            'road': 0.90,
            'paddy': 0.70,
            'plantation': 0.40,
            'forest': 0.20,
            'shrub': 0.35,
            'default': 0.50
        }
        self.priorities = priorities or {
            'water': 100,
            'building': 90,
            'road': 80,
            'settlement': 70,
            'paddy': 50,
            'plantation': 40,
            'shrub': 30,
            'forest': 20,
            'default': 0
        }

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return any(k.startswith('landuse_') for k in registry.list_sources())

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        landuse_raster = np.full(grid_shape, self.coefficients['default'],
                                 dtype=np.float32)

        landuse_sources = [k for k in registry.list_sources()
                           if k.startswith('landuse_')]

        if not landuse_sources:
            logger.warning("      No landuse data found, using default")
            return landuse_raster

        priority_order = sorted(
            landuse_sources,
            key=lambda x: self.priorities.get(
                x.replace('landuse_', ''), 0
            )
        )

        logger.info(f"      Rasterizing {len(priority_order)} landuse layers...")

        ctx = ProcessingContext(grid_shape, transform, crs)

        for name in priority_order:
            source = registry.get(name)
            lu_type = name.replace('landuse_', '')
            coeff = self.coefficients.get(lu_type, 0.5)

            if source.data.empty:
                continue

            mask = ctx.rasterize_vector(source).astype(bool)

            landuse_raster[mask] = coeff
            logger.debug(f"         {lu_type}: {mask.sum()} pixels")

        self.result = landuse_raster
        self.log_result()
        return landuse_raster
