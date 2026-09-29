import numpy as np

from murasa.core.base import RasterParameterPlugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger
from murasa.core.terrain import calculate_tpi


class TPIPlugin(RasterParameterPlugin):
    """
    Topographic Position Index (TPI) Plugin (Weiss 2001, Guisan et al. 1999).

    Measures relative topographic position:
    TPI = z_0 - mean(z_R).
    Negative TPI corresponds to valleys, ravines, and topographic depressions prone to flood pooling.
    Positive TPI corresponds to ridges, hills, and exposed crests.
    With inverse=True (default for flood risk), lower elevation positions receive higher susceptibility scores.
    """
    source_keys = ['tpi']
    inverse = True

    def __init__(self, weight: float = 1.0, inverse: bool = True, radius_cells: int = 3, **kwargs):
        super().__init__(weight=weight, inverse=inverse, **kwargs)
        self.radius_cells = radius_cells

    def validate_requirements(self, registry):
        return registry.has('tpi') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('tpi'):
            tpi = ctx.resample_raster(registry.get('tpi'))
        else:
            logger.info("      Computing TPI from DEM...")

            if registry.has('dem'):
                dem = ctx.resample_raster(registry.get('dem'))
                tpi = calculate_tpi(dem, ctx.resolution, radius_cells=self.radius_cells)
            else:
                logger.error("      Cannot compute TPI: No DEM found")
                return ctx.empty_grid()

        valid = np.isfinite(tpi)
        if not valid.any():
            return ctx.empty_grid(fill=0.5)

        normalized = self._normalize(tpi)
        if self.inverse:
            normalized = 1.0 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
