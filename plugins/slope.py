import numpy as np
from murasa.core.base import RasterParameterPlugin
from murasa.core.processing import ProcessingContext, RasterResampler


class SlopePlugin(RasterParameterPlugin):
    """
    Slope-based risk

    inverse=True:  Flat areas = High risk (Flood - water accumulates)
    inverse=False: Steep areas = High risk (Landslide)
    """
    source_keys = ['slope']
    inverse = True

    def validate_requirements(self, registry):
        return registry.has('slope') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('slope'):
            slope = ctx.resample_raster(registry.get('slope'))
        else:
            dem = ctx.resample_raster(registry.get('dem'))

            from murasa.core.terrain import calculate_slope
            slope = calculate_slope(dem, ctx.resolution)

        normalized = self._normalize(slope)

        if self.inverse:
            normalized = 1 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
