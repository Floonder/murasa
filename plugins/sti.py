import numpy as np
from scipy.ndimage import gaussian_filter

from murasa.core.base import RasterParameterPlugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger
from murasa.core.terrain import calculate_slope, calculate_sti


class STIPlugin(RasterParameterPlugin):
    """
    Sediment Transport Index (STI) Plugin (Moore & Burch 1986).

    Models overland sediment carrying capacity and channel scour susceptibility:
    STI = (A_s / 22.13)^0.6 * (sin(beta) / 0.0896)^1.3
    Characterizes high-velocity sediment wash and erosion during flash flood events.
    """
    source_keys = ['sti']
    inverse = False

    def validate_requirements(self, registry):
        return registry.has('sti') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('sti'):
            sti = ctx.resample_raster(registry.get('sti'))
        else:
            logger.info("      Computing STI from DEM/components...")

            if registry.has('dem'):
                from murasa.core.hydro import fill_sinks

                dem = ctx.resample_raster(registry.get('dem'))
                dem = fill_sinks(dem)

                slope_rad = calculate_slope(dem, ctx.resolution, degrees=False)
                slope_rad = np.maximum(slope_rad, 0.001)

                flow_accum = 1.0 / (slope_rad + 0.001)
                flow_accum = gaussian_filter(flow_accum, sigma=2)

                sti = calculate_sti(flow_accum, slope_rad, ctx.resolution, log_scale=True)
            else:
                logger.error("      Cannot compute STI: No DEM found")
                return ctx.empty_grid()

        valid = np.isfinite(sti)
        if not valid.any():
            return ctx.empty_grid(fill=0.5)

        normalized = self._normalize(sti)
        if self.inverse:
            normalized = 1.0 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
