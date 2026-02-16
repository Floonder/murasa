import numpy as np
from scipy.ndimage import gaussian_filter

from murasa.core.base import RasterParameterPlugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


class TWIPlugin(RasterParameterPlugin):
    """
    Topographic Wetness Index
    """
    source_keys = ['twi']
    inverse = False

    def validate_requirements(self, registry):
        return registry.has('twi') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('twi'):
            twi = ctx.resample_raster(registry.get('twi'))
        else:
            logger.info("      Computing TWI from components...")

            if registry.has('slope'):
                slope_source = registry.get('slope')
                if hasattr(slope_source, 'path') and slope_source.path:
                    slope = ctx.resample_raster(slope_source)
                elif hasattr(slope_source, 'data') and isinstance(slope_source.data, np.ndarray):
                    slope = slope_source.data
                else:
                    slope = None
            else:
                slope = None

            if registry.has('dem'):
                from murasa.core.hydro import fill_sinks

                dem = ctx.resample_raster(registry.get('dem'))

                dem = fill_sinks(dem)

                from murasa.core.terrain import calculate_slope

                slope_rad_calc = calculate_slope(dem, ctx.resolution, degrees=False)

                if slope is None:
                    slope_rad = slope_rad_calc
                else:
                    slope_rad = np.radians(slope)

                slope_rad = np.maximum(slope_rad, 0.001)

                flow_accum = 1.0 / (slope_rad + 0.001)
                flow_accum = gaussian_filter(flow_accum, sigma=2)

                resolution = ctx.resolution
                twi = np.log((flow_accum * resolution) / np.tan(slope_rad))
            else:
                logger.error("      Cannot compute TWI: No DEM found")
                return ctx.empty_grid()

        valid = np.isfinite(twi)
        if not valid.any():
            return ctx.empty_grid(fill=0.5)

        normalized = self._normalize(twi)

        if self.inverse:
            normalized = 1 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
