import numpy as np
from scipy.ndimage import gaussian_filter

from murasa.core.base import RasterParameterPlugin
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger
from murasa.core.terrain import calculate_slope, calculate_spi


class SPIPlugin(RasterParameterPlugin):
    """
    Stream Power Index (SPI) Plugin (Moore et al. 1991).

    Quantifies the potential erosive energy of overland flow and flood scouring:
    SPI = A_s * tan(beta), where A_s is specific catchment area.
    Higher values indicate greater potential for concentrated stream power and inundation scour.
    """
    source_keys = ['spi']
    inverse = False

    def validate_requirements(self, registry):
        return registry.has('spi') or registry.has('dem')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        if registry.has('spi'):
            spi = ctx.resample_raster(registry.get('spi'))
        else:
            logger.info("      Computing SPI from DEM/components...")

            if registry.has('dem'):
                from murasa.core.hydro import fill_sinks

                dem = ctx.resample_raster(registry.get('dem'))
                dem = fill_sinks(dem)

                slope_rad = calculate_slope(dem, ctx.resolution, degrees=False)
                slope_rad = np.maximum(slope_rad, 0.001)

                # Contributing area proxy from slope and spatial smoothing
                flow_accum = 1.0 / (slope_rad + 0.001)
                flow_accum = gaussian_filter(flow_accum, sigma=2)

                spi = calculate_spi(flow_accum, slope_rad, ctx.resolution, log_scale=True)
            else:
                logger.error("      Cannot compute SPI: No DEM found")
                return ctx.empty_grid()

        valid = np.isfinite(spi)
        if not valid.any():
            return ctx.empty_grid(fill=0.5)

        normalized = self._normalize(spi)
        if self.inverse:
            normalized = 1.0 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result
