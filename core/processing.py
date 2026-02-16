import numpy as np
from scipy.ndimage import distance_transform_edt

from .data_source import DataSource
from .resampler import RasterResampler, VectorResampler


class ProcessingContext:
    """
    Usage in a plugin:
        ctx = ProcessingContext(grid_shape, transform, crs)
        dem = ctx.resample_raster(registry.get('dem'))
        risk = ctx.distance_to_features(registry.get('river'))
    """

    def __init__(self, grid_shape, transform, crs):
        self.grid_shape = grid_shape
        self.transform = transform
        self.crs = crs
        self.resolution = abs(transform.a)

    def resample_raster(self, source: DataSource,
                        method: str = "bilinear") -> np.ndarray:
        """
        Resample a raster DataSource to the analysis grid.

        Args:
            source: DataSource with raster data
            method: Resampling method (bilinear, nearest, cubic, etc.)

        Returns:
            Resampled array matching grid_shape
        """
        resampler = RasterResampler(
            self.grid_shape, self.transform, self.crs, method=method
        )
        return resampler.resample(source)

    def rasterize_vector(self, source: DataSource,
                         column: str = None,
                         all_touched: bool = True,
                         fill_value: float = 0.0) -> np.ndarray:
        """
        Rasterize a vector DataSource to the analysis grid.

        Args:
            source: DataSource with vector data
            column: Column name to use for burn-in values
            all_touched: Whether to rasterize all touched pixels
            fill_value: Fill value for empty pixels

        Returns:
            Rasterized array matching grid_shape
        """
        resampler = VectorResampler(
            self.grid_shape, self.transform, self.crs
        )
        return resampler.resample(
            source,
            value_column=column,
            all_touched=all_touched,
            fill_value=fill_value
        )

    def empty_grid(self, fill: float = 0.0,
                   dtype=np.float32) -> np.ndarray:
        return np.full(self.grid_shape, fill, dtype=dtype)

    def boolean_grid(self, fill: bool = False) -> np.ndarray:
        return np.full(self.grid_shape, fill, dtype=bool)

    def distance_to_features(self, source: DataSource,
                             max_distance: float = None) -> np.ndarray:
        """
        Compute distance from each pixel to the nearest vector feature.

        Args:
            source: DataSource with vector geometry
            max_distance: If set, distances beyond this are clipped.
                          The result is also inverted: close=1, far=0.

        Returns:
            Distance raster in CRS units (or normalized 0-1 if max_distance set)
        """
        feature_mask = self.rasterize_vector(source).astype(bool)
        dist_pixels = distance_transform_edt(~feature_mask)
        dist_units = dist_pixels * self.resolution

        if max_distance is not None:
            return np.clip(1 - (dist_units / max_distance), 0, 1).astype(np.float32)

        return dist_units.astype(np.float32)
