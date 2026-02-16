from abc import ABC, abstractmethod

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from murasa.core.data_source import DataSource


class BaseResampler(ABC):
    def __init__(self, grid_shape, transform, crs):
        self.grid_shape = grid_shape
        self.transform = transform
        self.crs = crs

    @abstractmethod
    def resample(self, source: DataSource) -> np.ndarray:
        pass

    def validate_output(self, result: np.ndarray) -> np.ndarray:
        pass

    def fill_nodata(self, data: np.ndarray, fill_value=0.0) -> np.ndarray:
        pass


class RasterResampler(BaseResampler):
    def __init__(self, grid_shape, transform, crs, method="bilinear"):
        super().__init__(grid_shape, transform, crs)
        self.method = method

    def _get_resampling_method(self):
        match self.method:
            case "bilinear":
                return Resampling.bilinear
            case "nearest":
                return Resampling.nearest
            case "cubic":
                return Resampling.cubic
            case "average":
                return Resampling.average
            case "mode":
                return Resampling.mode
            case "max":
                return Resampling.max
            case "min":
                return Resampling.min
            case "med":
                return Resampling.med
            case "q1":
                return Resampling.q1
            case "q3":
                return Resampling.q3
            case _:
                raise ValueError(f"Unknown resampling method: {self.method}")

    def resample(self, source: DataSource) -> np.ndarray:
        raster = np.zeros(self.grid_shape, dtype=np.float32)

        resampling = self._get_resampling_method()

        with rasterio.open(source.path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=raster,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=resampling
            )

        return raster


class VectorResampler(BaseResampler):
    def __init__(self, grid_shape, transform, crs):
        super().__init__(grid_shape, transform, crs)

    def resample(self, source: DataSource,
                 value_column: str = None,
                 all_touched: bool = True,
                 fill_value: float = 0.0) -> np.ndarray:

        gdf = source.data.to_crs(self.crs)
        if value_column and value_column in gdf.columns:
            shapes = [(geom, val) for geom, val in
                      zip(gdf.geometry, gdf[value_column])]
        else:
            shapes = [(geom, 1) for geom in gdf.geometry]
        result = rasterize(
            shapes=shapes,
            out_shape=self.grid_shape,
            transform=self.transform,
            fill=fill_value,
            dtype=np.float32,
            all_touched=all_touched
        )
        return result
