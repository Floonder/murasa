import numpy as np
from typing import Optional, Dict

from murasa.core.base import ParameterPlugin
from murasa.core.data_source import DataRegistry
from murasa.core.resampler import VectorResampler
from murasa.core.processing import ProcessingContext
from murasa.core.logger import logger


class RainfallPlugin(ParameterPlugin):
    """
    Rainfall intensity risk factor

    Higher rainfall = Higher flood risk
    """

    def __init__(self, name: str = "rainfall", weight: float = 0.15,
                 column_name: str = "gridcode",
                 value_mapping: Optional[Dict[int, float]] = None):
        super().__init__(name, weight)
        self.column_name = column_name
        self.value_mapping = value_mapping

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return registry.has('rainfall')

    def process(self, registry, grid_shape, transform, crs):
        self.log_processing()

        rain_source = registry.get('rainfall')
        gdf_rain = rain_source.data.to_crs(crs)

        col = self.column_name
        if col not in gdf_rain.columns:
            alternatives = ['intensity', 'value', 'curah_hujan', 'rainfall']
            col = next((c for c in alternatives if c in gdf_rain.columns), None)
            if not col:
                logger.warning("No rainfall value column found, using default")
                return np.full(grid_shape, 0.5, dtype=np.float32)

        if self.value_mapping:
            gdf_rain = gdf_rain.copy()
            gdf_rain['mapped_value'] = gdf_rain[col].map(self.value_mapping)

            unmapped = gdf_rain['mapped_value'].isna()
            if unmapped.any():
                logger.warning(f"      Some rainfall values not mapped. using raw value * 500 approximation.")
                gdf_rain.loc[unmapped, 'mapped_value'] = gdf_rain.loc[unmapped, col] * 500

            value_col = 'mapped_value'
        else:
            value_col = col

        logger.info(f"      Rasterizing {len(gdf_rain)} rainfall polygons. CRS: {gdf_rain.crs}")

        rain_source.data = gdf_rain
        rain_source.data = gdf_rain
        ctx = ProcessingContext(grid_shape, transform, crs)
        rainfall_raster = ctx.rasterize_vector(rain_source, column=value_col)

        logger.info(f"      Non-zero pixels: {np.count_nonzero(rainfall_raster)}")

        valid = rainfall_raster > 0
        if valid.any():
            unique_vals = np.unique(rainfall_raster[valid])
            if len(unique_vals) <= 1:
                normalized = np.where(valid, 0.5, 0.0).astype(np.float32)
            else:
                rank_map = {v: (i + 1) / len(unique_vals)
                            for i, v in enumerate(sorted(unique_vals))}
                normalized = np.zeros(grid_shape, dtype=np.float32)
                for val, rank in rank_map.items():
                    normalized[rainfall_raster == val] = rank
                logger.info(f"      Rank mapping: {dict((int(k), round(v, 3)) for k, v in rank_map.items())}")
        else:
            normalized = np.full(grid_shape, 0.5, dtype=np.float32)

        self.result = normalized
        self.log_result()
        return normalized
