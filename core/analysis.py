import numpy as np
import rasterio
from geopandas import GeoDataFrame
from rasterstats import zonal_stats
from tempfile import NamedTemporaryFile
import os
from typing import List, Dict, Any, Optional

from murasa.core.logger import logger
from murasa.core.normalize import classify_jenks

from murasa.core.data_source import DataSource


def calculate_zonal_stats(
        vector_gdf: GeoDataFrame,
        raster_source: DataSource,
        stats: List[str] = ['mean', 'count'],
        nodata: float = np.nan,
        categorical: bool = False,
        transform=None,
        crs=None
) -> List[Dict[str, Any]]:
    """
    Calculate zonal statistics for vector features against a raster source.
    Handles temporary file creation/cleanup required by rasterstats.
    
    Args:
        vector_gdf: GeoDataFrame with zones
        raster_source: DataSource (raster) or numpy array
        stats: List of statistics to compute
        nodata: Nodata value
        categorical: Whether raster is categorical
        transform: Affine transform (required if raster_source is array)
        crs: CRS (required if raster_source is array)
        
    Returns:
        List of dicts containing stats
    """

    if hasattr(raster_source, 'data'):

        if raster_source.data is None:
            logger.warning(f"DataSource {getattr(raster_source, 'name', 'unknown')} has no data loaded.")
            return []

        data = np.asanyarray(raster_source.data)

        if hasattr(raster_source, 'metadata'):
            transform = transform or raster_source.metadata.get('transform')
            crs = crs or raster_source.crs
    elif isinstance(raster_source, np.ndarray):
        data = raster_source
    else:

        try:
            logger.debug(f"Converting raster source of type {type(raster_source)} to numpy array...")
            data = np.array(raster_source)
        except Exception as e:
            logger.error(f"Invalid raster source type: {type(raster_source)}. Error: {e}")
            return []

    if transform is None or crs is None:
        raise ValueError("Transform and CRS required for array-based zonal stats")

    with NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        try:

            with rasterio.open(
                    tmp.name, 'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs=crs,
                    transform=transform,
                    nodata=nodata
            ) as dst:
                dst.write(data, 1)

            results = zonal_stats(
                vector_gdf,
                tmp.name,
                stats=stats,
                categorical=categorical,
                nodata=nodata
            )
            return results

        except Exception as e:
            logger.error(f"Zonal stats failed: {e}")
            return []
        finally:

            if os.path.exists(tmp.name):
                try:
                    os.remove(tmp.name)
                except:
                    pass
