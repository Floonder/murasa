from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from .logger import logger, log_success, log_error


class SourceType(Enum):
    """Supported data source types"""
    RASTER = "raster"
    VECTOR = "vector"
    POINT = "point"
    TABLE = "table"
    API = "api"
    POINT_CLOUD = "point_cloud"
    NETCDF = "netcdf"


@dataclass
class DataSource:
    """
    Generic data source container with lazy loading

    Supports:
    - Raster (GeoTIFF, SDAT, etc.)
    - Vector (Shapefile, GeoJSON, GeoPackage)
    - Tabular (CSV, Excel)
    - Point clouds
    """
    name: str
    source_type: SourceType
    path: Optional[Union[str, Path]] = None
    data: Any = None
    crs: str = "EPSG:4326"
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def load(self) -> 'DataSource':
        if self.data is not None:
            logger.debug(f"Data already loaded: {self.name}")
            return self

        if not self.path or not self.path.exists():
            raise FileNotFoundError(f"Data source not found: {self.path}")

        try:
            if self.source_type == SourceType.RASTER:
                self._load_raster()
            elif self.source_type == SourceType.VECTOR:
                self._load_vector()
            elif self.source_type == SourceType.POINT:
                self._load_vector()
            elif self.source_type == SourceType.TABLE:
                self._load_table()

            log_success(f"Loaded: {self.name} ({self.source_type.value})")
        except Exception as e:
            log_error(f"Failed to load {self.name}: {e}")
            raise

        return self

    def _load_raster(self):
        with rasterio.open(self.path) as src:
            self.data = src.read(1)
            self.metadata['transform'] = src.transform
            self.metadata['profile'] = src.profile
            self.metadata['bounds'] = src.bounds
            self.metadata['shape'] = (src.height, src.width)
            self.crs = str(src.crs)

            logger.debug(f"  Shape: {self.metadata['shape']}")
            logger.debug(f"  CRS: {self.crs}")

    def _load_vector(self):
        self.data = gpd.read_file(self.path).to_crs(self.crs)

        invalid = ~self.data.geometry.is_valid
        if invalid.any():
            count = invalid.sum()
            logger.warning(f"  Repairing {count} invalid geometries")
            self.data.loc[invalid, 'geometry'] = \
                self.data.loc[invalid, 'geometry'].buffer(0)

        empty = self.data.geometry.is_empty
        if empty.any():
            count = empty.sum()
            logger.warning(f"  Removing {count} empty geometries")
            self.data = self.data[~empty]

        self.metadata['count'] = len(self.data)
        self.metadata['bounds'] = self.data.total_bounds

        logger.debug(f"  Features: {self.metadata['count']}")

    def _load_table(self):
        if self.path.suffix == '.csv':
            self.data = pd.read_csv(self.path)
        elif self.path.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.path)
        else:
            raise ValueError(f"Unsupported table format: {self.path.suffix}")

        self.metadata['rows'] = len(self.data)
        self.metadata['columns'] = list(self.data.columns)

        logger.debug(f"  Rows: {self.metadata['rows']}")
        logger.debug(f"  Columns: {len(self.metadata['columns'])}")

    def is_loaded(self) -> bool:
        return self.data is not None

    def unload(self) -> None:
        self.data = None
        logger.debug(f"Unloaded: {self.name}")

    def get_bounds(self) -> Optional[tuple]:
        if 'bounds' in self.metadata:
            return self.metadata['bounds']
        return None

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"DataSource(name='{self.name}', type={self.source_type.value}, {status})"


class DataRegistry:
    """
    Centralized registry for all data sources
    """

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}

    def register(self, name: str, source: DataSource) -> None:
        if name in self.sources:
            logger.warning(f"Overwriting existing source: {name}")

        self.sources[name] = source
        log_success(f"Registered: {name}")

    def register_from_path(self, name: str, path: Union[str, Path],
                           source_type: SourceType, crs: str = "EPSG:4326") -> None:
        source = DataSource(
            name=name,
            source_type=source_type,
            path=path,
            crs=crs
        )
        self.register(name, source)

    def get(self, name: str, auto_load: bool = True) -> DataSource:
        if name not in self.sources:
            raise KeyError(f"Data source '{name}' not registered. "
                           f"Available: {self.list_sources()}")

        source = self.sources[name]

        if auto_load and not source.is_loaded():
            source.load()

        return source

    def has(self, name: str) -> bool:
        return name in self.sources

    def list_sources(self) -> list[str]:
        return list(self.sources.keys())

    def list_by_type(self, source_type: SourceType) -> list[str]:
        return [name for name, src in self.sources.items()
                if src.source_type == source_type]

    def unload_all(self) -> None:
        for source in self.sources.values():
            source.unload()
        logger.info("All data sources unloaded")

    def get_memory_usage(self) -> Dict[str, int]:
        usage = {}
        for name, source in self.sources.items():
            if source.is_loaded():
                if isinstance(source.data, np.ndarray):
                    usage[name] = source.data.nbytes
                elif isinstance(source.data, (gpd.GeoDataFrame, pd.DataFrame)):
                    usage[name] = source.data.memory_usage(deep=True).sum()
        return usage

    def print_summary(self) -> None:
        logger.info("\n" + "=" * 70)
        logger.info("DATA REGISTRY SUMMARY")
        logger.info("=" * 70)

        for name, source in self.sources.items():
            status = "Loaded" if source.is_loaded() else "Not loaded"
            logger.info(f"{status} | {source.source_type.value:8s} | {name}")

        logger.info("=" * 70 + "\n")

    def __len__(self) -> int:
        return len(self.sources)

    def __contains__(self, name: str) -> bool:
        return name in self.sources

    def register_vector(self, name: str, gdf: gpd.GeoDataFrame,
                        crs: str = None, **metadata) -> DataSource:
        """
        Shortcut: register a GeoDataFrame as a vector source.

        Args:
            name: Registry key
            gdf: GeoDataFrame to register
            crs: CRS string (defaults to gdf.crs)
            **metadata: Additional metadata fields

        Returns:
            Created DataSource
        """
        source = DataSource(
            name=name,
            source_type=SourceType.VECTOR,
            data=gdf,
            crs=crs or str(gdf.crs),
            metadata=metadata
        )
        self.register(name, source)
        return source

    def register_raster_array(self, name: str, array: np.ndarray,
                              crs: str = None, **metadata) -> DataSource:
        """
        Shortcut: register a numpy array as a raster source.

        Args:
            name: Registry key
            array: Numpy array to register
            crs: CRS string
            **metadata: Additional metadata fields

        Returns:
            Created DataSource
        """
        source = DataSource(
            name=name,
            source_type=SourceType.RASTER,
            data=array,
            crs=crs,
            metadata=metadata
        )
        self.register(name, source)
        return source


def load_vector_from_multiple_dirs(
        directories: list[Path],
        filename: str,
        target_crs: str = "EPSG:4326",
        validate_geometries: bool = True
) -> gpd.GeoDataFrame:
    """
    Generic helper to load and merge vector data from multiple directories
    
    This is a reusable utility for loader plugins that need to scan
    multiple directories for the same filename pattern.
    
    Args:
        directories: List of directories to search
        filename: Name of the vector file to load
        target_crs: Target coordinate reference system
        validate_geometries: Whether to repair invalid geometries
        
    Returns:
        Merged GeoDataFrame (empty if no files found)
    """
    frames = []

    for directory in directories:
        path = Path(directory) / filename
        if path.exists():
            try:
                gdf = gpd.read_file(path).to_crs(target_crs)

                if validate_geometries:
                    invalid = ~gdf.geometry.is_valid
                    if invalid.any():
                        count = invalid.sum()
                        logger.debug(f"  Repairing {count} invalid geometries in {path.name}")
                        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].buffer(0)

                    empty = gdf.geometry.is_empty
                    if empty.any():
                        count = empty.sum()
                        logger.debug(f"  Removing {count} empty geometries from {path.name}")
                        gdf = gdf[~empty]

                frames.append(gdf)
                logger.debug(f"  Loaded: {path.name} ({len(gdf)} features)")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)
    return merged
