from pathlib import Path
from typing import Optional, Dict
import numpy as np

from .risk_engine import RiskEngine
from murasa.core.config import EngineConfig
from murasa.core.data_source import DataSource, SourceType, load_admin_boundaries, load_rivers
from murasa.plugins.risk_factors import (
    RainfallPlugin, ElevationPlugin, SlopePlugin,
    TWIPlugin, ProximityPlugin, LandUsePlugin
)
from murasa.core.logger import logger, log_section


class FloodRiskEngine(RiskEngine):
    """
    Pre-configured engine for flood risk assessment

    The defaults are specific for my use case. Can also serve as an example.

    Administration boundaries as well.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        log_section("FLOOD RISK ENGINE INITIALIZED")

    def auto_configure(self,
                       dem_path: Optional[Path] = None,
                       slope_path: Optional[Path] = None,
                       twi_path: Optional[Path] = None,
                       rainfall_path: Optional[Path] = None,
                       river_path: Optional[Path] = None,
                       admin_dirs: Optional[list] = None,
                       landuse_paths: Optional[Dict[str, Path]] = None) -> None:
        """
        Auto-configure flood risk engine

        Args:
            dem_path: DEM raster path
            slope_path: Slope raster path (optional, computed from DEM if None)
            twi_path: TWI raster path (optional, computed from DEM if None)
            rainfall_path: Rainfall vector path
            river_path: River network path
            admin_dirs: List of admin boundary directories
            landuse_paths: Dict of {'type': path} for landuse layers
        """
        log_section("AUTO-CONFIGURING FLOOD ENGINE")

        if dem_path:
            self.register_data_from_path('dem', dem_path, SourceType.RASTER)

        if slope_path:
            self.register_data_from_path('slope', slope_path, SourceType.RASTER)

        if twi_path:
            self.register_data_from_path('twi', twi_path, SourceType.RASTER)

        if rainfall_path:
            self.register_data_from_path('rainfall', rainfall_path, SourceType.VECTOR)

        if river_path:
            self.register_data_from_path('river', river_path, SourceType.VECTOR)
        elif admin_dirs:
            try:
                gdf_rivers = load_rivers(admin_dirs, target_crs=self.crs)
                if not gdf_rivers.empty:
                    self.register_data('river', DataSource(
                        name='river',
                        source_type=SourceType.VECTOR,
                        data=gdf_rivers,
                        crs=self.crs
                    ))
            except Exception as e:
                logger.warning(f"Could not load rivers: {e}")

        if admin_dirs:
            try:
                gdf_admin = load_admin_boundaries(admin_dirs, target_crs=self.crs)

                gdf_admin = self._apply_spatial_filter(gdf_admin)

                self.register_data('admin', DataSource(
                    name='admin',
                    source_type=SourceType.VECTOR,
                    data=gdf_admin,
                    crs=self.crs
                ))
            except Exception as e:
                logger.warning(f"Could not load admin boundaries: {e}")

        if landuse_paths:
            for lu_type, lu_path in landuse_paths.items():
                self.register_data_from_path(
                    f'landuse_{lu_type}',
                    lu_path,
                    SourceType.VECTOR
                )

        self.register_parameter(TWIPlugin(weight=0.25))
        self.register_parameter(ElevationPlugin(weight=0.20, inverse=True))
        self.register_parameter(LandUsePlugin(weight=0.20))

        if rainfall_path:
            self.register_parameter(RainfallPlugin(weight=0.15))

        self.register_parameter(SlopePlugin(weight=0.10, inverse=True))

        if river_path or (admin_dirs and self.registry.has('river')):
            self.register_parameter(ProximityPlugin(
                weight=0.10,
                feature_name='river',
                max_distance=500
            ))

        logger.info("Flood risk engine configured")

    def _apply_spatial_filter(self, gdf: 'gpd.GeoDataFrame') -> 'gpd.GeoDataFrame':
        """Apply hierarchical spatial filtering"""
        filter_level = self.config.spatial.get_filter_level()

        if filter_level == 'full':
            logger.info(f"  Using full scope: {len(gdf)} units")
            return gdf

        col_map = {
            'district': 'WADMKC',
            'city': 'WADMKK',
            'province': 'WADMPR'
        }

        col_name = col_map.get(filter_level)
        if col_name not in gdf.columns:
            logger.warning(f"  Column '{col_name}' not found, using full scope")
            return gdf

        filter_values = self.config.spatial.get_active_filter()

        gdf_upper = gdf[col_name].str.upper()
        filter_upper = [v.upper() for v in filter_values]

        filtered = gdf[gdf_upper.isin(filter_upper)].copy()

        logger.info(f"  Filtered by {filter_level} ({filter_values}): "
                    f"{len(filtered)} units")

        return filtered


class LandslideRiskEngine(RiskEngine):
    """
    Pre-configured engine for landslide risk assessment

    Default parameters (INVERTED logic from flood):
    - Slope: 35% (steep = high risk)
    - Land Use: 25% (vegetation protects)
    - Elevation: 15% (high altitude)
    - Rainfall: 15% (trigger)
    - Geology: 10% (if available)

    This also servers as an example for now. I still haven't tested this.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        log_section("LANDSLIDE RISK ENGINE INITIALIZED")

    def auto_configure(self,
                       dem_path: Optional[Path] = None,
                       slope_path: Optional[Path] = None,
                       rainfall_path: Optional[Path] = None,
                       geology_path: Optional[Path] = None,
                       admin_dirs: Optional[list] = None,
                       landuse_paths: Optional[Dict[str, Path]] = None) -> None:

        log_section("AUTO-CONFIGURING LANDSLIDE ENGINE")

        if dem_path:
            self.register_data_from_path('dem', dem_path, SourceType.RASTER)

        if slope_path:
            self.register_data_from_path('slope', slope_path, SourceType.RASTER)

        if rainfall_path:
            self.register_data_from_path('rainfall', rainfall_path, SourceType.VECTOR)

        if geology_path:
            self.register_data_from_path('geology', geology_path, SourceType.VECTOR)

        if admin_dirs:
            try:
                gdf_admin = load_admin_boundaries(admin_dirs, target_crs=self.crs)
                self.register_data('admin', DataSource(
                    name='admin',
                    source_type=SourceType.VECTOR,
                    data=gdf_admin,
                    crs=self.crs
                ))
            except Exception as e:
                logger.warning(f"Could not load admin boundaries: {e}")

        if landuse_paths:
            for lu_type, lu_path in landuse_paths.items():
                self.register_data_from_path(
                    f'landuse_{lu_type}',
                    lu_path,
                    SourceType.VECTOR
                )

        self.register_parameter(SlopePlugin(weight=0.35, inverse=False))

        landslide_coeff = {
            'forest': 0.2,
            'bare_soil': 0.95,
            'agriculture': 0.7,
            'settlement': 0.6,
            'default': 0.5
        }
        self.register_parameter(LandUsePlugin(
            weight=0.25,
            coefficients=landslide_coeff
        ))

        self.register_parameter(ElevationPlugin(weight=0.15, inverse=False))

        if rainfall_path:
            self.register_parameter(RainfallPlugin(weight=0.15))

        logger.info("Landslide risk engine configured")


class DroughtRiskEngine(RiskEngine):
    """
    Pre-configured engine for drought risk assessment

    Default parameters:
    - Rainfall deficit: 30%
    - TWI (low = high risk): 25%
    - Land Use (irrigation access): 20%
    - Soil type: 15%
    - Elevation: 10%

    This also servers as an example for now. I also haven't tested this.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        log_section("DROUGHT RISK ENGINE INITIALIZED")

    def auto_configure(self,
                       rainfall_path: Optional[Path] = None,
                       twi_path: Optional[Path] = None,
                       dem_path: Optional[Path] = None,
                       soil_path: Optional[Path] = None,
                       admin_dirs: Optional[list] = None) -> None:

        log_section("AUTO-CONFIGURING DROUGHT ENGINE")

        if rainfall_path:
            self.register_data_from_path('rainfall', rainfall_path, SourceType.VECTOR)

        if twi_path:
            self.register_data_from_path('twi', twi_path, SourceType.RASTER)

        if dem_path:
            self.register_data_from_path('dem', dem_path, SourceType.RASTER)

        if soil_path:
            self.register_data_from_path('soil', soil_path, SourceType.VECTOR)

        if admin_dirs:
            try:
                gdf_admin = load_admin_boundaries(admin_dirs, target_crs=self.crs)
                self.register_data('admin', DataSource(
                    name='admin',
                    source_type=SourceType.VECTOR,
                    data=gdf_admin,
                    crs=self.crs
                ))
            except Exception as e:
                logger.warning(f"Could not load admin boundaries: {e}")

        if rainfall_path:
            self.register_parameter(RainfallPlugin(weight=0.30))

        if twi_path or dem_path:
            self.register_parameter(TWIPlugin(weight=0.25))

        logger.info("Drought risk engine configured")


def create_engine(hazard_type: str, config: Optional[EngineConfig] = None) -> RiskEngine:
    """
    Factory function to create pre-configured engines

    Args:
        hazard_type: 'flood', 'landslide', or 'drought'
        config: Optional configuration

    Returns:
        Configured RiskEngine instance
    """
    engines = {
        'flood': FloodRiskEngine,
        'landslide': LandslideRiskEngine,
        'drought': DroughtRiskEngine
    }

    hazard_type = hazard_type.lower()
    if hazard_type not in engines:
        raise ValueError(f"Unknown hazard type: {hazard_type}. "
                         f"Available: {list(engines.keys())}")

    return engines[hazard_type](config)
