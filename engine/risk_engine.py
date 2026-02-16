from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask

from murasa.core.config import EngineConfig
from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.logger import logger, log_section, log_subsection, log_success
from murasa.core.base import ParameterPlugin, PostProcessorPlugin


class RiskEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.registry = DataRegistry()
        self.parameters: List[ParameterPlugin] = []
        self.post_processors: List[PostProcessorPlugin] = []

        self.grid_shape: Optional[Tuple[int, int]] = None
        self.transform: Optional[Any] = None
        self.crs: str = self.config.spatial.target_crs

        self.study_area_mask: Optional[np.ndarray] = None
        self.admin_units: Optional[gpd.GeoDataFrame] = None

        self.risk_raster: Optional[np.ndarray] = None
        self.exclusion_mask: Optional[np.ndarray] = None

    def register_data(self, name: str, source: DataSource) -> None:
        self.registry.register(name, source)

    def register_data_from_path(self, name: str, path: Path,
                                source_type: SourceType) -> None:
        source = DataSource(
            name=name,
            source_type=source_type,
            path=path,
            crs=self.crs
        )
        self.registry.register(name, source)

    def auto_register_from_config(self) -> None:
        log_subsection("Auto-registering data sources from factors")

        for name, factor in self.config.factors.items():
            if factor.source_path and factor.source_path.exists():
                source_type = (SourceType.RASTER if factor.source_type == 'raster'
                               else SourceType.VECTOR)
                self.register_data_from_path(name, factor.source_path, source_type)
                logger.info(f"  {name}: {factor.source_path}")
            elif factor.source_type == 'derived':
                logger.info(f"  {name}: derived (will be computed)")
            else:
                logger.warning(f"  {name}: path not found or not specified")

        log_success(f"Registered {len(self.registry)} data sources")

    def register_parameter(self, plugin: ParameterPlugin) -> bool:
        if not plugin.validate_requirements(self.registry):
            logger.warning(f"Plugin '{plugin.name}' missing required data")
            return False

        self.parameters.append(plugin)
        log_success(f"Registered: {plugin.name} (weight={plugin.weight:.3f})")
        return True

    def register_post_processor(self, plugin: PostProcessorPlugin) -> None:
        self.post_processors.append(plugin)
        log_success(f"Registered Post-Processor: {plugin.name}")

    def clear_parameters(self) -> None:
        self.parameters.clear()
        logger.info("Cleared all parameters")

    def normalize_weights(self) -> None:
        total = sum(p.weight for p in self.parameters)

        if abs(total - 1.0) < 0.001:
            return

        logger.warning(f"Weights sum to {total:.3f}, normalizing...")
        for param in self.parameters:
            param.weight /= total

    def setup_grid(self, bounds: Optional[Tuple] = None,
                   from_admin: bool = True) -> None:
        study_area_key = self.config.spatial.study_area_key
        if from_admin and self.registry.has(study_area_key):
            admin_source = self.registry.get(study_area_key)
            bounds = admin_source.data.total_bounds
            logger.info(f"Grid bounds from {study_area_key} data")

        if bounds is None:
            raise ValueError(f"Grid bounds required (provide bounds or {study_area_key} data)")

        resolution = self.config.spatial.resolution
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)

        self.transform = from_bounds(*bounds, width, height)
        self.grid_shape = (height, width)

        log_success(f"Grid: {width}×{height} pixels @ {resolution}m")

    def set_study_area(self, admin_gdf: Optional[gpd.GeoDataFrame] = None) -> None:
        study_area_key = self.config.spatial.study_area_key
        if admin_gdf is None:
            if not self.registry.has(study_area_key):
                logger.warning(f"No study area defined using key '{study_area_key}', processing full grid")
                return
            admin_gdf = self.registry.get(study_area_key).data

        self.admin_units = admin_gdf

        self.study_area_mask = geometry_mask(
            admin_gdf.geometry,
            transform=self.transform,
            invert=True,
            out_shape=self.grid_shape
        )

        log_success(f"Study area: {len(admin_gdf)} units from key '{study_area_key}'")

    def calculate_risk(self) -> np.ndarray:
        log_section("CALCULATING RISK")

        if not self.parameters:
            raise ValueError("No parameters registered")

        if self.grid_shape is None or self.transform is None:
            raise ValueError("Grid not setup (call setup_grid first)")

        self.normalize_weights()
        self.exclusion_mask = np.zeros(self.grid_shape, dtype=bool)

        risk_components = []

        for plugin in self.parameters:
            try:
                component = plugin.process(
                    self.registry,
                    self.grid_shape,
                    self.transform,
                    self.crs
                )

                output_name = f"risk_{plugin.name}"
                self.registry.register(
                    output_name,
                    DataSource(output_name, SourceType.RASTER, data=component, crs=self.crs)
                )

                if plugin.is_exclusion:
                    mask = (component > 0).astype(bool)
                    self.exclusion_mask |= mask
                    logger.info(f"      Added exclusion mask from {plugin.name}: {mask.sum()} pixels")
                else:
                    component = np.nan_to_num(component, nan=0.0)
                    weighted = component * plugin.weight
                    risk_components.append(weighted)

            except Exception as e:
                logger.error(f"      Failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not risk_components:
            raise ValueError("No valid risk components computed!")

        self.risk_raster = np.sum(risk_components, axis=0).astype(np.float32)

        if self.exclusion_mask.any():
            self.risk_raster[self.exclusion_mask] = np.nan
            logger.info(f"   Total excluded area: {self.exclusion_mask.sum()} pixels")

        if self.study_area_mask is not None:
            self.risk_raster[~self.study_area_mask] = np.nan

        log_section("RISK CALCULATION COMPLETE")
        logger.info(f"Range: {np.nanmin(self.risk_raster):.3f} - "
                    f"{np.nanmax(self.risk_raster):.3f}")
        logger.info(f"Mean: {np.nanmean(self.risk_raster):.3f}")

        return self.risk_raster

    def export_raster(self, output_path: Path) -> None:
        if self.risk_raster is None:
            raise ValueError("No risk raster calculated")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            'driver': 'GTiff',
            'height': self.grid_shape[0],
            'width': self.grid_shape[1],
            'count': 1,
            'dtype': np.float32,
            'crs': self.crs,
            'transform': self.transform,
            'nodata': np.nan,
            'compress': 'lzw'
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(self.risk_raster, 1)

        log_success(f"Raster saved: {output_path}")

    def export_statistics(self, output_path: Path) -> None:
        import json

        stats = {
            'parameters': {
                p.name: {
                    'weight': p.weight,
                    'statistics': p.get_statistics()
                }
                for p in self.parameters
            },
            'risk': {
                'min': float(np.nanmin(self.risk_raster)),
                'max': float(np.nanmax(self.risk_raster)),
                'mean': float(np.nanmean(self.risk_raster)),
                'std': float(np.nanstd(self.risk_raster))
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        log_success(f"Statistics saved: {output_path}")

    def export_all(self, output_dir: Optional[Path] = None) -> None:
        if output_dir is None:
            output_dir = self.config.output.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        self.export_raster(output_dir / "risk_map.tif")

        self.export_statistics(output_dir / "statistics.json")

        log_success(f"All outputs saved to: {output_dir}")

        if self.post_processors:
            log_section("POST-PROCESSING")
            for pp in self.post_processors:
                try:
                    pp.post_process(
                        self.registry,
                        self.risk_raster,
                        self.transform,
                        self.crs,
                        output_dir,
                        config=self.config
                    )
                except Exception as e:
                    logger.error(f"Post-processor {pp.name} failed: {e}")
                    import traceback
                    traceback.print_exc()

    def print_summary(self) -> None:
        log_section("ENGINE SUMMARY")

        logger.info(f"Data sources: {len(self.registry)}")
        self.registry.print_summary()

        logger.info(f"\nParameters: {len(self.parameters)}")
        for param in self.parameters:
            logger.info(f"  - {param}")

        if self.grid_shape:
            logger.info(f"\nGrid: {self.grid_shape[1]}×{self.grid_shape[0]} pixels")
            logger.info(f"Resolution: {self.config.spatial.resolution}m")

        if self.admin_units is not None:
            logger.info(f"\nStudy area: {len(self.admin_units)} units")

    def run(self) -> np.ndarray:
        log_section("RISK ENGINE - FULL RUN")

        self.auto_register_from_config()
        self.setup_grid()
        self.set_study_area()

        risk = self.calculate_risk()

        self.export_all()

        log_section("COMPLETE")
        return risk
