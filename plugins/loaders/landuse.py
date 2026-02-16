from pathlib import Path
from typing import Dict, List, Optional
import geopandas as gpd
import fiona
import pandas as pd

from murasa.core.base import DataLoaderPlugin
from murasa.core.data_source import DataRegistry, DataSource, SourceType
from murasa.core.logger import log_success, log_warning


class LandUseLoader(DataLoaderPlugin):
    """
    Load land use/land cover data from RBI directories
    
    Categories: settlement, forest, shrub, plantation, paddy, water, roads
    """

    provides = [
        "landuse_settlement", "landuse_forest", "landuse_shrub",
        "landuse_plantation", "landuse_paddy", "landuse_water", "landuse_road"
    ]
    requires = []

    DEFAULT_PATTERNS = {
        'settlement': [
            'PEMUKIMAN_AR', 'BANGUNAN_AR', 'INDUSTRI_AR',
            'NIAGA_AR', 'PEMERINTAHAN_AR', 'PENDIDIKAN_AR',
            'ARENAOLAHRAGA_AR', 'SARANAIBADAH_AR', 'KESEHATAN_AR'
        ],
        'forest': [
            'HUTAN', 'HUTANLAHANRENDAH', 'HUTANKERING', 'HUTANRAWA'
        ],
        'shrub': [
            'SEMAKBELUKAR', 'HERBADANRUMPUT', 'LADANG',
            'ALANG', 'TANAMANCAMPUR', 'TEGALAN'
        ],
        'plantation': [
            'PERKEBUNAN', 'KEBUN'
        ],
        'paddy': [
            'SAWAH', 'EMPANG', 'TAMBAK'
        ],
        'water': [
            'DANAU_AR', 'WADUK_AR', 'SITU_AR', 'RAWA_AR'
        ]
    }

    ROAD_PATTERNS = ['JALAN_LN', 'JALAN']

    ROAD_WIDTHS = {
        'tol': 12.0,
        'arteri': 8.0,
        'kolektor': 5.0,
        'lokal': 3.5,
        'default': 3.0
    }

    def __init__(self, config):
        super().__init__("LandUseLoader")
        self.admin_dirs = [Path(d) for d in config.paths.admin_dirs]
        self.target_crs = config.spatial.target_crs
        self.patterns = self.DEFAULT_PATTERNS
        self.load_roads = True
        self.road_widths = self.ROAD_WIDTHS

    @classmethod
    def can_handle(cls, config) -> bool:
        """Check if admin directories are configured"""
        return config.paths.admin_dirs is not None and len(config.paths.admin_dirs) > 0

    def load(self, registry: DataRegistry) -> Dict[str, DataSource]:
        """Load all land use categories and register to registry"""
        self.log_loading("Loading land use data...")

        results = {}

        for category, patterns in self.patterns.items():
            gdf = self._load_layers(patterns)

            if not gdf.empty:
                source_name = f"landuse_{category}"
                source = DataSource(
                    name=source_name,
                    source_type=SourceType.VECTOR,
                    data=gdf,
                    crs=self.target_crs,
                    metadata={'category': category, 'feature_count': len(gdf)}
                )
                registry.register(source_name, source)
                results[source_name] = source
                self.log_loading(f"  {category}: {len(gdf)} features")
            else:
                log_warning(f"  No data found for {category}")

        if self.load_roads:
            gdf_roads = self._load_roads_as_polygons()

            if not gdf_roads.empty:
                source = DataSource(
                    name="landuse_road",
                    source_type=SourceType.VECTOR,
                    data=gdf_roads,
                    crs=self.target_crs,
                    metadata={'category': 'road', 'feature_count': len(gdf_roads)}
                )
                registry.register("landuse_road", source)
                results["landuse_road"] = source
                self.log_loading(f"  roads: {len(gdf_roads)} features (buffered)")

        self.loaded_sources = results
        log_success(f"Loaded {len(results)} land use categories")

        return results

    def _load_layers(self, filename_patterns: List[str]) -> gpd.GeoDataFrame:
        """Scan all admin dirs for files matching patterns and merge them"""
        frames = []

        for admin_dir in self.admin_dirs:
            if not admin_dir.exists():
                continue

            for shp_file in admin_dir.glob("*.shp"):
                if self._match_pattern(shp_file.name, filename_patterns):
                    try:
                        gdf = gpd.read_file(shp_file).to_crs(self.target_crs)
                        gdf = self._standardize_columns(gdf, shp_file.stem)
                        frames.append(gdf)
                    except Exception as e:
                        log_warning(f"Failed to load SHP {shp_file.name}: {e}")

            for gdb_dir in admin_dir.glob("*.gdb"):
                try:
                    layers = fiona.listlayers(str(gdb_dir))
                    for layer in layers:
                        if self._match_pattern(layer, filename_patterns):
                            try:
                                gdf = gpd.read_file(gdb_dir, layer=layer).to_crs(self.target_crs)
                                gdf = self._standardize_columns(gdf, layer)
                                frames.append(gdf)
                            except Exception as e:
                                log_warning(f"Failed to load layer {layer}: {e}")
                except Exception as e:
                    log_warning(f"Failed to read GDB {gdb_dir.name}: {e}")

        if not frames:
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=self.target_crs)

    def _load_roads_as_polygons(self) -> gpd.GeoDataFrame:
        """Load road line features and buffer them to polygons"""
        frames = []

        for admin_dir in self.admin_dirs:
            if not admin_dir.exists():
                continue

            for shp_file in admin_dir.glob("*.shp"):
                if self._match_pattern(shp_file.name, self.ROAD_PATTERNS):
                    try:
                        gdf = gpd.read_file(shp_file).to_crs(self.target_crs)
                        gdf = self._standardize_columns(gdf, shp_file.stem)
                        frames.append(gdf)
                    except Exception as e:
                        log_warning(f"Failed to load road SHP {shp_file.name}: {e}")

            for gdb_dir in admin_dir.glob("*.gdb"):
                try:
                    layers = fiona.listlayers(str(gdb_dir))
                    for layer in layers:
                        if self._match_pattern(layer, self.ROAD_PATTERNS):
                            try:
                                gdf = gpd.read_file(gdb_dir, layer=layer).to_crs(self.target_crs)
                                gdf = self._standardize_columns(gdf, layer)
                                frames.append(gdf)
                            except:
                                pass
                except:
                    pass

        if not frames:
            return gpd.GeoDataFrame()

        gdf_lines = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=self.target_crs)

        gdf_metric = gdf_lines.to_crs("EPSG:3857")

        gdf_metric['geometry'] = gdf_metric.apply(
            lambda row: row.geometry.buffer(self._get_road_width(row.get('REMARK', ''))),
            axis=1
        )

        return gdf_metric.to_crs(self.target_crs)

    def _get_road_width(self, remark) -> float:
        """Determine road buffer width based on REMARK field"""
        if not isinstance(remark, str):
            return self.road_widths.get('default', 3.0)

        r = remark.lower()

        if 'tol' in r:
            return self.road_widths.get('tol', 12.0)
        if 'arteri' in r:
            return self.road_widths.get('arteri', 8.0)
        if 'kolektor' in r:
            return self.road_widths.get('kolektor', 5.0)
        if 'lokal' in r:
            return self.road_widths.get('lokal', 3.5)

        return self.road_widths.get('default', 3.0)

    def _match_pattern(self, filename: str, patterns: List[str]) -> bool:
        fname = filename.upper()
        return any(p.upper() in fname for p in patterns)

    def _standardize_columns(self, gdf: gpd.GeoDataFrame, source_name: str) -> gpd.GeoDataFrame:
        """Standardize column names across different data sources"""
        rename_map = {}

        for col in gdf.columns:
            c = col.upper()
            if 'NAMOBJ' in c:
                rename_map[col] = 'NAMOBJ'
            elif 'REMARK' in c or 'CATATAN' in c:
                rename_map[col] = 'REMARK'

        gdf = gdf.rename(columns=rename_map)

        if 'NAMOBJ' not in gdf.columns:
            gdf['NAMOBJ'] = 'Unknown'
        if 'REMARK' not in gdf.columns:
            gdf['REMARK'] = source_name

        return gdf
