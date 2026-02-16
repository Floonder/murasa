from pathlib import Path
from murasa.core.base import DataLoaderPlugin
from murasa.core.file_scanner import scan_filename
from murasa.core.logger import log_success, log_error


class AdminBoundariesLoader(DataLoaderPlugin):
    """
    Load administrative boundaries from RBI directories
    Expected filename: ADMINISTRASIDESA_AR_25K.shp
    """
    provides = ["admin"]
    requires = []

    def __init__(self, config):
        super().__init__("AdminBoundariesLoader")
        self.admin_dirs = config.paths.admin_dirs
        self.target_crs = config.spatial.target_crs
        self.filename = "ADMINISTRASIDESA_AR_25K.shp"

    @classmethod
    def can_handle(cls, config) -> bool:
        return config.paths.admin_dirs is not None and len(config.paths.admin_dirs) > 0

    def load(self, registry):
        self.log_loading(f"Loading administrative boundaries from {len(self.admin_dirs)} directories...")

        gdf = scan_filename(self.admin_dirs, self.filename, self.target_crs)

        if gdf.empty:
            log_error(f"No {self.filename} found in provided directories")
            return {}

        source = registry.register_vector("admin", gdf, self.target_crs, feature_count=len(gdf))
        self.loaded_sources["admin"] = source

        log_success(f"Loaded administrative boundaries: {len(gdf)} features")
        return self.loaded_sources
