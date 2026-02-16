from pathlib import Path
from murasa.core.base import DataLoaderPlugin
from murasa.core.file_scanner import scan_all
from murasa.core.logger import log_success, log_warning


class RiversLoader(DataLoaderPlugin):
    """
    Load river networks from RBI directories
    Expected patterns: SUNGAI_LN, SUNGAI, RIVER
    """
    provides = ["river"]
    requires = []

    RIVER_PATTERNS = ['SUNGAI_LN', 'SUNGAI', 'RIVER']

    def __init__(self, config):
        super().__init__("RiversLoader")
        self.admin_dirs = [Path(d) for d in config.paths.admin_dirs]
        self.target_crs = config.spatial.target_crs

    @classmethod
    def can_handle(cls, config) -> bool:
        return config.paths.admin_dirs is not None and len(config.paths.admin_dirs) > 0

    def load(self, registry):
        self.log_loading("Loading river networks...")

        gdf_rivers = scan_all(self.admin_dirs, self.RIVER_PATTERNS, self.target_crs)

        if gdf_rivers.empty:
            log_warning("No river data found in admin directories")
            return {}

        source = registry.register_vector("river", gdf_rivers, self.target_crs, feature_count=len(gdf_rivers))
        self.loaded_sources["river"] = source

        log_success(f"Loaded river networks: {len(gdf_rivers)} features")
        return self.loaded_sources
