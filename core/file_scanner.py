from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from .logger import logger, log_warning


def scan_shapefiles(directories: List[Path],
                    patterns: List[str],
                    target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Scan directories for shapefiles matching pattern names.

    Args:
        directories: List of directories to search
        patterns: Filename patterns to match (case-insensitive substring match)
        target_crs: Target CRS to reproject to

    Returns:
        Merged GeoDataFrame (empty if no files found)
    """
    frames = []

    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue

        for shp_file in directory.glob("*.shp"):
            if _match_pattern(shp_file.name, patterns):
                try:
                    gdf = gpd.read_file(shp_file).to_crs(target_crs)
                    frames.append(gdf)
                    logger.debug(f"  Found: {shp_file.name} ({len(gdf)} features)")
                except Exception as e:
                    log_warning(f"Failed to load SHP {shp_file.name}: {e}")

    return _merge_frames(frames, target_crs)


def scan_gdb_layers(directories: List[Path],
                    patterns: List[str],
                    target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Scan directories for GeoDatabase layers matching pattern names.

    Args:
        directories: List of directories to search
        patterns: Layer name patterns to match (case-insensitive substring match)
        target_crs: Target CRS to reproject to

    Returns:
        Merged GeoDataFrame (empty if no layers found)
    """
    try:
        import fiona
    except ImportError:
        log_warning("fiona not installed, cannot scan GDB files")
        return gpd.GeoDataFrame()

    frames = []

    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue

        for gdb_dir in directory.glob("*.gdb"):
            try:
                layers = fiona.listlayers(str(gdb_dir))
                for layer in layers:
                    if _match_pattern(layer, patterns):
                        try:
                            gdf = gpd.read_file(gdb_dir, layer=layer).to_crs(target_crs)
                            frames.append(gdf)
                            logger.debug(
                                f"  Found: {gdb_dir.name}/{layer} ({len(gdf)} features)"
                            )
                        except Exception as e:
                            log_warning(f"Failed to load GDB layer {layer}: {e}")
            except Exception as e:
                log_warning(f"Failed to read GDB {gdb_dir.name}: {e}")

    return _merge_frames(frames, target_crs)


def scan_all(directories: List[Path],
             patterns: List[str],
             target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Scan both shapefiles and GDB layers.

    Args:
        directories: List of directories to search
        patterns: Filename/layer patterns to match
        target_crs: Target CRS to reproject to

    Returns:
        Merged GeoDataFrame from all sources
    """
    gdf_shp = scan_shapefiles(directories, patterns, target_crs)
    gdf_gdb = scan_gdb_layers(directories, patterns, target_crs)

    frames = []
    if not gdf_shp.empty:
        frames.append(gdf_shp)
    if not gdf_gdb.empty:
        frames.append(gdf_gdb)

    return _merge_frames(frames, target_crs)


def scan_filename(directories: List[Path],
                  filename: str,
                  target_crs: str = "EPSG:4326",
                  validate_geometries: bool = True) -> gpd.GeoDataFrame:
    """
    Scan directories for an exact filename match (e.g., ADMINISTRASIDESA_AR_25K.shp).

    This replaces the existing `load_vector_from_multiple_dirs` function.

    Args:
        directories: List of directories to search
        filename: Exact filename to match
        target_crs: Target CRS to reproject to
        validate_geometries: Whether to remove invalid geometries

    Returns:
        Merged GeoDataFrame
    """
    frames = []

    for directory in directories:
        directory = Path(directory)
        file_path = directory / filename

        if file_path.exists():
            try:
                gdf = gpd.read_file(file_path).to_crs(target_crs)

                if validate_geometries:
                    invalid = ~gdf.geometry.is_valid
                    if invalid.any():
                        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid].geometry.buffer(0)
                        logger.debug(f"  Fixed {invalid.sum()} invalid geometries in {file_path.name}")

                frames.append(gdf)
                logger.debug(f"  Found: {file_path.name} ({len(gdf)} features)")
            except Exception as e:
                log_warning(f"Failed to load {file_path.name}: {e}")

    return _merge_frames(frames, target_crs)


def _match_pattern(name: str, patterns: List[str]) -> bool:
    name_upper = name.upper()
    return any(p.upper() in name_upper for p in patterns)


def _merge_frames(frames: list, target_crs: str) -> gpd.GeoDataFrame:
    """Merge list of GeoDataFrames into one."""
    if not frames:
        return gpd.GeoDataFrame()

    if len(frames) == 1:
        return frames[0]

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=target_crs
    )
