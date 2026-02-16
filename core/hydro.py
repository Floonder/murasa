import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import grey_dilation, binary_dilation

from .logger import logger

"""
[TODO]

This is not currently used.

I'm planning to implement more core logic to be easily used later.
"""


def fill_sinks(dem: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Fill local depressions (sinks) in DEM using iterative morphological reconstruction.
    Based on Wang & Liu (2006) concept.
    """
    logger.debug("   Filling sinks in DEM...")

    rows, cols = dem.shape

    result = np.full_like(dem, np.inf)

    mask_boundary = np.ones_like(dem, dtype=bool)
    mask_boundary[1:-1, 1:-1] = False
    mask_boundary |= np.isnan(dem)

    result[mask_boundary] = dem[mask_boundary]

    kernel = generate_binary_structure(2, 2)
    from scipy.ndimage import minimum_filter

    prev = result.copy()

    for _ in range(1000) if getattr(dem, 'size', 0) < 1e6 else range(100):
        min_n = minimum_filter(result, footprint=kernel, mode='constant', cval=np.inf)
        result = np.maximum(dem, min_n)

        if np.array_equal(result, prev):
            break
        prev = result.copy()

    return result.astype(np.float32)


def flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """
    Compute simplified D8 Flow Accumulation proxy based on slope.
    """
    dy, dx = np.gradient(dem)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_rad = np.maximum(slope_rad, 0.001)

    return 1.0 / slope_rad
