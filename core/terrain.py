import numpy as np


def calculate_gradient(dem: np.ndarray, cell_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate first derivatives (gradients) of the DEM.
    
    Args:
        dem: Digital Elevation Model array
        cell_size: Resolution of the DEM (in same units as elevation)
        
    Returns:
        (dz_dy, dz_dx) tuple
    """

    dz_dy, dz_dx = np.gradient(dem, cell_size)
    return dz_dy, dz_dx


def calculate_slope(dem: np.ndarray, cell_size: float = 1.0, degrees: bool = True) -> np.ndarray:
    """
    Calculate slope from DEM.
    
    Args:
        dem: DEM array
        cell_size: Resolution
        degrees: Return in degrees (True) or radians (False)
    """
    dz_dy, dz_dx = calculate_gradient(dem, cell_size)

    hypot = np.hypot(dz_dx, dz_dy)
    slope_rad = np.arctan(hypot)

    if degrees:
        return np.degrees(slope_rad)
    return slope_rad


def calculate_aspect(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """
    Calculate aspect (direction of slope) in degrees [0, 360].
    North = 0, East = 90, South = 180, West = 270.
    """
    dz_dy, dz_dx = calculate_gradient(dem, cell_size)

    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    return aspect


def calculate_curvature(dem: np.ndarray, cell_size: float = 1.0, method: str = 'profile') -> np.ndarray:
    """
    Compute terrain curvature (second derivative metrics).
    
    Args:
        method: 'profile', 'plan', or 'total'
    """

    dz_dy, dz_dx = np.gradient(dem, cell_size)

    d2z_dx2 = np.gradient(dz_dx, cell_size, axis=1)
    d2z_dy2 = np.gradient(dz_dy, cell_size, axis=0)
    d2z_dxdy = np.gradient(dz_dx, cell_size, axis=0)

    p = dz_dx
    q = dz_dy
    p2 = p * p
    q2 = q * q

    if method == "profile":

        num = p2 * d2z_dx2 + 2 * p * q * d2z_dxdy + q2 * d2z_dy2
        denom = (p2 + q2) * np.power(1 + p2 + q2, 1.5)

        denom = np.where(denom == 0, 1.0, denom)

        return -num / denom

    elif method == "plan":

        num = q2 * d2z_dx2 - 2 * p * q * d2z_dxdy + p2 * d2z_dy2
        denom = np.power(p2 + q2, 1.5)
        denom = np.where(denom == 0, 1.0, denom)

        return num / denom

    elif method == "total":
        return d2z_dx2 + d2z_dy2

    elif method == "tangential":

        num = q2 * d2z_dx2 - 2 * p * q * d2z_dxdy + p2 * d2z_dy2
        denom = (p2 + q2) * np.sqrt(1 + p2 + q2)
        denom = np.where(denom == 0, 1.0, denom)
        return num / denom

    else:
        raise ValueError(f"Unknown curvature method: {method}")
