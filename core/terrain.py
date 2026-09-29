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


def calculate_tpi(dem: np.ndarray, cell_size: float = 1.0, radius_cells: int = 3) -> np.ndarray:
    """
    Calculate Topographic Position Index (TPI) (Weiss 2001, Guisan et al. 1999).

    TPI = z_0 - mean(z_R), comparing cell elevation to average elevation
    within an annular/circular neighborhood. Negative values indicate valleys/depressions
    (flood risk), positive indicate ridges/crests.

    Args:
        dem: DEM elevation array
        cell_size: Resolution of the grid cells
        radius_cells: Radius of neighborhood in cells (default 3)

    Returns:
        TPI array with same shape as dem
    """
    from scipy.ndimage import uniform_filter

    window_size = 2 * radius_cells + 1
    mean_elevation = uniform_filter(dem.astype(float), size=window_size, mode='nearest')
    tpi = dem - mean_elevation
    return tpi


def calculate_spi(
    flow_accumulation: np.ndarray,
    slope_rad: np.ndarray,
    cell_size: float = 1.0,
    log_scale: bool = True,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Calculate Stream Power Index (SPI) (Moore et al. 1991).

    SPI = A_s * tan(beta), where A_s is specific catchment area (flow accumulation * cell_size)
    and beta is slope in radians. Measures potential erosion and flood stream power.

    Args:
        flow_accumulation: Flow accumulation or contributing area proxy
        slope_rad: Terrain slope in radians
        cell_size: Resolution of grid cell
        log_scale: Return log(SPI + epsilon) for numerical stability and MCDA scaling
        epsilon: Small positive constant to prevent log(0)

    Returns:
        SPI array
    """
    slope_clamped = np.maximum(slope_rad, 0.001)
    # Specific catchment area As = flow_accumulation * cell_size
    specific_catchment_area = np.maximum(flow_accumulation, 0.0) * cell_size

    spi = specific_catchment_area * np.tan(slope_clamped)

    if log_scale:
        return np.log(spi + epsilon)
    return spi


def calculate_sti(
    flow_accumulation: np.ndarray,
    slope_rad: np.ndarray,
    cell_size: float = 1.0,
    log_scale: bool = True,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Calculate Sediment Transport Index (STI) (Moore & Burch 1986).

    STI = (A_s / 22.13)^0.6 * (sin(beta) / 0.0896)^1.3
    Characterizes sheet and rill erosion capacity and channel scour susceptibility.

    Args:
        flow_accumulation: Contributing flow accumulation
        slope_rad: Slope angle in radians
        cell_size: Grid cell resolution
        log_scale: Return log(STI + epsilon) for numerical stability and MCDA scaling
        epsilon: Small constant for stability

    Returns:
        STI array
    """
    slope_clamped = np.maximum(slope_rad, 0.001)
    specific_catchment_area = np.maximum(flow_accumulation, 0.0) * cell_size

    term1 = np.power(np.maximum(specific_catchment_area / 22.13, 0.0), 0.6)
    term2 = np.power(np.maximum(np.sin(slope_clamped) / 0.0896, 0.0), 1.3)

    sti = term1 * term2

    if log_scale:
        return np.log(sti + epsilon)
    return sti
