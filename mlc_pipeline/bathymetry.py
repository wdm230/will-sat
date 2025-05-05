import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter

def compute_gradient_bathymetry(dem, water_mask, z_below, d1_ratio, d2_ratio, sigma):
    """
    Applies a gradient-based experimental bathymetry to fill the water gap in the DEM.

    Parameters:
      dem: 2D numpy array (float) representing the DEM.
      water_mask: Binary mask (same shape as dem) where water_mask > 0 indicates water.
      z_below: Target water elevation (e.g., -5 means 5m below the baseline level).
      d1_ratio: Fraction of maximum gap distance used for the initial ramp (shore-to-depth).
      d2_ratio: Fraction of maximum gap distance used for the flat (constant depth) region.
      sigma: Standard deviation for Gaussian smoothing (to blend the transitions).

    Returns:
      dem_modified: The DEM with experimental bathymetry applied in the water areas.
    """
    # Copy the DEM and remove water pixels by setting them to NaN
    dem_modified = dem.copy()
    dem_modified[water_mask > 0] = np.nan

    # Create a boolean mask of the water gap (where we lack elevation)
    water_gap = np.isnan(dem_modified)
    # The land area is defined as not water gap.
    land_mask = ~water_gap

    # Create a binary image for distance transform:
    # land pixels (non-gap) become 1, so we can measure distance from the gap to land.
    inv_mask = land_mask.astype(np.uint8)
    # Compute the Euclidean distance transform for each water-gap pixel.
    dist = cv2.distanceTransform(inv_mask, distanceType=cv2.DIST_L2, maskSize=5)
    # Set distances to zero outside of the gap.
    dist_gap = np.where(water_gap, dist, 0)

    # Determine the maximum distance within the gap.
    max_distance = np.nanmax(dist_gap)
    if max_distance == 0:
        return dem_modified  # Nothing to fill if no gap is found.

    # Calculate ramp lengths based on provided ratios.
    d1 = d1_ratio * max_distance
    d2 = d2_ratio * max_distance

    # Prepare an array to hold the newly computed depths.
    new_depth = np.zeros_like(dist_gap, dtype=np.float32)

    # Vectorized piecewise computation:
    # --- Ramp from shore to target depth ---
    ramp1 = (dist_gap <= d1)
    new_depth[ramp1] = (dist_gap[ramp1] / d1) * z_below

    # --- Flat region at target depth ---
    flat = (dist_gap > d1) & (dist_gap <= (d1 + d2))
    new_depth[flat] = z_below

    # --- Ramp from flat region back towards land (if the gap spans to a second shoreline) ---
    ramp2 = dist_gap > (d1 + d2)
    d3 = max_distance - d1 - d2  # remaining distance for the second ramp
    if d3 == 0:
        new_depth[ramp2] = z_below
    else:
        new_depth[ramp2] = z_below + ((dist_gap[ramp2] - d1 - d2) / d3) * (-z_below)

    # Replace water gap (NaN) pixels in DEM with the computed bathymetry.
    dem_modified[water_gap] = new_depth[water_gap]

    # Smooth the entire DEM to blend the new bathymetry naturally with the surrounding terrain.
    dem_smoothed = gaussian_filter(dem_modified, sigma=sigma)
    return dem_smoothed

def fill_experimental_bathymetry(dem, water_mask, z_below=-5, d1_ratio=0.3, d2_ratio=0.5, sigma=1):
    """
    Wrapper to apply the gradient-based experimental bathymetry to the DEM.

    Parameters:
      dem: 2D numpy array (float) of the DEM.
      water_mask: Binary water mask (nonzero values indicate water).
      z_below: Desired water surface elevation (typically a negative value for depth).
      d1_ratio: Fraction of max gap distance for the initial slope.
      d2_ratio: Fraction of max gap distance for the flat region.
      sigma: Smoothing parameter for Gaussian filtering.

    Returns:
      Modified DEM with the experimental bathymetry applied.
    """
    return compute_gradient_bathymetry(dem, water_mask, z_below, d1_ratio, d2_ratio, sigma)
