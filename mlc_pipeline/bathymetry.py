import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

def apply_centerline_bathymetry(dem_data, water_mask, params=None):
    """
    Applies experimental bathymetry to water pixels using a centerline-based method with
    local coordinate systems and blending with the original DEM near the shoreline.
    
    Parameters:
        dem_data (np.ndarray): 2D array with DEM elevations.
        water_mask (np.ndarray): Binary water mask (nonzero = water).
        params (dict, optional): Parameters to control behavior:
            - 'bathy_delta': Depth difference between bank and center (default 2.5).
            - 'shape_type': Type of cross-sectional shape. Options: 'quadratic',
                            'triangular', 'catenary' (default 'quadratic').
            - 'catenary_param': Parameter (c) controlling the catenary sharpness (default 1.0).
            - 'blend_distance': Number of pixels over which to blend experimental bathymetry (default 50).
    
    Returns:
        modified_dem (np.ndarray): DEM where water pixels have been adjusted to follow
                                   the experimental bathymetry.
    """
    # Ensure water_mask is binary
    water_bin = (water_mask > 0).astype(np.uint8)

    # STEP 1: Extract the centerline using skeletonization.
    skeleton = skeletonize(water_bin.astype(bool)).astype(np.uint8)
    if np.count_nonzero(skeleton) == 0:
        skeleton = water_bin.copy()
    
    # Get coordinates of skeleton pixels.
    skeleton_coords = np.argwhere(skeleton > 0)

    # --- DENSIFY THE SKELETON USING A SPLINE ---
    # Here we attempt a simple ordering: sort the skeleton coordinates by row.
    # This is a heuristic that may work if your centerline is roughly vertical.
    # For more complex cases, a more robust ordering (e.g., via graph-based methods) might be needed.
    if skeleton_coords.shape[0] > 3:
        # Order by the first coordinate (row)
        ordered_points = skeleton_coords[np.argsort(skeleton_coords[:, 0])]
        try:
            # Fit a parametric spline (s=0 ensures an exact interpolation)
            tck, u = splprep([ordered_points[:, 0], ordered_points[:, 1]], s=0)
            # Sample the spline at a high resolution.
            high_res_param = np.linspace(0, 1, 10000)  # Adjust resolution as needed.
            dense_centerline = np.array(splev(high_res_param, tck)).T  # Shape (1000, 2)
            # Use the dense spline points as the new skeleton coordinates.
            skeleton_coords = dense_centerline
        except Exception as e:
            # If spline fitting fails, fall back to the original skeleton
            print("Spline interpolation failed, using original skeleton: ", e)
    # --- END OF SPLINE DENSIFICATION ---

    # STEP 2: For each water pixel, find its nearest centerline pixel.
    water_coords = np.argwhere(water_bin > 0)  # (row, col) indices for water pixels
    if skeleton_coords.shape[0] == 0:
        skeleton_coords = water_coords.copy()  # Fallback if skeleton extraction fails.
    
    # Use a KD-tree for efficient nearest-neighbor search.
    tree = cKDTree(skeleton_coords)
    distances, indices = tree.query(water_coords)
    
    # Compute the maximum distance for normalization.
    d_max = distances.max() if distances.max() > 0 else 1.0
    
    # STEP 3: Compute a local tangent (and hence normal) at each skeleton pixel.
    # A simple way to approximate a tangent is to compute the gradient on the skeleton image.
    grad_y, grad_x = np.gradient(skeleton.astype(float))
    
    # Setup parameters from params input.
    if params is None:
        params = {}
    shape_type     = params.get('shape_type', 'quadratic')
    catenary_param = params.get('catenary_param', 1.0)   # Only used if shape_type=='catenary'
    bathy_delta    = params.get('bathy_delta', 2.5)        # Depth difference from bank to center
    blend_distance = params.get('blend_distance', 50)      # Pixels over which to blend bathymetry
    
    # Determine bank elevation from land pixels in the DEM.
    land_pixels = dem_data[water_bin == 0]
    if land_pixels.size == 0:
        raise ValueError("No land pixels found to determine bank elevation.")
    z_bank = np.median(land_pixels)
    # *** Changed: Invert z by using addition (adjust to your flipped mesh requirements)
    z_center = abs(bathy_delta) - z_bank  # This line is based on your inverted mesh logic.
    
    # Prepare an empty array for the new water elevations.
    new_water_elev = np.zeros_like(dem_data, dtype=float)

    # For each water pixel, compute experimental elevation.
    for idx, (i, j) in enumerate(water_coords):
        # Get nearest centerline coordinate.
        nearest = skeleton_coords[indices[idx]]
        
        # Approximate the tangent at the centerline point.
        ty = grad_y[int(nearest[0]), int(nearest[1])]
        tx = grad_x[int(nearest[0]), int(nearest[1])]
        if tx == 0 and ty == 0:
            tangent = np.array([0, 1], dtype=float)
        else:
            tangent = np.array([ty, tx], dtype=float)
            norm_t = np.linalg.norm(tangent)
            tangent = tangent / norm_t if norm_t > 0 else np.array([0, 1], dtype=float)
        
        # Compute the local normal (perpendicular to the tangent).
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        
        # Vector from the centerline to the water pixel.
        vec = np.array([i - nearest[0], j - nearest[1]], dtype=float)
        # Project the vector onto the normal and take absolute value.
        signed_dist = np.dot(vec, normal)
        u = abs(signed_dist) / d_max  # u=0 at centerline, u=1 at bank.
        u = np.clip(u, 0, 1)
        
        # Apply the chosen shape function.
        if shape_type == 'quadratic':
            shape_val = (1 - u)**2
        elif shape_type == 'triangular':
            shape_val = 1 - u
        elif shape_type == 'catenary':
            c = catenary_param
            # Avoid division by zero. The mapping is re-normalized to ensure the correct endpoints.
            if abs(c) < 1e-5:
                shape_val = 1 - u
            else:
                shape_val = (np.cosh(c*u) - np.cosh(c)) / (1 - np.cosh(c))
        else:
            # Default fallback to quadratic
            shape_val = (1 - u)**2
        
        # Map the shape value to an elevation between z_center and z_bank.
        # At u=0: exp_elev = z_center, and at u=1: exp_elev = z_bank.
        exp_elev = z_center + (z_bank - z_center) * shape_val
        new_water_elev[i, j] = exp_elev

    # STEP 4: Blend the computed experimental water elevation with the original DEM.
    # Compute the distance (in pixels) from each water pixel to land.
    distance_to_land = cv2.distanceTransform(water_bin, cv2.DIST_L2, 5)
    # A linear blend: weight goes from 1 (deep in water) to 0 (at shore).
    blend_factor = np.clip(distance_to_land / blend_distance, 0, 1)
    
    modified_dem = dem_data.copy().astype(float)
    water_idx = np.where(water_bin > 0)
    modified_dem[water_idx] = (blend_factor[water_idx] * new_water_elev[water_idx] +
                               (1 - blend_factor[water_idx]) * dem_data[water_idx])
    
    return modified_dem
