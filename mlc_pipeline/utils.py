# mlc_pipeline/utils.py
import logging
from pathlib import Path
import cv2
import numpy as np

def auto_utm_epsg(bbox):
    """
    Given a bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84,
    compute the center and return the corresponding UTM EPSG code.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    utm_zone = int((center_lon + 180) / 6) + 1
    if center_lat >= 0:
        return 32600 + utm_zone
    else:
        return 32700 + utm_zone

def split_bbox(bbox, n):
    """
    Split the overall bounding box along longitude into n equal parts.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    dx = (max_lon - min_lon) / n
    return [[min_lon + i * dx, min_lat, min_lon + (i + 1) * dx, max_lat] for i in range(n)]

def setup_logger(log_file="pipeline.log", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any existing handlers.
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler.
    from pathlib import Path
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def laplacian_smoothing(mask, iterations=5, lambda_val=0.2):
    """
    Iteratively smooth the image using a Laplacian operator.
    
    Parameters:
      mask: Input image (in 0-255, float32).
      iterations: Number of iterations.
      lambda_val: Smoothing factor (controls the amount subtracted per iteration).
      
    Returns:
      smoothed: The smoothed image.
    """
    # Ensure the mask is float32
    smooth = mask.astype(np.float32)
    
    for i in range(iterations):
        # Compute the Laplacian—the second derivative (difference to local average)
        lap = cv2.Laplacian(smooth, cv2.CV_32F)
        # Subtract a fraction of the Laplacian to diffuse high-frequency details.
        smooth = smooth - lambda_val * lap
        # Clip to maintain valid range.
        smooth = np.clip(smooth, 0, 255)
    return smooth

def smooth_mask_with_laplacian(mask, iterations=5, lambda_val=0.2, threshold_val=127):
    """
    Smooth a binary mask using iterative Laplacian smoothing and re-thresholding.
    
    Parameters:
      mask: Input binary mask (numpy array). It can be either in the 0-1 range or 0-255.
      iterations: How many iterations to run the Laplacian smoothing.
      lambda_val: The smoothing factor (a higher lambda means more smoothing per iteration).
      threshold_val: The threshold value for binarization (in the 0-255 range).
      
    Returns:
      smoothed_mask: A binary mask (0 or 255) with smoother boundaries.
    """
    # If the mask is normalized (0-1), scale it up to 0-255.
    if mask.max() <= 1:
        mask_scaled = (mask * 255).astype(np.uint8)
    else:
        mask_scaled = mask.astype(np.uint8)
    
    # Apply Laplacian-based smoothing.
    smoothed = laplacian_smoothing(mask_scaled, iterations=iterations, lambda_val=lambda_val)
    
    # Convert the smoothed image to 8-bit for thresholding.
    smoothed_uint8 = smoothed.astype(np.uint8)
    
    # Threshold to get a binary mask. Adjust the threshold if necessary.
    _, binary_smoothed = cv2.threshold(smoothed_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    
    return binary_smoothed


def get_boundary_nodes(self):
    """Return sorted unique node indices on the boundary."""
    segs = getattr(self, 'boundary_segments', None)
    if segs is None:
        raise ValueError("boundary_segments not defined. Run build_adv_front_mesh first.")
    nodes = np.unique(segs.flatten())
    return nodes

def get_boundary_elements(self, mesh):
    """Return element indices that share at least one boundary edge."""
    segs = self.boundary_segments
    # build a set of frozenset edges for quick lookup
    b_edges = set(map(lambda e: tuple(sorted(e)), segs.tolist()))
    boundary_elems = []
    for eid, face in enumerate(mesh.faces):
        # check each edge of the triangle
        edges = [tuple(sorted((face[i], face[(i+1)%3]))) for i in range(3)]
        if any(e in b_edges for e in edges):
            boundary_elems.append(eid)
    return np.array(boundary_elems, dtype=int)
