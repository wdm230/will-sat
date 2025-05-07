# mlc_pipeline/utils.py
import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib as plt
import rasterio
import os
import shutil
import yaml


def copy_config_file(source_path: str, dest_dir: str):
    """
    Copy an existing config.yaml into the output directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(source_path, os.path.join(dest_dir, os.path.basename(source_path)))

def write_config_dict(cfg: dict, dest_dir: str, filename: str = "config.yaml"):
    """
    Dump the in-memory config dict to YAML in the output directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, filename)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)

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

def find_edge_nodes_by_coord(self, mesh, mask_shape, tol=None):
    """
    Return the indices of mesh.vertices touching the image border
    (within 'tol' pixels).
    """
    if tol is None:
        tol = self.boundary_tol
    H, W = mask_shape
    verts = mesh.vertices[:, :2]  # only x,y
    xs, ys = verts[:,0], verts[:,1]

    on_left   = xs <=     tol
    on_right  = xs >= (W-1) - tol
    on_top    = ys <=     tol
    on_bottom = ys >= (H-1) - tol

    mask = on_left | on_right | on_top | on_bottom
    return np.nonzero(mask)[0]  # array of vertex‐indices

