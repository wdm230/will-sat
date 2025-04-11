# mlc_pipeline/utils.py
import logging
from pathlib import Path

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
