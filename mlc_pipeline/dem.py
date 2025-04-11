# mlc_pipeline/dem.py
import cv2
import numpy as np
import requests
import concurrent.futures
from scipy.ndimage import gaussian_filter
import ee
from mlc_pipeline.utils import split_bbox
import logging

class DEMHandler:
    def __init__(self, bbox, config):
        self.bbox = bbox
        self.n_subboxes = config.get("n_subboxes", 10)
        self.sigma = config.get("sigma", 3)

    def fetch_dem_tile_ee(self, sub_bbox, scale):
        geometry = ee.Geometry.Rectangle(sub_bbox)
        dem = ee.Image("USGS/SRTMGL1_003").clip(geometry)
        params = {
            'region': geometry.coordinates().getInfo(),
            'scale': scale,
            'format': 'GeoTIFF'
        }
        url = dem.getDownloadURL(params)
        response = requests.get(url)
        if response.status_code == 200:
            data = np.asarray(bytearray(response.content), dtype=np.uint8)
            dem_tile = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if dem_tile is None:
                raise Exception("Failed to decode DEM tile.")
            return dem_tile
        else:
            raise Exception(f"Failed to download DEM tile (status code: {response.status_code})")

    def get_dem(self, scale):
        sub_bboxes = split_bbox(self.bbox, self.n_subboxes)
        tiles = [None] * len(sub_bboxes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(self.fetch_dem_tile_ee, sb, scale): idx 
                               for idx, sb in enumerate(sub_bboxes)}
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    tiles[idx] = future.result()
                except Exception as e:
                    logging.warning(f"Failed to fetch DEM tile for {sub_bboxes[idx]} at index {idx}: {e}")
        valid_tiles = [tile for tile in tiles if tile is not None]
        if not valid_tiles:
            raise RuntimeError("No DEM tiles were downloaded successfully.")
        min_height = min(tile.shape[0] for tile in valid_tiles)
        resized_tiles = [cv2.resize(tile, (tile.shape[1], min_height), interpolation=cv2.INTER_LANCZOS4)
                         for tile in valid_tiles]
        full_dem = np.hstack(resized_tiles)
        return full_dem

    def smooth_dem(self, dem_data):
        logging.info("Smoothing DEM data...")
        return gaussian_filter(dem_data, sigma=self.sigma)

    def save_dem(self, dem_data, dem_data_smooth, output_dir):
        import matplotlib.pyplot as plt
        from pathlib import Path
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dem_path = out_dir / "dem.png"
        plt.figure()
        plt.imshow(dem_data, cmap="terrain")
        plt.colorbar(label="Elevation (m)")
        plt.savefig(str(dem_path), bbox_inches='tight', dpi=500)
        plt.close()
        logging.info(f"Saved DEM image to {dem_path}")

        dem_smooth_path = out_dir / "dem_smoothed.png"
        plt.figure()
        plt.imshow(dem_data_smooth, cmap="terrain")
        plt.colorbar(label="Elevation (m)")
        plt.savefig(str(dem_smooth_path), bbox_inches='tight', dpi=500)
        plt.close()
        logging.info(f"Saved smoothed DEM image to {dem_smooth_path}")
