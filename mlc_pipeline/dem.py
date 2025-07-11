# mlc_pipeline/dem.py

import cv2
import numpy as np
import requests
import concurrent.futures
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import ee
from mlc_pipeline.utils import split_bbox
import logging

def _strip_zm(coords):
    if isinstance(coords[0], (float, int)):
        return coords[:2]
    return [_strip_zm(c) for c in coords]

class DEMHandler:
    def __init__(self, config: dict):
        self.cfg = config

        if self.cfg.get('shapefile', False):
            # read and reproject shapefile to WGS84
            gdf = gpd.read_file(self.cfg['shapefile_path'])
            if gdf.crs is None:
                if 'shapefile_crs' in self.cfg:
                    gdf = gdf.set_crs(self.cfg['shapefile_crs'])
                else:
                    raise ValueError("Shapefile has no CRS; set 'shapefile_crs' in config")
            gdf = gdf.to_crs(epsg=4326)

            # store GeoJSON for masking if needed
            geom = gdf.unary_union.__geo_interface__
            geom['coordinates'] = _strip_zm(geom['coordinates'])
            self.geojson = geom

            # use its bounding box for download
            self.bbox = list(gdf.total_bounds)
        else:
            self.geojson = None
            self.bbox = config['bbox']

        self.n_subboxes = config.get("n_subboxes", 10)
        self.sigma = config.get("sigma", 3)

    def fetch_dem_tile_ee(self, sub_bbox, scale):
        region = (
            ee.Geometry(self.geojson)
            if self.geojson
            else ee.Geometry.Rectangle(sub_bbox)
        )
        dem = ee.Image("USGS/SRTMGL1_003").clip(region)
        params = {'region': region.coordinates().getInfo(),
                  'scale': scale,
                  'format': 'GeoTIFF'}
        url = dem.getDownloadURL(params)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(f"DEM download failed: {resp.status_code}")
        data = np.asarray(bytearray(resp.content), dtype=np.uint8)
        tile = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if tile is None:
            raise Exception("Failed to decode DEM tile.")
        return tile

    def get_dem(self, scale):
        """
        If shapefile is used, download exactly one DEM tile over its bbox.
        Otherwise split bbox into subboxes as before.
        """
        if self.geojson:
            # single download over full bbox
            return self.fetch_dem_tile_ee(self.bbox, scale)

        # split into tiles
        sub_bboxes = split_bbox(self.bbox, self.n_subboxes)
        tiles = [None] * len(sub_bboxes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = {
                ex.submit(self.fetch_dem_tile_ee, sb, scale): i
                for i, sb in enumerate(sub_bboxes)
            }
            for f in concurrent.futures.as_completed(futures):
                idx = futures[f]
                try:
                    tiles[idx] = f.result()
                except Exception as e:
                    logging.warning(f"DEM tile {idx} failed: {e}")

        valid = [t for t in tiles if t is not None]
        if not valid:
            raise RuntimeError("No DEM tiles succeeded")
        # stitch horizontally after normalizing heights
        min_h = min(t.shape[0] for t in valid)
        resized = [
            cv2.resize(t, (t.shape[1], min_h), interpolation=cv2.INTER_LANCZOS4)
            for t in valid
        ]
        return np.hstack(resized)

    def smooth_dem(self, dem_data):
        logging.info("Smoothing DEM data...")
        return gaussian_filter(dem_data, sigma=self.sigma)

    def save_dem(self, dem_data, dem_data_smooth, output_dir):
        import matplotlib.pyplot as plt
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for arr, fname in [(dem_data, "dem.png"), (dem_data_smooth, "dem_smoothed.png")]:
            p = out / fname
            plt.figure()
            plt.imshow(arr, cmap="terrain")
            plt.colorbar(label="Elevation (m)")
            plt.savefig(str(p), bbox_inches='tight', dpi=500)
            plt.close()
            logging.info(f"Saved {fname} to {p}")
