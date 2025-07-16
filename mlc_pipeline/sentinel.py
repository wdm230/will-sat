import logging
import yaml
import ee
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from mlc_pipeline.utils import auto_utm_epsg, setup_logger, split_bbox
from mlc_pipeline.exclusion import ExclusionEditor
from mlc_pipeline.dem import DEMHandler
from mlc_pipeline.classification import Classifier
from mlc_pipeline.meshing import MeshBuilder
from mlc_pipeline.curvimesh import CurviMeshBuilder
from mlc_pipeline.hotstart import HotstartBuilder
from mlc_pipeline.bc_maker import BCBuilder
from mlc_pipeline.metadata import MeshMetadata
from mlc_pipeline.bathymetry import Bathymetry


def _drop_zm(coords):
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (list, tuple)):
            return [_drop_zm(c) for c in coords]
        return coords[:2]
    return coords


class SentinelHandler:
    def __init__(self, config: dict):
        self.cfg = config
        self.sd = self.cfg.get('sentinel', {})['start_date']
        self.ed = self.cfg.get('sentinel', {})['end_date']

        if self.cfg.get('shapefile', False):
            import geopandas as gpd
            gdf = gpd.read_file(self.cfg['shapefile_path'])
            logging.info(f"Loaded shapefile from {self.cfg['shapefile_path']}, initial CRS={gdf.crs}")
            if gdf.crs is None:
                if 'shapefile_crs' in self.cfg:
                    gdf = gdf.set_crs(self.cfg['shapefile_crs'])
                else:
                    raise ValueError("Shapefile has no CRS; set 'shapefile_crs' in config")
            gdf = gdf.to_crs(epsg=4326)
            logging.info("Reprojected shapefile to EPSG:4326")

            if gdf.geom_type.isin(['LineString', 'MultiLineString']).any():
                buf = self.cfg.get('shapefile_buffer', 100)
                minx, miny, maxx, maxy = gdf.total_bounds
                utm_epsg = auto_utm_epsg([minx, miny, maxx, maxy])
                gdf = gdf.to_crs(epsg=utm_epsg).buffer(buf).to_crs(epsg=4326)

            union = gdf.unary_union
            geom = union.__geo_interface__
            clean = _drop_zm(geom['coordinates'])
            self.geojson = {'type': geom['type'], 'coordinates': clean}

            minx, miny, maxx, maxy = gdf.total_bounds
            self.bbox = [minx, miny, maxx, maxy]
        else:
            self.geojson = None
            self.bbox = config['bbox']
            logging.info(f"Using bbox from config: {self.bbox}")

    def download_s2_median_mndwi(self, scale):
        logging.info(f"Starting download_s2_median_mndwi at scale={scale}")
        region = ee.Geometry(self.geojson) if self.geojson else ee.Geometry.Rectangle(self.bbox)
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(self.sd, self.ed)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        )
        mndwi_coll = coll.map(lambda img: img.normalizedDifference(['B3','B11']).rename('MNDWI'))
        mndwi_median = mndwi_coll.median().clip(region)
        if self.geojson:
            mask = ee.Image.constant(1).clip(ee.Geometry(self.geojson))
            mndwi_median = mndwi_median.updateMask(mask)
        viz = mndwi_median.visualize(min=-1, max=1, palette=['black','white','blue']).unmask(0)
        url = viz.getThumbURL({'region': region.coordinates().getInfo(), 'format': 'png', 'scale': scale})
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        arr = np.array(img)
        date = ee.Date(coll.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
        logging.info(f"First scene date: {date}")
        return arr, date

    def download_s2_true_color(self, scale, bbox, target_epsg, width, height):
        region = ee.Geometry.Rectangle(self.bbox)

        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(self.sd, self.ed)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        )
        rgb_median = coll.median().select(['B4','B3','B2']).clip(region)
        viz = rgb_median.visualize(min=[0,0,0], max=[3000,3000,3000])
        params = {
            'region': [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]],
            'crs': f"EPSG:{target_epsg}",
            'dimensions': [width, height]
        }
        url = viz.getThumbURL(params)
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.array(img)

    def download_subboxes(self, indexed):
        scale = self.cfg.get('sentinel', {}).get('scale', 20)
        img, date = self.download_s2_median_mndwi(scale)
        if img is None:
            logging.error("No image returned")
            return []
        return [(0, self.bbox, img, date, scale)]


