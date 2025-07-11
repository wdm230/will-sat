import ee
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import concurrent.futures
import geopandas as gpd
from mlc_pipeline.utils import auto_utm_epsg

def _drop_zm(coords):
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (list, tuple)):
            return [_drop_zm(c) for c in coords]
        return coords[:2]
    return coords

class SentinelHandler:
    def __init__(self, config: dict):
        self.cfg = config

        if self.cfg.get('shapefile', False):
            # Read and reproject shapefile to WGS84
            gdf = gpd.read_file(self.cfg['shapefile_path'])
            logging.info(f"Loaded shapefile from {self.cfg['shapefile_path']}, initial CRS={gdf.crs}")
            if gdf.crs is None:
                if 'shapefile_crs' in self.cfg:
                    gdf = gdf.set_crs(self.cfg['shapefile_crs'])
                else:
                    raise ValueError("Shapefile has no CRS; set 'shapefile_crs' in config")
            gdf = gdf.to_crs(epsg=4326)
            logging.info("Reprojected shapefile to EPSG:4326")

            # If line, buffer into polygon
            if gdf.geom_type.isin(['LineString', 'MultiLineString']).any():
                buf = self.cfg.get('shapefile_buffer', 100)
                minx, miny, maxx, maxy = gdf.total_bounds
                utm_epsg = auto_utm_epsg([minx, miny, maxx, maxy])

                gdf = gdf.to_crs(epsg=utm_epsg).buffer(buf).to_crs(epsg=4326)

            # Union and clean coordinates
            union = gdf.unary_union
            geom = union.__geo_interface__
            clean = _drop_zm(geom['coordinates'])
            self.geojson = {'type': geom['type'], 'coordinates': clean}


            # bounding box from shapefile
            minx, miny, maxx, maxy = gdf.total_bounds
            self.bbox = [minx, miny, maxx, maxy]

        else:
            self.geojson = None
            self.bbox = config['bbox']
            logging.info(f"Using bbox from config: {self.bbox}")

    def download_s2_median_mndwi(self, scale):
        """
        Download median MNDWI composite over region, masked by shapefile polygon if given.
        """
        logging.info(f"Starting download_s2_median_mndwi at scale={scale}")
        # define region geometry
        region = ee.Geometry(self.geojson) if self.geojson else ee.Geometry.Rectangle(self.bbox)


        # filter and compose median
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate('2023-01-01', '2023-01-31')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        )
        try:
            count = coll.size().getInfo()
        except Exception as e:
            logging.warning(f"Could not fetch collection size: {e}")

        # compute MNDWI per image then take median
        mndwi_coll = coll.map(
            lambda img: img.normalizedDifference(['B3','B11']).rename('MNDWI')
        )
        mndwi_median = mndwi_coll.median().clip(region)
        logging.info("Computed median MNDWI composite and clipped to region")

        # mask outside polygon (if provided)
        if self.geojson:
            poly = ee.Geometry(self.geojson)
            mask = ee.Image.constant(1).clip(poly)
            mndwi_median = mndwi_median.updateMask(mask)

        # visualize and fill masked with black
        viz = (
            mndwi_median
              .visualize(min=-1, max=1, palette=['black','white','blue'])
              .unmask(0)
        )


        url = viz.getThumbURL({
            'region': region.coordinates().getInfo(),
            'format': 'png',
            'scale': scale
        })

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            arr = np.array(img)
            date = ee.Date(coll.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
            logging.info(f"First scene date: {date}")
            return arr, date
        except Exception as e:
            logging.error(f"Thumbnail download failed: {e}")
            return None, None

    def download_subboxes(self, indexed):
        """
        Single-region median download (ignores subboxes).
        """
        scale = self.cfg.get('sentinel', {}).get('scale', 20)
        logging.info(f"download_subboxes called with indexed={indexed}, scale={scale}")
        img, date = self.download_s2_median_mndwi(scale)
        if img is None:
            logging.error("No image returned from download_s2_median_mndwi")
            return []
        return [(0, self.bbox, img, date, scale)]
