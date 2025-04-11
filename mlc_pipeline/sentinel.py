import ee
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import concurrent.futures

class SentinelHandler:
    def __init__(self, bbox, config):
        self.bbox = bbox
        self.num_samples = config.get("num_samples", 50)
        
    def download_s2_sampled_image_with_scale(self, sub_bbox, scale, num_samples=50):
        """
        Attempts to download a Sentinel-2 composite for the given sub_bbox at the specified scale.
        Returns a tuple (sub_bbox, img_array, scale) on success, or (None, None, None) on failure.
        """
        region = ee.Geometry.Rectangle(sub_bbox)
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(region)
                        .filterDate('2022-01-01', '2023-12-31')
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))
        size = collection.size().getInfo()
        if size == 0:
            logging.warning(f"No images for {sub_bbox} in the specified date range.")
            return None, None, None
    
        list_img = collection.toList(size)
        ns = num_samples if size >= num_samples else size
        step = size / ns
        indices = ee.List.sequence(0, size - 1, step)
        sampled = indices.map(lambda i: ee.Image(list_img.get(ee.Number(i).round())))
        sampled_collection = ee.ImageCollection(sampled)
        composite = sampled_collection.mean()
    
        bands = composite.bandNames().getInfo()
        if 'B3' not in bands or 'B11' not in bands:
            logging.warning(f"Required bands missing for {sub_bbox}. Bands: {bands}")
            return None, None, None
    
        mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        mndwi_rgb = mndwi.visualize(min=-1, max=1, palette=['black', 'white', 'blue'])
    
        params = {'region': region.coordinates().getInfo(), 'format': 'png', 'scale': scale}
        try:
            url = mndwi_rgb.getThumbURL(params)
            response = requests.get(url)
            if response.status_code == 200:
                from io import BytesIO
                from PIL import Image
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_array = np.array(img)
                return sub_bbox, img_array, scale
            else:
                return None, None, None
        except Exception as e:
            return None, None, None
    
    
    def download_subboxes(self, indexed_sub_bboxes):
        """
        Concurrently download composites for a list of enumerated sub_bboxes.
        Each element in indexed_sub_bboxes should be a tuple: (index, sub_bbox).
        This function will try each candidate scale for all sub-boxes together; if one fails at a given scale,
        a single warning is logged and the function moves on to the next scale.
        Returns a list of tuples: (index, sub_bbox, img_array, used_scale) if successful.
        """
        candidate_scales = [5, 10, 15, 20, 30, 50]
        for scale in candidate_scales:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_index = {
                    executor.submit(self.download_s2_sampled_image_with_scale, sb, scale): idx 
                    for idx, sb in indexed_sub_bboxes
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    result = future.result()  # result is (sub_bbox, img_array, scale) or (None, None, None)
                    results.append((idx, result))
            # Check if every sub_bbox succeeded at this candidate scale.
            if all(res[1][0] is not None for res in results):
                # Sort by the explicit index and package the results.
                return [(idx, sub_bbox, img_array, scale) for idx, (sub_bbox, img_array, _) in sorted(results, key=lambda tup: tup[0])]
            else:
                logging.warning(f"Scale {scale} failed, moving to next size.")
        logging.warning("Failed to sample composite for all candidate scales.")
        return []



    def save_composite(self, composite_img, output_path):
        from pathlib import Path
        import matplotlib.pyplot as plt
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(op), composite_img)
        logging.info(f"Saved composite image to {op}")
