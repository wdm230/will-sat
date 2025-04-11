# mlc_pipeline/pipeline.py
import logging
import yaml
import ee
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

from mlc_pipeline.utils import auto_utm_epsg, setup_logger, split_bbox
from mlc_pipeline.dem import DEMHandler
from mlc_pipeline.sentinel import SentinelHandler
from mlc_pipeline.classification import Classifier
from mlc_pipeline.meshing import MeshBuilder

class MLCPipeline:
    def __init__(self, config):
        self.config = config
        self.bbox = config["bbox"]
        self.output_dir = config.get("output_dir", "mlc_images")
        self.loc_tag = config.get("loc_tag", "ms_river")
        self.subdir = Path(self.output_dir) / self.loc_tag
        self.subdir.mkdir(parents=True, exist_ok=True)
        
        self.target_epsg = auto_utm_epsg(self.bbox)

        # Instantiate handlers.
        self.dem_handler = DEMHandler(self.bbox, config.get("dem", {}))
        self.sentinel_handler = SentinelHandler(self.bbox, config.get("sentinel", {}))
        self.classifier = Classifier(config.get("classification", {}))
        self.mesh_builder = MeshBuilder(config.get("meshing", {}))

    def run(self):
        logging.info("Authenticating Earth Engine (if required)...")
        ee.Authenticate()  # Will prompt if needed.
        logging.info("Initializing Earth Engine...")
        ee.Initialize(project="water-seg-wdm230")

        # --- Download Sentinel composites for sub-boxes concurrently ---
        # In mlc_pipeline/pipeline.py, within MLCPipeline.run()

        # --- Download Sentinel composites for sub-boxes concurrently ---
        n_subboxes = self.config.get("sentinel", {}).get("n_subboxes", 10)
        logging.info(f"Splitting bbox into {n_subboxes} sub-boxes for concurrent Sentinel download...")
        sub_bboxes = split_bbox(self.bbox, n_subboxes)
        
        # Create an enumerated list of sub-boxes: each element is (index, sub_bbox)
        indexed_sub_bboxes = list(enumerate(sub_bboxes))
        
        # Note: the SentinelHandler.download_subboxes method is updated below to accept this list.
        sampled_subimages = self.sentinel_handler.download_subboxes(indexed_sub_bboxes)
        if not sampled_subimages:
            logging.error("No subimages were downloaded successfully.")
            raise RuntimeError("No subimages were downloaded successfully.")
        
        # Sort subimages by their original index.
        sampled_subimages.sort(key=lambda tup: tup[0])
        # Extract the image arrays (the tuple structure is: (index, sub_bbox, img_array, used_scale))
        arrays = [img_array for (_, _, img_array, _) in sampled_subimages]
        
        min_height = min(arr.shape[0] for arr in arrays)
        resized_arrays = [
            cv2.resize(arr, (arr.shape[1], min_height), interpolation=cv2.INTER_LANCZOS4)
            for arr in arrays
        ]
        merged_image = np.hstack(resized_arrays)
        used_scale = min([scale for (_, _, _, scale) in sampled_subimages])
        composite_path = self.subdir / "merged_sentinel_mndwi.png"
        plt.imsave(str(composite_path), merged_image)
        logging.info(f"Saved merged Sentinel composite to {composite_path}")

        # --- Maximum Likelihood Classification ---
        logging.info("Performing Maximum Likelihood Classification (MLC)...")
        mlc_mask = self.classifier.classify(merged_image, output_dir=str(self.subdir))
        
        # --- DEM retrieval and processing ---
        logging.info("Retrieving DEM tiles...")
        dem_data = self.dem_handler.get_dem(used_scale)
        dem_data_float = dem_data.astype(np.float32)
        dem_data_smooth = self.dem_handler.smooth_dem(dem_data_float)
        self.dem_handler.save_dem(dem_data, dem_data_smooth, output_dir=str(self.subdir))
        logging.info(f"DEM data type: {dem_data_smooth.dtype}")

        water_pixels = dem_data_smooth[mlc_mask > 0]
        if water_pixels.size > 0:
            avg_water_level = np.min(water_pixels)
        else:
            logging.warning("No water pixels found in DEM for average water level. Using 0.")
            avg_water_level = 0
        logging.info(f"Average water elevation computed: {avg_water_level}")
        
        # 2. Create a modified DEM: Set DEM values inside the water mask to the average water level.
        modified_dem = dem_data_smooth.copy()
        modified_dem[mlc_mask > 0] = avg_water_level
        
        # 3. Dilate the water mask to include additional shoreline.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shoreline_dilation_iterations = self.config.get("shoreline_dilation_iterations", 20)
        shoreline_mask = cv2.dilate(mlc_mask, kernel, iterations=shoreline_dilation_iterations)
        
        # 4. Use the expanded shoreline mask and the modified DEM for mesh building.
        logging.info("Building advanced-front mesh with extra shoreline information...")
        adv_mesh, face_materials = self.mesh_builder.build_adv_front_mesh(shoreline_mask, modified_dem)

        # --- Georeference mesh ---
        logging.info("Georeferencing mesh...")
        # Use the original mask dimensions (mlc_mask) for width and height.
        mask_height, mask_width = mlc_mask.shape
        geo_mesh = self.mesh_builder.georeference(adv_mesh, self.bbox, mask_width, mask_height, self.target_epsg)
        
        # --- Save the final mesh ---
        mesh_path = self.subdir / "mesh.3dm"
        self.mesh_builder.save_mesh(geo_mesh, face_materials, str(mesh_path))
        logging.info(f"Saved final mesh to {mesh_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the MLC Pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config YAML file.")
    parser.add_argument("--log", type=str, default="pipeline.log", help="Path to the log file.")
    args = parser.parse_args()
    setup_logger(log_file=args.log)
    logging.info("Loading configuration...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    pipeline = MLCPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
