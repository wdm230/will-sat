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
from mlc_pipeline.hotstart import generate_hotstart

class MLCPipeline:
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path
        self.bbox = config["bbox"]
        self.output_dir = config.get("output_dir", "mlc_images")
        self.loc_tag = config.get("loc_tag", "river")
        self.subdir = Path(self.output_dir) / self.loc_tag
        self.subdir.mkdir(parents=True, exist_ok=True)
        self.project_name = config.get("project_name")
        self.target_epsg = auto_utm_epsg(self.bbox)
        self.dilation_iterations = config.get("dilation_iterations", 10)
        # Instantiate handlers.
        self.dem_handler = DEMHandler(self.bbox, config.get("dem", {}))
        self.sentinel_handler = SentinelHandler(self.bbox, config.get("sentinel", {}))
        self.classifier = Classifier(config.get("classification", {}))
        self.mesh_builder = MeshBuilder(config.get("meshing", {}))

    def run(self):
        logging.info("Authenticating Earth Engine (if required)...")
        ee.Authenticate()
        logging.info("Initializing Earth Engine...")
        ee.Initialize(project=self.project_name)

        # --- Sentinel processing ---
        n_subboxes = self.config.get("sentinel", {}).get("n_subboxes", 10)
        sub_bboxes = split_bbox(self.bbox, n_subboxes)
        sampled = self.sentinel_handler.download_subboxes(list(enumerate(sub_bboxes)))
        sampled.sort(key=lambda x: x[0])
        arrays = [img for (_, _, img, _) in sampled]
        min_h = min(arr.shape[0] for arr in arrays)
        merged = np.hstack([cv2.resize(arr, (arr.shape[1], min_h), interpolation=cv2.INTER_LANCZOS4)
                             for arr in arrays])
        used_scale = min(scale for (_, _, _, scale) in sampled)
        plt.imsave(str(self.subdir / "merged_sentinel_mndwi.png"), merged)

        # --- Classification ---
        mlc_mask = self.classifier.classify(merged, output_dir=str(self.subdir))

        # --- DEM processing ---
        dem = self.dem_handler.get_dem(used_scale)
        dem_f = dem.astype(np.float32)
        dem_smooth = self.dem_handler.smooth_dem(dem_f)
        self.dem_handler.save_dem(dem, dem_smooth, output_dir=str(self.subdir))

        # Align DEM to mask
        h_dem, w_dem = dem_smooth.shape
        h_mask, w_mask = mlc_mask.shape
        if (h_dem, w_dem) != (h_mask, w_mask):
            dem_smooth = dem_smooth[:h_mask, :w_mask]

        # Compute water level
        rows, cols = np.where(mlc_mask > 0)
        if rows.size:
            rows = np.clip(rows, 0, dem_smooth.shape[0] - 1)
            cols = np.clip(cols, 0, dem_smooth.shape[1] - 1)
            avg_level = float(dem_smooth[rows, cols].mean())
        else:
            avg_level = 0.0
        modified_dem = dem_smooth.copy()
        modified_dem[rows, cols] = avg_level

        # Shoreline mask
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shore_mask = cv2.dilate(mlc_mask, kern, iterations=self.dilation_iterations)

        # Build mesh
        adv_mesh, face_mats = self.mesh_builder.build_adv_front_mesh(shore_mask, modified_dem)
        geo_mesh = self.mesh_builder.georeference(adv_mesh, self.bbox, w_mask, h_mask, self.target_epsg)

        # Boundary reporting
        boundary_nodes = self.mesh_builder.get_boundary_nodes()
        boundary_elems = self.mesh_builder.get_boundary_elements(geo_mesh)
        logging.info(f"Boundary node indices: {boundary_nodes}")
        logging.info(f"Boundary element indices: {boundary_elems}")

        # Save mesh
        mesh_path = self.subdir / "mesh.3dm"
        self.mesh_builder.save_mesh(geo_mesh, face_mats, str(mesh_path))
        logging.info(f"Saved mesh to {mesh_path}")

        # Generate hotstart
        if self.config.get('hotstart', {}).get('enabled', False):
            hot_path = self.subdir / "hotstart.hot"
            logging.info("Generating hotstart file...")
            out_hot = generate_hotstart(self.config_path, hot_path)
            logging.info(f"Hotstart file created at {out_hot}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the MLC Pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--log", type=str, default="pipeline.log")
    args = parser.parse_args()
    setup_logger(log_file=args.log)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pipeline = MLCPipeline(cfg, args.config)
    pipeline.run()

if __name__ == "__main__":
    main()
