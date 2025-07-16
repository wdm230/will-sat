import logging
import yaml
import ee
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
from pyproj import Transformer


from mlc_pipeline.utils import auto_utm_epsg, setup_logger, split_bbox
from mlc_pipeline.exclusion import ExclusionEditor
from mlc_pipeline.dem import DEMHandler
from mlc_pipeline.sentinel import SentinelHandler
from mlc_pipeline.classification import Classifier
from mlc_pipeline.meshing import MeshBuilder
from mlc_pipeline.curvimesh import CurviMeshBuilder
from mlc_pipeline.hotstart import HotstartBuilder
from mlc_pipeline.bc_maker import BCBuilder
from mlc_pipeline.metadata import MeshMetadata
from mlc_pipeline.bathymetry import Bathymetry
class MLCPipeline:
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path

        # Determine bbox (from shapefile or config)
        self.dem_handler = DEMHandler(self.config)
        self.bbox = self.dem_handler.bbox

        # Output directories
        self.output_dir = config.get("output_dir", "mlc_images")
        self.loc_tag = config.get("loc_tag", "river")
        self.subdir = Path(self.output_dir) / self.loc_tag
        self.subdir.mkdir(parents=True, exist_ok=True)

        # Earth Engine project
        self.project_name = config.get("project_name")

        # Projection
        self.target_epsg = auto_utm_epsg(self.bbox)

        # Flags
        self.curvi_enabled = config.get("meshing", {}).get("curvi", True)

        # Handlers
        self.sentinel_handler = SentinelHandler(self.config)
        self.classifier = Classifier(self.config.get("classification", {}))
        self.mesh_builder = MeshBuilder(self.config.get("meshing", {}))
        self.curvi_builder = CurviMeshBuilder(self.mesh_builder, self.config.get("meshing", {}))
        self.hotstart_builder = HotstartBuilder(self.config.get("hotstart", {}))
        self.bc_builder = BCBuilder(self.config.get("boundary", {}))

    def run(self):
        # Authenticate and init EE
        logging.info("Authenticating Earth Engine...")
        ee.Authenticate()
        logging.info("Initializing Earth Engine...")
        ee.Initialize(project=self.project_name)

        # --- Sentinel processing and tiling ---
        n = self.config.get("sentinel", {}).get("n_subboxes", 10)
        # If using shapefile, split its bbox; otherwise split the config bbox
        if self.config.get("shapefile", False):
            tiles = split_bbox(self.sentinel_handler.bbox, n)
        else:
            tiles = split_bbox(self.bbox, n)
        indexed = list(enumerate(tiles))

        sampled = self.sentinel_handler.download_subboxes(indexed)
        sampled.sort(key=lambda x: x[0])  # sort by tile index

        # Classify each tile and collect masks and their bboxes
        tile_masks = []
        tile_bboxes = []
        for idx, bb, arr, _, _ in sampled:
            raw_mask = self.classifier.classify(arr, output_dir=str(self.subdir))
            bin_mask = (raw_mask > 0).astype(np.uint8)
            tile_masks.append(bin_mask)
            tile_bboxes.append(bb)

        # Determine grid layout based on bbox origins
        minxs = sorted({bb[0] for bb in tile_bboxes})
        minys = sorted({bb[1] for bb in tile_bboxes})
        nx = len(minxs)
        ny = len(minys)

        # Assume all tiles have the same shape
        tile_h, tile_w = tile_masks[0].shape
        canvas_h = ny * tile_h
        canvas_w = nx * tile_w

        # Stitch masks into a full-size canvas
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for (idx, bb, *_), mask in zip(sampled, tile_masks):
            col = minxs.index(bb[0])
            row = (ny - 1) - minys.index(bb[1])  # flip y-axis index
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            canvas[y0:y1, x0:x1] = mask

        # Save and log the merged mask
        merged_path = self.subdir / "merged_mlc_mask.png"
        plt.imsave(str(merged_path), canvas, cmap='gray')
        logging.info(f"Merged and saved final MLC mask to {merged_path}")

        mlc_mask = canvas

        # Optional manual exclusion
        if self.config.get("exclusion", {}).get("enabled", False):
            editor = ExclusionEditor(mlc_mask)
            mlc_mask = editor.edit()
            excl_path = self.subdir / f"{self.loc_tag}_mask_excluded.png"
            plt.imsave(str(excl_path), mlc_mask, cmap='gray')
            logging.info(f"Saved excluded mask to {excl_path}")

        # --- DEM processing ---
        # Determine the resolution from the sampled tiles
        used_scale = min(scale for *_, scale in sampled)
        dem = self.dem_handler.get_dem(used_scale)
        dem_f = dem.astype(np.float32)
        dem_smooth = self.dem_handler.smooth_dem(dem_f)
        self.dem_handler.save_dem(dem, dem_smooth, output_dir=str(self.subdir))

        # Ensure DEM and mask have the same dimensions
        mask_h, mask_w = mlc_mask.shape
        dem_h, dem_w = dem_smooth.shape
        if (dem_h, dem_w) != (mask_h, mask_w):
            logging.info(f"Resizing DEM from {(dem_h, dem_w)} to {(mask_h, mask_w)} to align with mask")
            dem_smooth = cv2.resize(dem_smooth, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)

        # Compute average water elevation
        rows, cols = np.where(mlc_mask > 0)
        avg_level = float(dem_smooth[rows, cols].mean()) if rows.size else 0.0
        modified_dem = dem_smooth.copy()
        modified_dem[rows, cols] = avg_level

        # --- Mesh building ---
        adv_mesh, face_mats = self.mesh_builder.build_adv_front_mesh(mlc_mask, modified_dem)
        if self.curvi_enabled:
            # Build curvi mesh and capture logical dims
            mesh, face_mats = self.curvi_builder.build(adv_mesh, avg_level, mlc_mask)
            eta, nu = self.curvi_builder.eta, self.curvi_builder.nv
            shape_axis = self.curvi_builder.shape_axis
            # Generate custom bathymetry surface
            bt_cfg = self.config.get('bathymetry', {})
            bt = Bathymetry(bt_cfg)
            Z = bt.generate(eta, nu, avg_level)

            # Apply Z directly to mesh vertices without rebuilding
            verts = mesh.vertices.copy()
            if Z.size == verts.shape[0]:
                verts[:, 2] = Z.ravel()
            else:
                # if mesh vertices > grid nodes, assume ordering matches and trim/pad
                verts[:, 2] = np.resize(Z.ravel(), verts[:, 2].shape)
            mesh.vertices = verts
        else:
            mesh, face_mats = adv_mesh, face_mats
            eta, nu = mlc_mask.shape


        # Georeference and save mesh
        geo_mesh = (
            self.curvi_builder.georeference(mesh, self.bbox, mask_w, mask_h, self.target_epsg)
            if self.curvi_enabled else
            self.mesh_builder.georeference(mesh, self.bbox, mask_w, mask_h, self.target_epsg)
        )
        mesh_path = self.subdir / f"{self.loc_tag}.3dm"
        self.mesh_builder.save_mesh(geo_mesh, face_mats, str(mesh_path))


        # --- Save mesh plot ---
        verts = np.array([v for v in geo_mesh.vertices])
        faces = [f for f in geo_mesh.faces]
        fig, ax = plt.subplots()
        for f in faces:
            poly = verts[f]
            xs = np.append(poly[:,0], poly[0,0])
            ys = np.append(poly[:,1], poly[0,1])
            ax.plot(xs, ys, linewidth=0.1, color='black')
        ax.set_aspect('equal')
        ax.axis('on')
        plot_path = self.subdir / f"{self.loc_tag}_mesh.png"
        fig.savefig(str(plot_path), dpi=500, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved mesh plot to {plot_path}")

        fig, ax = plt.subplots(figsize=(8, 6))
        X = verts[:,0].reshape((eta, nu))
        Y = verts[:,1].reshape((eta, nu))
        cs = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)


        ax.set_aspect('equal')
        ax.axis('on')
        cbar = fig.colorbar(cs, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Depth / Elevation", rotation=270, labelpad=15)

        contour_path = self.subdir / f"{self.loc_tag}_bathymetry.png"
        fig.savefig(str(contour_path), dpi=500, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved bathymetry contour to {contour_path}")

        minx, miny, maxx, maxy = self.bbox
        true_rgb = self.sentinel_handler.download_s2_true_color(
            scale=used_scale,
            bbox=self.bbox,
            target_epsg=self.target_epsg,
            width=mask_w,
            height=mask_h
        )
        transformer = Transformer.from_crs("epsg:4326", self.target_epsg, always_xy=True)
        minx_t, miny_t = transformer.transform(minx, miny)
        maxx_t, maxy_t = transformer.transform(maxx, maxy)


        extent = (minx_t, maxx_t, miny_t, maxy_t)

        fig, ax = plt.subplots(figsize=(8,6))


        ax.imshow(
            true_rgb,
            origin='upper',    
            extent=extent,
            interpolation='nearest',
            zorder=0
        )

        cf = ax.contourf(
            X, Y, Z,
            levels=100,
            cmap='jet',
            alpha=1.0,
            zorder=1
        )

        ax.set_aspect('equal')
        ax.axis('on')

        fig.colorbar(cf, ax=ax, shrink=0.6, pad=0.02)
        fig.savefig(str(self.subdir/f"{self.loc_tag}_overlay_proj.png"), dpi=500,
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


        # --- Metadata ---
        meta = MeshMetadata(
            geo_mesh, epsg=self.target_epsg,
            loc_tag=self.loc_tag, project_name=self.project_name
        )
        meta_file = meta.write_txt(self.subdir)
        logging.info(f"Metadata written to {meta_file}")

        # --- Hotstart & BC ---
        if self.config.get("hotstart", {}).get("enabled", False):
            hot_path = self.subdir / f"{self.loc_tag}.hot"
            self.hotstart_builder.build(geo_mesh, hot_path)
            logging.info(f"Hotstart file generated: {hot_path}")
        if self.config.get("boundary", {}).get("enabled", False):
            bc_path = self.subdir / f"{self.loc_tag}.bc"
            self.bc_builder.build(geo_mesh, bc_path)
            logging.info(f"Boundary conditions written to {bc_path}")


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
