# mlc_pipeline/metadata.py

import os
import numpy as np
from datetime import datetime

class MeshMetadata:
    """
    Compute and write mesh stats: counts, area, angles, aspect ratios,
    plus EPSG, bbox, elevation stats, timestamp, loc_tag, project_name.
    """

    def __init__(self, mesh, epsg=None, loc_tag=None, project_name=None):
        self.vertices = mesh.vertices   # n×3 array
        self.faces    = mesh.faces      # m×3 array
        self.epsg            = epsg
        self.loc_tag         = loc_tag
        self.project_name    = project_name
        self.generated_at    = datetime.now()
        # compute everything
        self._compute_stats()
        self._compute_bbox()
        self._compute_z_stats()

    def _compute_stats(self):
        v = self.vertices
        f = self.faces
        pts = v[f]  # (n_elems, 3, 3)
        e0 = pts[:,1,:] - pts[:,0,:]
        e1 = pts[:,2,:] - pts[:,1,:]
        e2 = pts[:,0,:] - pts[:,2,:]
        # edge lengths in XY plane
        l0 = np.linalg.norm(e0[:,:2], axis=1)
        l1 = np.linalg.norm(e1[:,:2], axis=1)
        l2 = np.linalg.norm(e2[:,:2], axis=1)
        # areas
        cross = np.cross(e0, e2)
        self.areas = 0.5 * np.linalg.norm(cross, axis=1)
        # angles via law of cosines
        cos0 = np.clip((l0**2 + l2**2 - l1**2)/(2*l0*l2), -1,1)
        cos1 = np.clip((l0**2 + l1**2 - l2**2)/(2*l0*l1), -1,1)
        cos2 = np.clip((l1**2 + l2**2 - l0**2)/(2*l1*l2), -1,1)
        angles = np.vstack((np.arccos(cos0), np.arccos(cos1), np.arccos(cos2))).T
        self.angles = angles  # radians
        # aspect ratios
        lens = np.vstack((l0, l1, l2))
        self.aspect_ratios = np.max(lens,axis=0)/np.min(lens,axis=0)

        self.n_nodes = v.shape[0]
        self.n_elems = f.shape[0]

    def _compute_bbox(self):
        xs = self.vertices[:,0]
        ys = self.vertices[:,1]
        self.bbox_min_x = float(xs.min()); self.bbox_max_x = float(xs.max())
        self.bbox_min_y = float(ys.min()); self.bbox_max_y = float(ys.max())
        self.bbox_area = (self.bbox_max_x - self.bbox_min_x) * (self.bbox_max_y - self.bbox_min_y)

    def _compute_z_stats(self):
        zs = self.vertices[:,2]
        self.z_min  = float(zs.min())
        self.z_max  = float(zs.max())
        self.z_mean = float(zs.mean())

    def area_stats(self):
        return {
            "total": float(self.areas.sum()),
            "min":   float(self.areas.min()),
            "max":   float(self.areas.max()),
            "mean":  float(self.areas.mean()),
        }

    def angle_stats(self):
        deg = np.degrees(self.angles)
        return {
            "min":  float(deg.min()),
            "max":  float(deg.max()),
            "mean": float(deg.mean()),
        }

    def aspect_ratio_stats(self):
        return {
            "min":  float(self.aspect_ratios.min()),
            "max":  float(self.aspect_ratios.max()),
            "mean": float(self.aspect_ratios.mean()),
        }

    def write_txt(self, out_path):
        """
        Write a human-readable metadata file.
        out_path can be a directory or a full filepath.
        """
        # decide filepath
        if os.path.isdir(out_path):
            path = os.path.join(out_path, "mesh_metadata.txt")
        else:
            path = out_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w") as f:
            f.write("Mesh Metadata\n")
            f.write("=============\n")
            f.write(f"Generated at:    {self.generated_at.isoformat()}\n")
            if self.project_name:
                f.write(f"Project name:    {self.project_name}\n")
            if self.loc_tag:
                f.write(f"Location tag:    {self.loc_tag}\n")
            if self.epsg:
                f.write(f"EPSG code:       EPSG:{self.epsg}\n")
            f.write("\n")

            f.write("Bounding box (projected)\n")
            f.write(f"  X min: {self.bbox_min_x}\n")
            f.write(f"  X max: {self.bbox_max_x}\n")
            f.write(f"  Y min: {self.bbox_min_y}\n")
            f.write(f"  Y max: {self.bbox_max_y}\n")
            f.write(f"  Area : {self.bbox_area}\n\n")

            f.write("Elevation (Z)\n")
            f.write(f"  Min  : {self.z_min}\n")
            f.write(f"  Max  : {self.z_max}\n")
            f.write(f"  Mean : {self.z_mean}\n\n")

            f.write(f"Number of nodes   : {self.n_nodes}\n")
            f.write(f"Number of elements: {self.n_elems}\n\n")

            a = self.area_stats()
            f.write("Element area (units²)\n")
            for k in ["total","min","max","mean"]:
                f.write(f"  {k.title():<5}: {a[k]}\n")
            f.write("\n")

            ang = self.angle_stats()
            f.write("Triangle angles (degrees)\n")
            for k in ["min","max","mean"]:
                f.write(f"  {k.title():<5}: {ang[k]}\n")
            f.write("\n")

            ar = self.aspect_ratio_stats()
            f.write("Aspect ratio (longest/shortest)\n")
            for k in ["min","max","mean"]:
                f.write(f"  {k.title():<5}: {ar[k]}\n")

        return path
