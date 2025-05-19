import logging
import numpy as np
import pygridgen
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path


class SimpleMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

class CurviMeshBuilder:
    def __init__(self, mesh_builder, config):
        self.mbuilder = mesh_builder
        self.ni = config.get("ni", 20)
        self.nj = config.get("nj", 200)
        logging.info(f"CurviMeshBuilder initialized with ni={self.ni}, nj={self.nj}")

    def _pick_corners(self, contour, mask_shape):
        """
        Locate the four image-corner points by closest distance to the exact border corners,
        then sort them clockwise.
        contour: (M+1,2) closed loop array
        mask_shape: (height, width)
        returns: list of 4 indices in contour (without final duplicate) in CW order
        """
        h, w = mask_shape
        pts = contour[:-1]
        exact = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
        corners = []
        for (x0,y0) in exact:
            d = np.hypot(pts[:,0]-x0, pts[:,1]-y0)
            idx = int(np.argmin(d))
            if d[idx] > 1e-3:
                logging.warning(f"_pick_corners: using approximate corner at {pts[idx]} for target ({x0},{y0}), distance {d[idx]:.3f}")
            corners.append(idx)
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[corners,1]-centroid[1], pts[corners,0]-centroid[0])
        order = np.argsort(angles)
        sorted_corners = [corners[i] for i in order]
        logging.info(f"Picked corner indices (approx): {sorted_corners}")
        return sorted_corners

    def _compute_shape(self, contour, corners):
        deltas = np.diff(contour, axis=0)
        seg_lengths = np.hypot(deltas[:,0], deltas[:,1])
        cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
        lengths = []
        for i in range(4):
            i0 = corners[i]
            i1 = corners[(i+1)%4]
            if i1 >= i0:
                length = cumlen[i1] - cumlen[i0]
            else:
                length = (cumlen[-1] - cumlen[i0]) + cumlen[i1]
            lengths.append(length)
        logging.info(f"Side lengths: {lengths}")
        len_u = 0.5*(lengths[0] + lengths[2])
        len_v = 0.5*(lengths[1] + lengths[3])
        nu = self.ni if len_u < len_v else self.nj
        nv = self.nj if len_u < len_v else self.ni
        return nv, nu

    def build(self, adv_mesh, avg_z, mask_shape):
        """
        Build a curvilinear mesh along the water contour only, ignoring loops on image borders.
        """
        # 1) Extract all boundary loops in pixel-space
        loops = self.mbuilder.get_boundary_loops(adv_mesh, mask_shape)
        h, w = mask_shape
        # 2) Filter out loops touching the image border
        interior = []
        for loop, mats in loops:
            coords = adv_mesh.vertices[loop, :2]
            # keep only loops with all points strictly inside borders
            if np.all((coords[:,0] > 0) & (coords[:,0] < w-1) &
                      (coords[:,1] > 0) & (coords[:,1] < h-1)):
                interior.append((loop, mats))

        # pick the longest interior loop
        ext_loop, face_mats = max(interior, key=lambda lm: len(lm[0]))

        # 3) Build closed contour array
        contour = adv_mesh.vertices[ext_loop, :2]
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        # Pick corners by proximity to image corners
        corners = self._pick_corners(contour, mask_shape)
        beta = np.zeros(len(contour), dtype=float)
        beta[corners] = 1.0
        beta *= 4.0 / beta.sum()
        
        plt.figure(figsize=(6,6))
        plt.plot(contour[:,0], contour[:,1], '-o', markersize=2)
        # highlight corners in red
        corner_pts = contour[corners]
        plt.plot(corner_pts[:,0], corner_pts[:,1], 'ro', markersize=5)
        plt.axis('equal')
        plt.title("Boundary Contour fed to pygridgen (corners in red)")
        plt.savefig("boundary_debug.png", dpi=150)
        plt.close()

        
        # Compute grid shape and swap axes for pygridgen
        nv, nu = self._compute_shape(contour, corners)
        shape = (nu, nv)
        logging.info(f"Calling pygridgen with shape (nx,ny): {shape}")

        # Centre coords to avoid overflow
        centroid = contour.mean(axis=0)
        pts = contour - centroid

        # Build curvilinear grid
        grid = pygridgen.Gridgen(pts[:,0], pts[:,1], beta, shape=shape)
        X = grid.x + centroid[0]
        Y = grid.y + centroid[1]
        logging.info(f"Generated grid min/max X: {X.min()}/{X.max()}, Y: {Y.min()}/{Y.max()}")

        # Handle Z (scalar or array)
        if np.isscalar(avg_z):
            Z = np.full_like(X, avg_z, dtype=float)
        else:
            Z = avg_z

        # Assemble vertices and triangulate quads
        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        nx, ny = shape
        faces = []
        for i in range(ny-1):
            for j in range(nx-1):
                n0 = i*nx + j
                n1 = n0 + 1
                n2 = n0 + nx
                n3 = n2 + 1
                faces.append([n0, n2, n1])
                faces.append([n1, n2, n3])
        faces = np.array(faces, dtype=int)

        mesh = SimpleMesh(vertices=verts, faces=faces)
        mats = np.ones(len(faces), dtype=int)
        logging.info("CurviMeshBuilder.build: build complete")
        return mesh, mats

    def georeference(self, mesh, bbox, width, height, epsg):
        logging.info("CurviMeshBuilder.georeference: applying georeference")
        return self.mbuilder.georeference(mesh, bbox, width, height, epsg)

    def save(self, mesh, face_mats, path):
        logging.info(f"CurviMeshBuilder.save: saving mesh to {path}")
        ext = Path(path).suffix.lower()
        if ext in ['.stl', '.obj']:
            tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
            tri.export(path)
            logging.info(f"Exported mesh to {path}")
        else:
            self.mbuilder.save_mesh(mesh, face_mats, path)
            logging.info(f"Saved mesh to {path}")
