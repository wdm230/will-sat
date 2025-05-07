import cv2
import numpy as np
import triangle as tr
import logging
from shapely.geometry import Polygon
from pyproj import Transformer, CRS as projCRS

class MeshBuilder:
    def __init__(self, config):
        self.size = config.get("size", 50)
        self.dilation_iterations = config.get("dilation_iterations", 2)
        self.smoothing_iterations = config.get("smoothing_iterations", 2)
        self.boundary_tol = config.get("boundary_tol", 10)
        self.chaikin_alpha = config.get("chaikin_alpha", 0.25)
        self.resample_len_frac = config.get("resample_len_frac", 0.15)
        
    def preserve_boundary_points(self, contour, mask_shape, tol=None):
        if tol is None:
            tol = self.boundary_tol
        h, w = mask_shape
        out = contour.copy()
        for i, (x, y) in enumerate(contour):
            if abs(x) < tol:
                out[i, 0] = 0
            if abs(x - (w - 1)) < tol:
                out[i, 0] = w - 1
            if abs(y) < tol:
                out[i, 1] = 0
            if abs(y - (h - 1)) < tol:
                out[i, 1] = h - 1
        return out

    def resample_contour(self, contour, target_spacing):
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        deltas = np.diff(contour, axis=0)
        seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
        total_len = cumlen[-1]
        distances = np.arange(0, total_len, target_spacing)
        xs = np.interp(distances, cumlen, contour[:, 0])
        ys = np.interp(distances, cumlen, contour[:, 1])
        resamp = np.vstack([xs, ys]).T
        if not np.allclose(resamp[0], resamp[-1]):
            resamp = np.vstack([resamp, resamp[0]])
        return resamp

    def chaikin_smoothing(self, contour, iterations=1):
        alpha = self.chaikin_alpha
        target_spacing = self.size * self.resample_len_frac

        # ensure closed
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        for _ in range(iterations):
            new = []
            for p0, p1 in zip(contour, contour[1:]):
                Q = (1-alpha)*p0 + alpha*p1
                R = alpha*p0 + (1-alpha)*p1
                new.append(Q); new.append(R)
            # re-close
            new = np.vstack(new)
            if not np.allclose(new[0], new[-1]):
                new = np.vstack([new, new[0]])
            # re-resample so points stay evenly spaced
            contour = self.resample_contour(new, target_spacing)

        return contour

    def polygon_to_pslg_with_holes(self, polygon: Polygon):
        ext = np.array(polygon.exterior.coords[:-1], dtype=float)
        vertices = ext.copy()
        n_ext = len(ext)
        segments = [[i, (i + 1) % n_ext] for i in range(n_ext)]
        holes = []
        offset = n_ext
        for interior in polygon.interiors:
            ic = np.array(interior.coords[:-1], dtype=float)
            if len(ic) < 3:
                continue
            vertices = np.vstack([vertices, ic])
            n_ic = len(ic)
            for i in range(n_ic):
                segments.append([offset + i, offset + ((i + 1) % n_ic)])
            holes.append(list(Polygon(ic).representative_point().coords)[0])
            offset += n_ic
        return vertices, np.array(segments, dtype=int), np.array(holes, dtype=float) if holes else None

    def build_adv_front_mesh(self, binary_mask, dem_data):
        mask_bin = (binary_mask > 0).astype(np.uint8)
        mask_shape = binary_mask.shape[:2]
        if self.dilation_iterations > 0:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_bin = cv2.dilate(mask_bin, kern, iterations=self.dilation_iterations)
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            raise ValueError("No contours found in mask")
        hierarchy = hierarchy[0]
        outer = max([i for i, h in enumerate(hierarchy) if h[3] == -1], key=lambda i: cv2.contourArea(contours[i]))
        ext = contours[outer].squeeze().astype(float)
        if not np.array_equal(ext[0], ext[-1]):
            ext = np.vstack([ext, ext[0]])
        if self.smoothing_iterations > 0:
            ext = self.chaikin_smoothing(ext, iterations=self.smoothing_iterations)
        ext = self.resample_contour(ext, target_spacing=self.size * 0.15)
        ext = self.preserve_boundary_points(ext, mask_shape)
        def children(idx):
            c = hierarchy[idx][2]
            while c != -1:
                yield c; c = hierarchy[c][0]
            # after you've defined `children` and before building the Polygon…

        hole_loops = []
        for cid in children(outer):
            loop = contours[cid].squeeze().astype(float)
            if not np.allclose(loop[0], loop[-1]):
                loop = np.vstack([loop, loop[0]])
    
            # — apply the same smoothing pipeline to the interior loop —
            if self.smoothing_iterations > 0:
                loop = self.chaikin_smoothing(loop,
                                              iterations=self.smoothing_iterations)
            loop = self.resample_contour(loop,
                                         target_spacing=self.size * self.resample_len_frac)
            loop = self.preserve_boundary_points(loop, mask_shape)
    
            hole_loops.append(loop)

        poly = Polygon(ext.tolist(), [hl.tolist() for hl in hole_loops])
        if not poly.is_valid:
            poly = poly.buffer(0)
        verts2d, segs, holes = self.polygon_to_pslg_with_holes(poly)
        pslg = {'vertices': verts2d, 'segments': segs}
        if holes is not None:
            pslg['holes'] = holes
        opts = f"pqa{self.size}q30Djuz"
        tri = tr.triangulate(pslg, opts)
        v2d = tri['vertices']; tri_faces = tri['triangles']
        avg_z = float(np.mean(dem_data))
        verts3d = np.hstack([v2d, np.full((v2d.shape[0], 1), avg_z)])
        class SimpleMesh:
            def __init__(self, vertices, faces):
                self.vertices = vertices
                self.faces = faces
        mesh = SimpleMesh(verts3d, tri_faces)
        # store boundary segments for BC/hotstart
        self.boundary_segments = segs
        return mesh, np.ones(len(tri_faces), dtype=int)

    def georeference(self, mesh, bbox, width, height, target_epsg):
        logging.info("Georeferencing mesh...")
        verts = mesh.vertices.copy()
        verts[:, 1] = (height - 1) - verts[:, 1]
        min_lon, min_lat, max_lon, max_lat = bbox
        verts[:, 0] = min_lon + (verts[:, 0] / (width - 1)) * (max_lon - min_lon)
        verts[:, 1] = min_lat + (verts[:, 1] / (height - 1)) * (max_lat - min_lat)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        x2, y2 = transformer.transform(verts[:, 0], verts[:, 1])
        verts[:, 0], verts[:, 1] = x2, y2
        crs_target = projCRS.from_epsg(target_epsg)
        if 'foot' in crs_target.axis_info[0].unit_name.lower():
            verts[:, 2] *= 3.28084
        mesh.vertices = verts
        return mesh

    def save_mesh(self, mesh, face_materials, output_path):
        with open(output_path, 'w') as f:
            f.write("MESH2D\nNUM_MATERIALS_PER_ELEM 1\n")
            for i, v in enumerate(mesh.vertices, start=1):
                f.write(f"ND {i} {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            for eid, (face, mat) in enumerate(zip(mesh.faces, face_materials), start=1):
                n1, n2, n3 = face + 1
                f.write(f"E3T {eid} {n1} {n2} {n3} {mat}\n")
        logging.info(f"Saved mesh to {output_path}")
        
    def get_boundary_loops(self, mesh, mask_shape, tol=None, gap_tol=None):
        """
        Return a list of (node_list, None) for each contiguous stretch of
        vertices touching any of the four image edges.  If the mask
        touches the same side in multiple disjoint parts, each becomes
        its own loop.
    
        - tol: how close to x=0/W-1 or y=0/H-1 counts as “on the edge”
        - gap_tol: max allowed gap between successive sorted coords before
                   splitting into a new loop.  Defaults to ~1.5×your resample spacing.
        """
        if tol is None:
            tol = self.boundary_tol
        H, W = mask_shape
    
        # pick out the x,y coords
        verts = mesh.vertices[:, :2]
        xs, ys = verts[:,0], verts[:,1]
    
        # default gap tolerance ≈ 1.5× your contour resample spacing
        if gap_tol is None:
            gap_tol = self.size * self.resample_len_frac * 1.5
    
        loops = []
        # for each side: (name, boolean‐mask, sorter, which coord to check)
        sides = [
            ('left',   xs <=     tol, lambda idx: idx[np.argsort(ys[idx])], ys),
            ('bottom', ys >= (H-1)-tol, lambda idx: idx[np.argsort(xs[idx])], xs),
            ('right',  xs >= (W-1)-tol, lambda idx: idx[np.argsort(-ys[idx])], ys),
            ('top',    ys <=     tol, lambda idx: idx[np.argsort(-xs[idx])], xs),
        ]
    
        for side, m, sorter, coord in sides:
            idxs = np.nonzero(m)[0]
            if idxs.size < 2:
                continue
            sorted_idxs = sorter(idxs)
    
            # split into clusters whenever gap > gap_tol
            clusters = []
            current = [sorted_idxs[0]]
            for prev, cur in zip(sorted_idxs, sorted_idxs[1:]):
                if abs(coord[cur] - coord[prev]) <= gap_tol:
                    current.append(cur)
                else:
                    if len(current) >= 2:
                        clusters.append(current)
                    current = [cur]
            if len(current) >= 2:
                clusters.append(current)
    
            # record each run as its own loop
            for c in clusters:
                loops.append((c, None))
    
        return loops
