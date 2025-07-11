import cv2
import numpy as np
import triangle as tr
import logging
from shapely.geometry import Polygon
from pyproj import Transformer, CRS as projCRS
from collections import Counter


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
        
    def get_boundary_loops(self, mesh, mask_shape, tol=None):
        """
        Returns a list of (node_index_list, None) for each connected
        chain of mesh-boundary edges that touch the image border.
        Nodes are ordered in the actual edge-connectivity sequence.
        """
        if tol is None:
            tol = self.boundary_tol
        H, W = mask_shape
        
        # 1) find which vertices lie on the image edge
        verts = mesh.vertices[:, :2]
        xs, ys    = verts[:,0], verts[:,1]
        on_border = set(np.nonzero(
            (xs <= tol) |
            (xs >= (W-1)-tol) |
            (ys <= tol) |
            (ys >= (H-1)-tol)
        )[0])
        
        # 2) gather mesh edges (from triangles)
        edge_list = []
        for tri in mesh.faces:
            # tri is [i0,i1,i2]
            for a,b in ((0,1),(1,2),(2,0)):
                i,j = tri[a], tri[b]
                edge_list.append(tuple(sorted((i,j))))
        # count them
        edge_count = Counter(edge_list)
        
        # 3) pick only those edges that:
        #    a) are boundary‐edges in the mesh (count == 1)
        #    b) both endpoints are on the image border
        boundary_edges = [
            (i,j) for (i,j),c in edge_count.items()
            if c==1 and i in on_border and j in on_border
        ]
        
        # 4) build adjacency
        adj = {}
        for i,j in boundary_edges:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        
        # 5) find connected components and walk each one
        loops = []
        visited = set()
        for start in adj:
            if start in visited:
                continue
        
            # collect component
            comp = set()
            stack = [start]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                for v in adj[u]:
                    if v not in comp:
                        stack.append(v)
            visited |= comp
        
            # pick a chain – open if possible, else closed
            ends = [u for u in comp if len(adj[u]) == 1]
            root = ends[0] if ends else next(iter(comp))
        
            # traverse
            seq = [root]
            prev, cur = None, root
            while True:
                nbrs = [n for n in adj[cur] if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                seq.append(nxt)
                prev, cur = cur, nxt
                # stop if we closed the loop
                if cur == root:
                    break
        
            # only keep chains of length ≥3, and strip off first & last
            if len(seq) >= 3:
                trimmed = seq
                # you still need at least two nodes to emit any EGS
                if len(trimmed) >= 2:
                    loops.append((trimmed, None))
        
        return loops
