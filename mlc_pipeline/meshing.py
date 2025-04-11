# mlc_pipeline/meshing.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import triangle
from shapely.geometry import Polygon
import logging
from pyproj import Transformer, CRS as projCRS

class MeshBuilder:
    def __init__(self, config):
        self.size = config.get("size", 50)

    def extract_corners(self, contour, epsilon_factor=None):
        """
        Extract corners using polygon approximation.
        contour: an Nx1x2 (or Nx2) array of contour points.
        epsilon_factor: fraction of the contour perimeter to use as approximation accuracy.
                        Defaults to self.epsilon_factor.
        Returns an Nx2 array with the corner points.
        """
        if epsilon_factor is None:
            epsilon_factor = self.epsilon_factor
        if contour.ndim == 2:
            contour = contour.reshape((-1, 1, 2))
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = approx.squeeze()
        if corners.ndim == 1:
            corners = corners[np.newaxis, :]
        return corners

    def preserve_boundary_points(self, approximated, mask_shape, tol=1e-3):
        """
        Adjust approximated corner points if they are close to the mask boundary.
        For each approximated point, if it is within tol of a boundary (0 or width-1/height-1),
        set it exactly to the boundary coordinate.
        """
        h, w = mask_shape[:2]
        corrected = approximated.copy()
        for i in range(len(approximated)):
            x, y = approximated[i]
            if abs(x) < tol:
                corrected[i, 0] = 0
            if abs(x - (w - 1)) < tol:
                corrected[i, 0] = w - 1
            if abs(y) < tol:
                corrected[i, 1] = 0
            if abs(y - (h - 1)) < tol:
                corrected[i, 1] = h - 1
        return corrected

    def save_debug_contour(self, binary_mask, contour, output_path):
        """
        Saves a debug image showing the binary mask with the given contour overlaid.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_mask, cmap="gray")
        if contour.ndim != 2:
            contour = contour.reshape((-1, 2))
        plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, label="Extracted Contour")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.title("Extracted Outer Contour")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.savefig(output_path, dpi=300)
        plt.close()
        logging.info(f"Debug contour image saved to {output_path}")

    def build_adv_front_mesh(self, binary_mask, dem_data, size=50):
        mask_bin = (binary_mask > 0).astype(np.uint8)
        cross_kernel = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]], dtype=np.uint8)
        mask_bin = cv2.dilate(mask_bin, cross_kernel, iterations=2)
        
        contours, hierarchy = cv2.findContours(mask_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            raise ValueError("No contours found in the mask!")
        hierarchy = hierarchy[0]
        outer_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]
        if not outer_indices:
            raise ValueError("No outer contour found!")
        largest_idx = max(outer_indices, key=lambda i: cv2.contourArea(contours[i]))
        
        outer_contour = contours[largest_idx].squeeze()
        if outer_contour.ndim != 2:
            raise ValueError("Contour extraction failed.")
        if not np.array_equal(outer_contour[0], outer_contour[-1]):
            outer_contour = np.vstack([outer_contour, outer_contour[0]])
        
        # Instead of extracting approximated corners and saving debug plots,
        # we simply use the outer_contour directly.
        exterior = outer_contour.tolist()

        # Process holes as in the original code.
        def get_child_indices(parent_idx, hierarchy):
            children = []
            child_idx = hierarchy[parent_idx][2]
            while child_idx != -1:
                children.append(child_idx)
                child_idx = hierarchy[child_idx][0]
            return children
        child_indices = get_child_indices(largest_idx, hierarchy)
        
        holes_coords = []
        for idx in child_indices:
            cnt = contours[idx].squeeze()
            if cnt.ndim != 2:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if mask_bin[cy, cx] == 0:
                if not np.array_equal(cnt[0], cnt[-1]):
                    cnt = np.vstack([cnt, cnt[0]])
                # For holes, you can choose to extract corners or use the raw contour.
                # Here, we use the raw contour.
                holes_coords.append(cnt)
        
        poly_kwargs = {}
        if holes_coords:
            poly_kwargs['holes'] = [hole.tolist() for hole in holes_coords]
        water_poly = Polygon(exterior, **poly_kwargs)
        if not water_poly.is_valid:
            water_poly = water_poly.buffer(0)
        if water_poly.geom_type == 'MultiPolygon':
            water_poly = max(water_poly.geoms, key=lambda p: p.area)
        
        def polygon_to_pslg_with_holes(polygon):
            exterior_coords = np.array(polygon.exterior.coords[:-1], dtype=np.float64)
            vertices = exterior_coords.copy()
            n_exterior = len(exterior_coords)
            segments = np.array([[i, (i + 1) % n_exterior] for i in range(n_exterior)], dtype=np.int32)
            holes = []
            offset = n_exterior
            for interior in polygon.interiors:
                inter_coords = np.array(interior.coords[:-1], dtype=np.float64)
                n_interior = len(inter_coords)
                if n_interior < 3:
                    continue
                vertices = np.vstack([vertices, inter_coords])
                inter_segments = np.array([[offset + i, offset + ((i + 1) % n_interior)] for i in range(n_interior)], dtype=np.int32)
                segments = np.vstack([segments, inter_segments])
                interior_poly = Polygon(inter_coords)
                hole_pt = interior_poly.representative_point()
                holes.append([hole_pt.x, hole_pt.y])
                offset += n_interior
            return vertices, segments, holes
        
        vertices, segments, holes = polygon_to_pslg_with_holes(water_poly)
        pslg = dict(vertices=vertices, segments=segments, holes=holes)
        triangulation_opts = f'pq30a{self.size}'
        triangulation = triangle.triangulate(pslg, triangulation_opts)
        mesh_vertices = triangulation.get('vertices', np.empty((0, 2), dtype=np.float64))
        mesh_triangles = triangulation.get('triangles', np.empty((0, 3), dtype=np.int32))
        
        vertices_3d = []
        H_dem, W_dem = dem_data.shape
        for v in mesh_vertices:
            x, y = v[0], v[1]
            i = int(round(y))
            j = int(round(x))
            i = min(max(i, 0), H_dem - 1)
            j = min(max(j, 0), W_dem - 1)
            vertices_3d.append([x, y, dem_data[i, j]])
        vertices_3d = np.array(vertices_3d)
        
        class SimpleMesh:
            def __init__(self, vertices, faces):
                self.vertices = vertices
                self.faces = faces
        
        adv_mesh = SimpleMesh(vertices_3d, mesh_triangles)
        face_materials = np.ones(mesh_triangles.shape[0], dtype=int)
        
        # (Optional: Remove or comment out this visualization if not needed)
        mask_vis = (mask_bin * 255).astype('uint8')
        contours_vis, _ = cv2.findContours(mask_vis.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_rgb, contours_vis, -1, (0, 0, 255), 2)
        plt.figure(figsize=(8, 8))
        plt.title("Mask with Contours")
        plt.axis("off")
        plt.close()
        
        return adv_mesh, face_materials


    def georeference(self, mesh, bbox, width, height, target_epsg):
        logging.info("Georeferencing mesh...")
        vertices = mesh.vertices.copy()
        vertices[:, 1] = (height - 1) - vertices[:, 1]
        min_lon, min_lat, max_lon, max_lat = bbox
        vertices[:, 0] = min_lon + (vertices[:, 0] / (width - 1)) * (max_lon - min_lon)
        vertices[:, 1] = min_lat + (vertices[:, 1] / (height - 1)) * (max_lat - min_lat)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        new_x, new_y = transformer.transform(vertices[:, 0], vertices[:, 1])
        vertices[:, 0] = new_x
        vertices[:, 1] = new_y
        mesh.vertices = vertices

        crs_target = projCRS.from_epsg(target_epsg)
        unit_name = crs_target.axis_info[0].unit_name.lower()
        if 'foot' in unit_name:
            mesh.vertices[:, 2] *= 3.28084
        return mesh

    def save_mesh(self, mesh, face_materials, output_path):
        with open(output_path, "w") as f:
            f.write("MESH2D\n")
            f.write("NUM_MATERIALS_PER_ELEM 1\n")
            for node_id, vertex in enumerate(mesh.vertices, start=1):
                f.write("ND {} {:.8e} {:.8e} {:.8e}\n".format(node_id, vertex[0], vertex[1], vertex[2]))
            for elem_id, (face, mat) in enumerate(zip(mesh.faces, face_materials), start=1):
                n1, n2, n3 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write("E3T {} {} {} {} {}\n".format(elem_id, n1, n2, n3, mat))
        logging.info(f"Saved mesh to {output_path}")
