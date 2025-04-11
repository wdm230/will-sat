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
        self.keep_fraction = config.get("keep_fraction", 0.15)



    def smooth_contour_fourier(self, contour, keep_fraction=None):
        if keep_fraction is None:
            keep_fraction = self.keep_fraction
        N = len(contour)
        complex_contour = contour[:, 0] + 1j * contour[:, 1]
        fft_coeff = np.fft.fft(complex_contour)
        cutoff = int(np.ceil(keep_fraction * N))
        fft_filtered = np.zeros_like(fft_coeff)
        fft_filtered[:cutoff] = fft_coeff[:cutoff]
        fft_filtered[-cutoff:] = fft_coeff[-cutoff:]
        smoothed = np.column_stack((np.real(np.fft.ifft(fft_filtered)), np.imag(np.fft.ifft(fft_filtered))))
        return smoothed

    def preserve_boundary_points(self, original, smoothed, mask_shape, tol=1e-3):
        h, w = mask_shape[:2]
        corrected = smoothed.copy()
        for i in range(len(original)):
            if abs(original[i, 0]) < tol or abs(original[i, 0] - (w - 1)) < tol:
                corrected[i, 0] = original[i, 0]
            if abs(original[i, 1]) < tol or abs(original[i, 1] - (h - 1)) < tol:
                corrected[i, 1] = original[i, 1]
        return corrected


    def smooth_polygon_fourier(self, poly, keep_fraction=0.1, image_shape=None):
        if keep_fraction is None:
            keep_fraction = self.keep_fraction
        """
        Apply Fourier smoothing on polygon boundaries.
        """
        exterior_coords = np.array(poly.exterior.coords)
        smoothed_exterior = self.smooth_contour_fourier(exterior_coords, keep_fraction=keep_fraction)
        if image_shape is not None:
            smoothed_exterior = self.preserve_boundary_points(exterior_coords, smoothed_exterior, image_shape)
        interiors = []
        for interior in poly.interiors:
            interior_coords = np.array(interior.coords)
            smoothed_interior = self.smooth_contour_fourier(interior_coords, keep_fraction=keep_fraction)
            if image_shape is not None:
                smoothed_interior = self.preserve_boundary_points(interior_coords, smoothed_interior, image_shape)
            interiors.append(smoothed_interior.tolist())
        return Polygon(smoothed_exterior.tolist(), holes=interiors)

    def build_adv_front_mesh(self, binary_mask, dem_data, size=50, keep_fraction=0.1, 
                         save_contour_plots=False, original_contour_path=None, fft_contour_path=None):
        """
        Builds an advanced mesh by:
          - Dilating the binary mask.
          - Extracting the largest water contour and its holes.
          - Applying Fourier smoothing to the contour.
          - Triangulating the smoothed polygon.
          - Lifting vertices to 3D using the DEM.
          
        If save_contour_plots is True and paths are provided, it will save:
          - The original extracted contour.
          - The Fourier-smoothed contour.
        """
        print("Building advanced front mesh...")
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
        
        # Apply Fourier smoothing and then preserve boundary points.
        smoothed_outer = self.smooth_contour_fourier(outer_contour, keep_fraction=keep_fraction)
        smoothed_outer = self.preserve_boundary_points(outer_contour, smoothed_outer, mask_bin.shape)
        
        # Save contour plots if requested.
        if save_contour_plots:
            if original_contour_path is not None:
                print(f"Saving original contour plot to {original_contour_path}...")
                plt.figure(figsize=(8, 6))
                plt.plot(outer_contour[:, 0], outer_contour[:, 1], 'b-')
                plt.gca().invert_yaxis()
                plt.xlabel("X (pixels)")
                plt.ylabel("Y (pixels)")
                plt.title("Original Contour")
                plt.savefig(original_contour_path, dpi=500)
                plt.close()
            if fft_contour_path is not None:
                print(f"Saving Fourier-smoothed contour plot to {fft_contour_path}...")
                plt.figure(figsize=(8, 6))
                plt.plot(smoothed_outer[:, 0], smoothed_outer[:, 1], 'r-')
                plt.gca().invert_yaxis()
                plt.xlabel("X (pixels)")
                plt.ylabel("Y (pixels)")
                plt.title("Fourier Smoothed Contour")
                plt.savefig(fft_contour_path, dpi=500)
                plt.close()
        
        # Get holes from child contours.
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
                smoothed_hole = self.smooth_contour_fourier(cnt, keep_fraction=keep_fraction)
                smoothed_hole = self.preserve_boundary_points(cnt, smoothed_hole, mask_bin.shape)
                holes_coords.append(smoothed_hole)
        
        exterior = smoothed_outer.tolist()
        poly_kwargs = {}
        if holes_coords:
            poly_kwargs['holes'] = [hole.tolist() for hole in holes_coords]
        water_poly = Polygon(exterior, **poly_kwargs)
        if not water_poly.is_valid:
            water_poly = water_poly.buffer(0)
        if water_poly.geom_type == 'MultiPolygon':
            water_poly = max(water_poly.geoms, key=lambda p: p.area)
        
        # Apply Fourier smoothing to the polygon boundaries.
        water_poly = self.smooth_polygon_fourier(water_poly, keep_fraction=keep_fraction, image_shape=mask_bin.shape)
        
        # Convert the polygon (with holes) into a PSLG.
        def polygon_to_pslg_with_holes(polygon):
            # Get exterior coordinates (making sure the polygon is closed).
            exterior_coords = np.array(polygon.exterior.coords[:-1], dtype=np.float64)
            vertices = exterior_coords.copy()
            n_exterior = len(exterior_coords)
            segments = np.array([[i, (i + 1) % n_exterior] for i in range(n_exterior)], dtype=np.int32)
            holes = []
            offset = n_exterior
            for interior in polygon.interiors:
                inter_coords = np.array(interior.coords[:-1], dtype=np.float64)
                n_interior = len(inter_coords)
                # Skip degenerate interiors.
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
        
        # Prepare the PSLG dictionary.
        pslg = dict(vertices=vertices, segments=segments, holes=holes)
        triangulation_opts = f'pq30a{size}'
        triangulation = triangle.triangulate(pslg, triangulation_opts)
        mesh_vertices = triangulation.get('vertices', np.empty((0, 2), dtype=np.float64))
        mesh_triangles = triangulation.get('triangles', np.empty((0, 3), dtype=np.int32))
        
        # Lift vertices to 3D using the DEM.
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
        
        # (Optional) Visualize the mask contours.
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
