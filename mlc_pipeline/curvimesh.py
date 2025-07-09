import logging
import numpy as np
import pygridgen
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
import cv2
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
        self.boundary_tol = config.get("boundary_tol", 10)
        self.interactive = config.get("interactive", False)
        
    def _interactive_pick_corners(self, contour, mask):
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = ListedColormap(['white', 'blue'])
        ax.imshow(mask, cmap=cmap, origin='upper')
        ax.plot(contour[:,0], contour[:,1], 'o-', color='red', markersize=4)
        selected = []
        scatter = ax.scatter([], [], c='cyan', s=75, marker='o')

        def onclick(event):
            if event.button == 3 and event.xdata is not None and event.ydata is not None:
                d = np.hypot(contour[:,0] - event.xdata, contour[:,1] - event.ydata)
                idx = int(np.argmin(d))
                if idx not in selected and len(selected) < 4:
                    selected.append(idx)
                    scatter.set_offsets(contour[selected])
                    fig.canvas.draw()

        def onundo(event):
            if selected:
                selected.pop()
                scatter.set_offsets(contour[selected] if selected else np.empty((0,2)))
                fig.canvas.draw()

        def onkeypress(event):
            if event.key == 'enter':
                plt.close(fig)

        ax_undo = plt.axes([0.8, 0.02, 0.1, 0.05])
        btn_undo = Button(ax_undo, 'Undo')
        btn_undo.on_clicked(onundo)

        ax.set_title('Right-click to select 4 corners; press Enter to finish or close window when done')
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkeypress)
        plt.show()

        if len(selected) != 4:
            raise ValueError(f"Interactive selection requires exactly 4 points, got {len(selected)}")
        logging.info(f"Interactive corners selected: {selected}")
        return selected


    def _order_corners_clockwise(self, contour, corners):
        pts = contour[corners]
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
        order = np.argsort(-angles)
        ordered = [corners[i] for i in order]
        logging.info(f"Ordered corners clockwise: {ordered}")
        return ordered

    def _compute_shape(self, contour, corners):
        deltas = np.diff(contour, axis=0)
        seg_lengths = np.hypot(deltas[:,0], deltas[:,1])
        cumlen = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        lengths = []
        for i in range(4):
            i0 = corners[i]
            i1 = corners[(i+1)%4]
            if i1 >= i0:
                length = cumlen[i1] - cumlen[i0]
            else:
                length = cumlen[-1] - cumlen[i0] + cumlen[i1]
            lengths.append(length)
        logging.info(f"Side lengths: {lengths}")
        len_u = 0.5*(lengths[0] + lengths[2])
        len_v = 0.5*(lengths[1] + lengths[3])
        nu = self.ni if len_u < len_v else self.nj
        nv = self.nj if len_u < len_v else self.ni
        return nv, nu

    def build(self, adv_mesh, avg_z, mask):
        # 1. Extract boundary contour from mask
        mask_bin = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask for boundary extraction")
        ext = max(contours, key=lambda c: len(c)).squeeze().astype(float)
        if ext.ndim == 1:
            ext = ext[np.newaxis, :]
        if not np.allclose(ext[0], ext[-1]):
            ext = np.vstack([ext, ext[0]])
        contour = ext

        # 2. Smooth contour while preserving edges
        h_mask, w_mask = mask.shape
        contour = self.mbuilder.chaikin_smoothing(contour, iterations=self.mbuilder.smoothing_iterations)
        if not np.allclose(contour[0], contour[-1]):
            contour[-1] = contour[0]
        boundary_tol = getattr(self.mbuilder, 'boundary_tol', 1.0)
        idx_edge = np.where(
            (contour[:,0] <= boundary_tol) | (contour[:,0] >= w_mask-1-boundary_tol) |
            (contour[:,1] <= boundary_tol) | (contour[:,1] >= h_mask-1-boundary_tol)
        )[0]
        for i in idx_edge:
            x,y = contour[i]
            x = 0.0 if x <= boundary_tol else (w_mask-1 if x >= w_mask-1-boundary_tol else x)
            y = 0.0 if y <= boundary_tol else (h_mask-1 if y >= h_mask-1-boundary_tol else y)
            contour[i] = [x,y]
        if not np.allclose(contour[0], contour[-1]):
            contour[-1] = contour[0]

        # 3. Pick & order corners
        if self.interactive:
            corners = self._interactive_pick_corners(contour, mask)
        else:
            deltas = np.diff(contour, axis=0)
            cumlen = np.concatenate(([0.0], np.cumsum(np.hypot(deltas[:,0], deltas[:,1]))))
            total = cumlen[-1]
            targets = [0.0, total*0.25, total*0.5, total*0.75]
            corners = [int(np.argmin(np.abs(cumlen - t))) for t in targets]
            logging.info(f"Picked corners at arc-length: {corners}")
        corners = self._order_corners_clockwise(contour, corners)
        if len(corners) != 4:
            raise ValueError(f"Expected 4 corners, got {len(corners)}: {corners}")

        # 4. Resample sides between corners
        nv, nu = self._compute_shape(contour, corners)
        side_counts = [nu, nv, nu, nv]
        resamp = []
        for i, count in enumerate(side_counts):
            i0, i1 = corners[i], corners[(i+1)%4]
            seg = contour[i0:i1+1] if i1>=i0 else np.vstack([contour[i0:], contour[:i1+1]])
            ds = np.hypot(*(np.diff(seg,axis=0).T))
            cum = np.concatenate(([0.0], np.cumsum(ds)))
            samples = np.linspace(0, cum[-1], count+1)[:-1]
            pts = []
            for s in samples:
                idx = np.searchsorted(cum, s)
                if idx==0:
                    pts.append(seg[0])
                else:
                    t0, t1 = cum[idx-1], cum[idx]
                    p0, p1 = seg[idx-1], seg[idx]
                    frac = (s-t0)/(t1-t0) if t1>t0 else 0.0
                    pts.append(p0 + frac*(p1-p0))
            resamp.append(np.array(pts))
        contour = np.vstack(resamp)
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        logging.info(f"Contour resampled to {len(contour)} points")

        # 5. Compute new corner indices on the resampled contour
        side_counts = [nu, nv, nu, nv]
        new_corners = [
            0,
            side_counts[0],
            side_counts[0] + side_counts[1],
            side_counts[0] + side_counts[1] + side_counts[2]
        ]
        logging.info(f"New corner indices on resampled contour: {new_corners}")

        # 6. Construct beta (integers) for Gridgen — no normalization needed
        beta = np.zeros(len(contour), dtype=int)
        for idx in new_corners:
            beta[idx] = 1

        # Debug overlay
        plt.figure(figsize=(6,6))
        cmap = ListedColormap(['white','blue'])
        plt.imshow(mask, cmap=cmap, origin='upper')
        plt.plot(contour[:,0], contour[:,1], 'o-', c='red', ms=3)
        pts = contour[new_corners]
        plt.scatter(pts[:,0], pts[:,1], c='cyan', s=75)
        plt.axis('off')
        plt.savefig('boundary_debug.png', dpi=150)
        plt.close()
        plt.figure(figsize=(6,6))
        cmap = ListedColormap(['white','blue'])
        plt.imshow(mask, cmap=cmap, origin='upper')
        plt.plot(contour[:,0], contour[:,1], 'o-', c='red', ms=3)
        pts = contour[new_corners]
        plt.scatter(pts[:,0], pts[:,1], c='cyan', s=75)
        plt.axis('off')
        plt.savefig('boundary_debug.png', dpi=150)
        plt.close()

        # Determine upper-left index for Gridgen (1-based)
        ul_corner = int(np.argmin(contour[:,0] + contour[:,1]))
        ul_idx = ul_corner + 1

        # Generate grid
        nu, nv = self._compute_shape(contour, new_corners)
        shape = (nu, nv)
        import time
        logging.info(f"Calling pygridgen with shape {shape}, ul_idx={ul_idx}")
        centroid = contour.mean(axis=0)
        pts = contour - centroid
        t0 = time.time()
        grid = pygridgen.Gridgen(
            pts[:,0], pts[:,1], beta,
            shape=shape,
            ul_idx=ul_idx,
            nnodes=14,
            precision=1e-6,
            checksimplepoly=False,
            verbose=True
        )
        t1 = time.time()
        logging.info(f"pygridgen completed in {t1-t0:.3f}s")

        X = grid.x + centroid[0]
        Y = grid.y + centroid[1]
        Z = avg_z if not np.isscalar(avg_z) else np.full_like(X, avg_z)

        # Triangulate
        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        ny_nodes, nx_nodes = X.shape
        faces = []
        for i in range(ny_nodes - 1):
            for j in range(nx_nodes - 1):
                n0 = i * nx_nodes + j
                n1 = n0 + 1
                n2 = n0 + nx_nodes
                n3 = n2 + 1
                if (i + j) % 2 == 0:
                    faces.append([n0, n2, n1])
                    faces.append([n1, n2, n3])
                else:
                    faces.append([n0, n2, n3])
                    faces.append([n0, n3, n1])
        faces = np.array(faces, dtype=int)
        mesh = SimpleMesh(verts, faces)
        mats = np.ones(len(faces), dtype=int)
        logging.info("CurviMeshBuilder.build complete")
        return mesh, mats

    def georeference(self, mesh, bbox, width, height, epsg):
        return self.mbuilder.georeference(mesh, bbox, width, height, epsg)

    def save(self, mesh, face_mats, path):
        ext = Path(path).suffix.lower()
        if ext in ['.stl', '.obj']:
            tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
            tri.export(path)
        else:
            self.mbuilder.save_mesh(mesh, face_mats, path)
        logging.info(f"Saved mesh to {path}")
