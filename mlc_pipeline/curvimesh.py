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

        ax.set_title('Right-click to select 4 corners; press Enter to finish')
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
        cw = [corners[i] for i in np.argsort(-angles)]
        # rotate so the first corner is top-left (min x+y)
        pts_cw = contour[cw]
        start = int(np.argmin(pts_cw[:,0] + pts_cw[:,1]))
        ordered = cw[start:] + cw[:start]
        logging.info(f"Ordered corners CW starting TL: {ordered}")
        return ordered

    def _compute_side_counts(self, contour, corners):
        # cumulative arc-length
        deltas = np.diff(contour, axis=0)
        seg_lens = np.hypot(deltas[:,0], deltas[:,1])
        cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
        # measure each side
        lengths = []
        for i in range(4):
            i0, i1 = corners[i], corners[(i+1)%4]
            if i1 >= i0:
                length = cumlen[i1] - cumlen[i0]
            else:
                length = (cumlen[-1] - cumlen[i0]) + cumlen[i1]
            lengths.append(length)
        # assign counts: two longest -> nj, two shortest -> ni
        idx_sorted = np.argsort(lengths)
        long_idxs = idx_sorted[2:]
        side_counts = [(self.nj if i in long_idxs else self.ni) for i in range(4)]
        logging.info(f"Side lengths: {lengths}")
        logging.info(f"Side counts by idx: {side_counts}")
        return side_counts

    def build(self, adv_mesh, avg_z, mask):
        # 1. Extract boundary contour
        mask_bin = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask")
        ext = max(contours, key=lambda c: len(c)).squeeze().astype(float)
        if ext.ndim == 1:
            ext = ext[np.newaxis, :]
        if not np.allclose(ext[0], ext[-1]):
            ext = np.vstack([ext, ext[0]])
        contour = ext
        # 2. Smooth & clamp to boundary
        contour = self.mbuilder.chaikin_smoothing(contour, iterations=self.mbuilder.smoothing_iterations)
        contour[-1] = contour[0]
        h, w = mask.shape
        tol = getattr(self.mbuilder, 'boundary_tol', 1.0)
        idx_edge = np.where(
            (contour[:,0] <= tol) | (contour[:,0] >= w-1-tol) |
            (contour[:,1] <= tol) | (contour[:,1] >= h-1-tol)
        )[0]
        for i in idx_edge:
            x,y = contour[i]
            x = 0.0 if x <= tol else (w-1 if x >= w-1-tol else x)
            y = 0.0 if y <= tol else (h-1 if y >= h-1-tol else y)
            contour[i] = [x,y]
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
            logging.info(f"Auto-picked corners: {corners}")
        corners = self._order_corners_clockwise(contour, corners)
        # 4. Compute side counts and resample
        side_counts = self._compute_side_counts(contour, corners)
        resamp = []
        for i, count in enumerate(side_counts):
            i0, i1 = corners[i], corners[(i+1)%4]
            seg = contour[i0:i1+1] if i1>=i0 else np.vstack([contour[i0:], contour[:i1+1]])
            ds = np.hypot(*(np.diff(seg, axis=0).T))
            cum = np.concatenate(([0.0], np.cumsum(ds)))
            samples = np.linspace(0, cum[-1], count+1)[:-1]
            pts = []
            for s in samples:
                idx = np.searchsorted(cum, s)
                if idx == 0:
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
        # 5. New corner indices & beta
        cum_counts = np.cumsum(side_counts)
        new_corners = [0, cum_counts[0], cum_counts[1], cum_counts[2]]
        beta = np.zeros(len(contour), dtype=int)
        for idx in new_corners:
            beta[idx] = 1
        # 6. Gridgen
        nv, nu = side_counts[0], side_counts[1]
        shape = (nu, nv)
        ul_corner = corners[0]
        centroid = contour.mean(axis=0)
        pts = contour - centroid
        grid = pygridgen.Gridgen(
            pts[:,0], pts[:,1], beta,
            shape=shape,
            ul_idx=ul_corner,
            nnodes=14,
            precision=1e-6,
            checksimplepoly=False,
            verbose=True
        )
        # 7. Build mesh
        X = grid.x + centroid[0]
        Y = grid.y + centroid[1]
        # ensure Z is an array matching X/Y
        if np.isscalar(avg_z):
            Z = np.full_like(X, avg_z)
        else:
            Z = avg_z
        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        ny, nx = X.shape
        faces = []
        for i in range(ny-1):
            for j in range(nx-1):
                n0 = i*nx + j
                n1 = n0 + 1
                n2 = n0 + nx
                n3 = n2 + 1
                if (i+j) % 2 == 0:
                    faces += [[n0, n2, n1], [n1, n2, n3]]
                else:
                    faces += [[n0, n2, n3], [n0, n3, n1]]
        mesh = SimpleMesh(verts, np.array(faces, dtype=int))
        mats = np.ones(len(faces), dtype=int)
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