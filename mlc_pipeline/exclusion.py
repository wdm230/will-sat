# mlc_pipeline/exclusion.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

class ExclusionEditor:
    """
    Let the user draw a polygon on `mask` by clicking vertices.
    Closing the window finishes the draw. Everything _inside_
    that polygon will be zeroed out.
    """
    def __init__(self, mask: np.ndarray):
        self.mask = mask.copy()
        self.pts = []

    def onclick(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.pts.append((x, y))
            event.inaxes.plot(x, y, 'ro')
            event.canvas.draw()

    def edit(self):
        fig, ax = plt.subplots()
        ax.imshow(self.mask, cmap='gray')
        ax.set_title("Click polygon vertices; close window when done")
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if len(self.pts) < 3:
            print("Fewer than 3 points—no exclusion applied.")
            return self.mask

        # build path and mask out inside
        poly = Path(self.pts)
        ny, nx = self.mask.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((X.ravel(), Y.ravel())).T
        inside = poly.contains_points(coords).reshape(self.mask.shape)

        new_mask = self.mask.copy()
        new_mask[inside] = 0
        return new_mask
