import numpy as np
import logging

class Bathymetry:
    """
    Generate logical-grid bathymetry patterns based on matrix indices.

    Config parameters (pass via dict or kwargs):
      shape: one of 'bathtub', 'parabola', 'v', 'gaussian', 'sine'
      bottom_height: depth of basin interior (h)
      top_height:    elevation of rim or ends (H)
      slope:         optional slope fraction for trapezoidal walls
      amplitude:     amplitude for sine tube
      sigma:         standard deviation for gaussian hill
      orientation:   'auto', 'short', 'long', 'rows', or 'cols'
    """
    def __init__(self, config: dict):
        self.shape = config.get('shape', 'bathtub')
        self.h = config.get('bottom_height', 0.0)
        self.H = config.get('top_height', 1.0)
        self.slope = config.get('slope', None)
        self.amplitude = config.get('amplitude', 1.0)
        self.sigma = config.get('sigma', 0.5)
        self.orientation = config.get('orientation', 'auto')

    def generate(self, eta: int, nu: int) -> np.ndarray:
        """
        Return a (eta x nu) array of heights, applying shape along specified axis.
        """
        logging.info(f"Bathymetry.generate called with eta={eta}, nu={nu}, shape='{self.shape}', bottom_height={self.h}, top_height={self.H}, orientation='{self.orientation}'")
        Z = np.full((eta, nu), self.h, dtype=float)

        rows = np.linspace(0, 1, eta)
        cols = np.linspace(0, 1, nu)
        logging.info(f"rows[0:3]={rows[:3]}, cols[0:3]={cols[:3]}")

        # Determine shape axis
        if self.orientation in ('auto', 'short'):
            shape_axis = 0 if eta <= nu else 1
        elif self.orientation == 'long':
            shape_axis = 1 if eta <= nu else 0
        elif self.orientation == 'rows':
            shape_axis = 0
        elif self.orientation == 'cols':
            shape_axis = 1
        else:
            raise ValueError(f"Invalid orientation '{self.orientation}'")
        extrude_axis = 1 - shape_axis
        logging.info(f"shape_axis={shape_axis} ('0'=rows, '1'=cols), extrude_axis={extrude_axis}")

        # build shape and extrude coordinates
        if shape_axis == 0:
            u = rows[:, None]
            v = cols[None, :]
            logging.info("Variation down rows (shape_axis=0)")
        else:
            u = cols[None, :]
            v = rows[:, None]
            logging.info("Variation across cols (shape_axis=1)")

        logging.info(f"Entering shape branch: {self.shape}")
        if self.shape == 'bathtub':
            # Determine length of the shape‐axis
            length = eta if shape_axis == 0 else nu

            # Start with a flat bottom profile
            profile = np.full(length, self.h, dtype=float)

            if self.slope is None:
                # just vertical walls at the ends
                profile[0] = self.H
                profile[-1] = self.H
            else:
                # sloped walls of width w at each end
                w = max(1, int(length * self.slope))
                ramp = np.linspace(self.H, self.h, w)
                profile[:w] = ramp
                profile[-w:] = ramp[::-1]

            # extrude that 1-D profile across the other axis
            if shape_axis == 0:
                Z = np.tile(profile[:, None], (1, nu))
            else:
                Z = np.tile(profile[None, :], (eta, 1))

        elif self.shape == 'parabola':
            logging.info("Bathymetry: parabola profile")
            A = self.H - self.h
            U = 2*u - 1
            Z_vals = self.h + A*(U**2)
            logging.info(f"Parabola Z_vals sample[0:3]={Z_vals.ravel()[:3]}")
            # tile along extrude axis
            if shape_axis == 0:
                Z = np.tile(Z_vals, (1, nu))
            else:
                Z = np.tile(Z_vals, (eta, 1))
            logging.info(f"After tile: center={Z[eta//2, nu//2]}")

        elif self.shape == 'v':
            logging.info("Bathymetry: V-shape")
            n = eta if shape_axis == 0 else nu
            mid = (n - 1) / 2
            idx = np.arange(n)
            norm = np.abs(idx - mid) / mid
            arr = self.h + (self.H - self.h) * norm
            logging.info(f"V-shape arr sample[0:3]={arr[:3]}")
            if shape_axis == 0:
                Z = np.tile(arr[:, None], (1, nu))
            else:
                Z = np.tile(arr[None, :], (eta, 1))

        else:
            raise ValueError(f"Unknown bathymetry shape: {self.shape}")

        logging.info(f"Bathymetry generated: Z.min()={Z.min()}, Z.max()={Z.max()}, center={Z[eta//2, nu//2]}")
        return Z
