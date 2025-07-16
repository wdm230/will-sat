import numpy as np
import logging

class Bathymetry:
    """
    Generate logical‐grid bathymetry: profile always down the rows
    (top→bottom), then extruded left→right across the columns.

    Config params:
      shape: one of 'bathtub', 'parabola', 'v', 'average'
      bottom_height (h), top_height (H), slope, amplitude, sigma
    """
    def __init__(self, config: dict):
        self.shape     = config.get('shape', 'bathtub')
        self.h         = config.get('bottom_height', 0.0)
        self.H         = config.get('top_height',    1.0)
        self.slope     = config.get('slope',         None)
        self.amplitude = config.get('amplitude',     1.0)
        self.sigma     = config.get('sigma',         0.5)

    def generate(self,
                 n_rows: int,
                 n_cols: int,
                 avg_z=None
                ) -> np.ndarray:
        """
        n_rows × n_cols grid.  Build a 1-D profile along the rows
        (length n_rows), then tile it across n_cols.
        """
        logging.info(
            f"Bathymetry.generate: shape={self.shape!r}, "
            f"h={self.h}, H={self.H}, grid=({n_rows}×{n_cols})"
        )

        # 1) coordinate vector along rows
        rows = np.linspace(0.0, 1.0, n_rows)

        length = n_rows
        A      = self.H - self.h
        profile = np.full(length, self.h, dtype=float)

        # 2) build the 1-D profile down the rows
        if self.shape == 'bathtub':
            if self.slope is None:
                profile[  0 ] = self.H
                profile[ -1 ] = self.H
            else:
                w    = max(1, int(length * self.slope))
                ramp = np.linspace(self.H, self.h, w)
                profile[:w]  = ramp
                profile[-w:] = ramp[::-1]

        elif self.shape == 'parabola':
            U = 2*rows - 1
            profile = self.h + A * (U**2)

        elif self.shape == 'v':
            idx = np.arange(length)
            mid = (length - 1) / 2
            norm = np.abs(idx - mid) / mid
            profile = self.h + A * norm

        elif self.shape == 'average':
            if avg_z is None:
                raise ValueError("Must pass avg_z when shape='average'")
            if np.isscalar(avg_z):
                profile = np.full(length, avg_z, dtype=float)
            else:
                arr = np.asarray(avg_z, dtype=float)
                if arr.shape[0] != length:
                    raise ValueError(
                        f"avg_z length {arr.shape[0]} != expected {length}"
                    )
                profile = arr

        else:
            raise ValueError(f"Unknown shape: {self.shape!r}")

        # 3) extrude that 1-D profile across the columns:
        #    profile.reshape((n_rows,1)) tiled (1, n_cols) → (n_rows, n_cols)
        Z = np.tile(profile.reshape((length, 1)), (1, n_cols))

        logging.info(f"Bathymetry Z.min()={Z.min()}, Z.max()={Z.max()}")
        return Z
