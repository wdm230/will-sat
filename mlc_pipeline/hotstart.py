import numpy as np
import rasterio
from pathlib import Path

class HotstartBuilder:
    def __init__(self, config: dict):
        # mode: 'previous', 'wse', or 'constant_depth'
        self.mode = config.get('mode', 'wse')
        self.value = config.get('value', None)
        self.previous_hot = config.get('previous_hot', None)
        # output filename (without path)
        self.filename = config.get('output', 'hotstart.hot')

    def parse_mesh_info(self, mesh_path: Path):
        """
        Parse mesh.3dm to count elements (E3T) and nodes (ND).
        """
        num_nodes = 0
        num_elems = 0
        with open(mesh_path, 'r') as f:
            for line in f:
                if line.startswith('ND '):
                    num_nodes += 1
                elif line.startswith('E3T '):
                    num_elems += 1
        return num_elems, num_nodes

    def parse_mesh_nodes(self, mesh_path: Path):
        """
        Read mesh.3dm and return Nx3 array of node coords.
        """
        nodes = []
        with open(mesh_path, 'r') as f:
            for line in f:
                if line.startswith('ND '):
                    parts = line.split()
                    x, y, z = map(float, parts[2:5])
                    nodes.append((x, y, z))
        return np.array(nodes)

    def parse_hot_dat(self, hot_path: Path, num_nodes: int):
        """
        Read previous .hot file, return depths array.
        """
        depths = np.zeros(num_nodes)
        with open(hot_path, 'r') as f:
            for line in f:
                if line.strip().startswith('TS '):
                    break
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                nid = int(parts[0])
                if 1 <= nid <= num_nodes:
                    depths[nid-1] = float(parts[1])
        return depths

    def build(self, mesh_path: str) -> Path:
        """
        Generate an ADH .hot file with standard header and depths, saving alongside the mesh.
        """
        mesh_path = Path(mesh_path)
        out_dir = mesh_path.parent
        out_path = out_dir / self.filename

        num_elems, num_nodes = self.parse_mesh_info(mesh_path)
        nodes = self.parse_mesh_nodes(mesh_path)

        # determine depths
        if self.mode == 'previous':
            if not self.previous_hot:
                raise ValueError("'previous_hot' must be set for mode 'previous'.")
            depths = self.parse_hot_dat(Path(self.previous_hot), num_nodes)
        elif self.mode == 'wse':
            if self.value is None:
                raise ValueError("'value' must be set for mode 'wse'.")
            depths = np.full(num_nodes, float(self.value))
        elif self.mode == 'constant_depth':
            if self.value is None:
                raise ValueError("'value' must be set for mode 'constant_depth'.")
            depths = nodes[:,2] + float(self.value)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

        # write hot file with ADH header
        with open(out_path, 'w') as f:
            f.write("DATASET\n")
            f.write("OBJTYPE \"mesh2d\"\n")
            f.write("BEGSCL\n")
            f.write(f"NC {num_elems}\n")
            f.write(f"ND {num_nodes}\n")
            f.write("NAME \"ioh\"\n")
            f.write("TS 0 0\n")
            for d in depths:
                f.write(f"{d:.6f}\n")
        return out_path

