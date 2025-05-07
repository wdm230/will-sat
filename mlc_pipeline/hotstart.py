import numpy as np
from pathlib import Path

class HotstartBuilder:
    def __init__(self, config: dict):
        self.mode            = config.get('mode', 'wse')
        self.value           = config.get('value', None)
        self.previous_hot    = config.get('previous_hot', None)
        self.default_filename= config.get('output', 'hotstart.hot')

    def _from_file(self, mesh_path: Path):
        """Read mesh file and return (num_elems, num_nodes, nodes_xyz)."""
        num_nodes = num_elems = 0
        nodes = []
        with mesh_path.open('r') as f:
            for line in f:
                if line.startswith('ND '):
                    num_nodes += 1
                    parts = line.split()
                    x,y,z = map(float, parts[2:5])
                    nodes.append((x,y,z))
                elif line.startswith('E3T '):
                    num_elems += 1
        return num_elems, num_nodes, np.array(nodes)

    def _from_object(self, mesh):
        """Read mesh‐object and return (num_elems, num_nodes, nodes_xyz)."""
        verts = np.asarray(mesh.vertices, dtype=float)
        num_nodes = verts.shape[0]
        num_elems = len(mesh.faces)
        return num_elems, num_nodes, verts

    def parse_hot_dat(self, hot_path: Path, num_nodes: int):
        depths = np.zeros(num_nodes, dtype=float)
        with hot_path.open('r') as f:
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

    def build(self, mesh, output_path: str = None) -> Path:
        """
        Generate an ADH .hot file.

        :param mesh: either a filesystem path to mesh.3dm (str/Path), or
                     a mesh‐object with .vertices and .faces attributes.
        :param output_path: optional full path (including filename) for the
                            hot file.  Defaults to writing next to the mesh
                            using config’s 'output'.
        """
        # determine output path
        if output_path:
            out_path = Path(output_path)
        else:
            # if mesh is a path, put .hot beside it; if object, require cwd
            parent = Path(mesh).parent if isinstance(mesh, (str, Path)) else Path.cwd()
            out_path = parent / self.default_filename

        # extract mesh info
        if isinstance(mesh, (str, Path)):
            mesh_path = Path(mesh)
            n_elems, n_nodes, nodes = self._from_file(mesh_path)
        else:
            n_elems, n_nodes, nodes = self._from_object(mesh)

        # pick depths
        if self.mode == 'previous':
            if not self.previous_hot:
                raise ValueError("'previous_hot' must be set for mode 'previous'")
            depths = self.parse_hot_dat(Path(self.previous_hot), n_nodes)

        elif self.mode == 'wse':
            if self.value is None:
                raise ValueError("'value' must be set for mode 'wse'")
            depths = np.full(n_nodes, float(self.value), dtype=float)

        elif self.mode == 'constant_depth':
            if self.value is None:
                raise ValueError("'value' must be set for mode 'constant_depth'")
            depths = nodes[:,2] + float(self.value)

        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        # write out .hot
        with out_path.open('w') as f:
            f.write("DATASET\n")
            f.write("OBJTYPE \"mesh2d\"\n")
            f.write("BEGSCL\n")
            f.write(f"NC {n_elems}\n")
            f.write(f"ND {n_nodes}\n")
            f.write("NAME \"ioh\"\n")
            f.write("TS 0 0\n")
            for d in depths:
                f.write(f"{d:.6f}\n")

        return out_path
