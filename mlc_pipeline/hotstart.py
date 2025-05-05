import yaml
import numpy as np
from pathlib import Path
import rasterio


def parse_mesh_info(mesh_path):
    """
    Parse a 3DM mesh file and return number of elements and nodes.
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


def parse_hot_dat(hot_path, num_nodes):
    """
    Read a previous ADH hotstart file and extract depth array.
    Assumes hot file lines: node_id depth [velocity]
    """
    depths = np.zeros(num_nodes)
    with open(hot_path, 'r') as f:
        # skip header until TS line
        for line in f:
            if line.strip().startswith('TS '):
                break
        for line in f:
            parts = line.split()
            if not parts:
                continue
            nid = int(parts[0])
            if 1 <= nid <= num_nodes:
                depths[nid - 1] = float(parts[1])
    return depths


def sample_raster_at_points(raster_path, points):
    """
    Sample a single-band raster at given (x,y) world coordinates.
    """
    values = []
    with rasterio.open(raster_path) as src:
        for x, y, _ in points:
            for val in src.sample([(x, y)]):
                values.append(val[0])
    return np.array(values)


def generate_hotstart(config_path: str, output_path: str) -> Path:
    """
    Generate an ADH .hot file with depth initial conditions per node.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    hot_cfg = cfg.get('hotstart', {})

    mesh_path = Path(hot_cfg.get('mesh_path',
                     Path(cfg.get('output_dir','mlc_images')) /
                     cfg.get('loc_tag','river') / 'mesh.3dm'))

    num_elems, num_nodes = parse_mesh_info(mesh_path)

    # Read mesh nodes for coordinate-based modes
    nodes = []
    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith('ND '):
                parts = line.split()
                x, y, z = map(float, parts[2:5])
                nodes.append((x, y, z))
    nodes = np.array(nodes)

    # Determine depths
    mode = hot_cfg.get('mode', 'wse')
    val = hot_cfg.get('value', None)
    depths = np.zeros(num_nodes)

    if mode == 'previous':
        prev = hot_cfg['previous_hot']
        depths = parse_hot_dat(prev, num_nodes)
    elif mode == 'wse':
        if val is None:
            raise ValueError("'value' must be set for wse mode")
        depths[:] = float(val)
    elif mode == 'constant_depth':
        if val is None:
            raise ValueError("'value' must be set for constant_depth mode")
        depths = nodes[:,2] + float(val)
    else:
        raise ValueError(f"Unknown hotstart mode: {mode}")

    with open(output_path, 'w') as f:
        f.write("DATASET\n")
        f.write("OBJTYPE \"mesh2d\"\n")
        f.write("BEGSCL\n")
        f.write(f"ND {num_elems}\n")
        f.write(f"NC {num_nodes}\n")
        f.write("NAME \"ioh\"\n")
        f.write("TS 0 0\n")
        for d in depths:
            f.write(f"{d:.6f}\n")
    return output_path


