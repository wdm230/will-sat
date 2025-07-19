import logging
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

class ConfigError(Exception):
    """Exception raised for invalid BCBuilder configuration."""
    pass

# Allowed and required card types per section
ALLOWED_CARDS = {
    "operation": {"SW2", "TRN", "INC", "PRE", "BLK", "BT", "BTS", "TEM", "TPG", "NF2", "WND", "WAV", "DAM"},
    "iteration": {"NIT", "MIT", "NTL", "ITL"},
    "global_material": {"MU", "G", "RHO", "MUC"},
    "materials": {"ML", "SRT", "TRT", "EVS", "EEV", "DF"},
    "time_series": {"XY1", "XY2", "XYC"},
    "solution_controls": {"db", "nb"},
    "friction": {"MNG", "MNC", "ERH", "SAV", "URV", "EDO", "ICE", "DUN", "SRF", "SDK", "BRD"},
    "constituents": {"SAL", "TMP", "VOR", "CON"},
    "time_controls": {"T0", "TF", "IDT", "ATF", "STD"},
    "output_control": {"OC", "OS", "FLX", "PRN", "PC", "ADP", "ELM", "LVL", "MEO"},
    "boundary": {"enabled", "interactive", "edge_string_names", "header"},
}

REQUIRED_CARDS = {
    "operation": {"SW2", "TRN"},
    "time_controls": {"T0", "TF"},
}

class BCBuilder:
    def __init__(self, cfg: dict, curvi_builder=None):
        """
        cfg: configuration dict, with BC settings nested under 'boundary'
        curvi_builder: CurviMeshBuilder instance providing eta, nv
        """
        self.cfg = cfg or {}
        self.bc_cfg = self.cfg.get('boundary', self.cfg)
        self.curvi = curvi_builder
        self.string_id_counter = 0
        self.string_map = {}
        self._validate_config()

    def _validate_config(self):
        bc = self.bc_cfg
        # Required cards validation
        for section, required in REQUIRED_CARDS.items():
            items = bc.get(section, {})
            if isinstance(items, dict):
                missing = required - set(items.keys())
                if missing:
                    logging.warning(f"Missing required cards in '{section}': {missing}")
        # Boundary setup validation
        if bc.get('enabled', False):
            if not bc.get('interactive', False):
                raise ConfigError("boundary.interactive must be true when boundary.enabled is true")
            ens = bc.get('edge_string_names', [])
            if not isinstance(ens, list) or len(ens) != 2:
                raise ConfigError("boundary.edge_string_names must be a list of two names")
            if not self.curvi:
                raise ConfigError("curvi_builder must be provided when boundary.enabled is true")

    def _next_string_id(self):
        """Auto-increment string ID."""
        self.string_id_counter += 1
        return self.string_id_counter

    def _write(self, f, text=""):
        f.write(text + "\n")

    def build(self, mesh, bc_path: str) -> Path:
        """
        Generate the .bc file at bc_path.
        mesh: the mesh object (provides vertices) if boundary.enabled.
        """
        bc = self.bc_cfg
        out = Path(bc_path)
        with out.open('w') as f:
            # Optional header
            header = bc.get('header')
            if header:
                self._write(f, header)

            # Core sections
            self._write_operation(f)
            self._write_iteration(f)
            self._write_constituents(f)
            self._write_global_material(f)
            self._write_materials(f)

            # Boundary strings
            if bc.get('enabled', False):
                self._write_boundary_strings(f, mesh)

            # Remaining BC sections
            self._write_time_series(f)
            self._write_output_controls(f)
            self._write_friction(f)
            self._write_solution_controls(f)
            self._write_time_controls(f)

            # End
            self._write(f, 'END')
        return out

    def _write_operation(self, f):
        cfg = self.bc_cfg.get('operation', {})
        self._write(f, '! Operation Parameters')
        for key, vals in cfg.items():
            line = f"OP {key}" if not vals else f"OP {key} " + ' '.join(map(str, vals))
            self._write(f, line)
        self._write(f)

    def _write_iteration(self, f):
        cfg = self.bc_cfg.get('iteration', {})
        self._write(f, '! Iteration Parameters')
        for key, vals in cfg.items():
            self._write(f, f"IP {key} " + ' '.join(map(str, vals)))
        self._write(f)

    def _write_constituents(self, f):
        cfg = self.bc_cfg.get('constituents', [])
        self._write(f, '! Constituent Properties')
        for c in cfg:
            params = ' '.join(map(str, c.get('params', [])))
            self._write(f, f"CN {c['type']} \"{c.get('name','')}\" {params}")
        self._write(f)

    def _write_global_material(self, f):
        cfg = self.bc_cfg.get('global_material', {})
        self._write(f, '! Global Material Properties')
        for key, vals in cfg.items():
            self._write(f, f"MP {key} " + ' '.join(map(str, vals)))
        self._write(f)

    def _write_materials(self, f):
        cfg = self.bc_cfg.get('materials', {})
        self._write(f, '! Material Properties')
        for region, mats in cfg.items():
            for m in mats:
                params = ' '.join(map(str, m.get('params', [])))
                self._write(f, f"MP {m['type']} {region} {params} ! Name: Material {region}")
            self._write(f)
        self._write(f)

    def _write_boundary_strings(self, f, mesh):
        bc = self.bc_cfg
        materials = bc.get('materials', {})
        # Collect material IDs (region numbers)
        mat_ids = sorted(int(r) for r in materials.keys())

        names = bc['edge_string_names']

        # Pull grid dimensions
        eta = getattr(self.curvi, 'eta', None)
        nv = getattr(self.curvi, 'nv', None)
        if eta is None or nv is None:
            raise ConfigError("CurviMeshBuilder must have eta and nv attributes for boundary strings.")
        # first and last columns (short sides)
        edges = [list(range(0, eta * nv, nv)), list(range(nv - 1, eta * nv, nv))]

        # Interactive prompt for edge selection
        fig, ax = plt.subplots()
        verts = mesh.vertices
        for idx, nodes in enumerate(edges):
            pts = np.array([verts[n][:2] for n in nodes])
            ax.plot(pts[:, 0], pts[:, 1], '-', color=['r', 'b'][idx])
        ax.set_title(f"Click on the '{names[0]}' edge to assign its boundary string")
        picked = []
        def on_click(evt):
            pts0 = np.array([verts[n][:2] for n in edges[0]])
            pts1 = np.array([verts[n][:2] for n in edges[1]])
            d0 = np.min(np.hypot(*(pts0 - (evt.xdata, evt.ydata)).T))
            d1 = np.min(np.hypot(*(pts1 - (evt.xdata, evt.ydata)).T))
            picked.append(0 if d0 < d1 else 1)
            plt.close(fig)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        # Determine selected and secondary boundaries
        sel = picked[0]
        primary_name, secondary_name = names[0], names[1]
        primary_nodes = edges[sel]
        secondary_nodes = edges[1 - sel]

        # Header for boundary strings
        self._write(f, '! Boundary Strings')

        # Write one MTS per material region (MTS regionID regionID)
        for mid in mat_ids:
            self._write(f, f"MTS {mid} {mid}")

        # Compute starting EGS ID as last material ID + 1
        last_mat_id = mat_ids[-1] if mat_ids else 0

        # Write EGS for each boundary string in config order
        for idx, (name, nodes) in enumerate(((primary_name, primary_nodes), (secondary_name, secondary_nodes)), start=1):
            sid = last_mat_id + idx
            self._write(f, f"! EGS for {name}")
            for a, b in zip(nodes[:-1], nodes[1:]):
                self._write(f, f"EGS {a+1} {b+1} {sid}")
            # Save for solution controls
            self.string_map[name] = sid

        self._write(f)







    def _write_time_series(self, f):
        cfg = self.bc_cfg.get('time_series', [])
        if not cfg:
            return
        self._write(f, '! Time Series')
        for ts in cfg:
            stype = ts.get('series_type', 'XY1')
            sid = ts.get('id')
            in_u = ts.get('in_units', 0)
            out_u = ts.get('out_units', 0)
            raw = ts.get('file') if 'file' in ts else ts.get('points', [])
            pts = []
            if isinstance(raw, str):
                for line in Path(raw).open():
                    nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)
                    if len(nums) >= 2:
                        pts.append(tuple(map(float, nums[:2])))
            else:
                pts = [(float(t), float(v)) for t, v in raw]
            self._write(f, f"{stype} {sid} {in_u} {out_u} {len(pts)}")
            for t, v in pts:
                self._write(f, f"{t:.2f} {v:.2f}")
            self._write(f)
        self._write(f)

    def _write_output_controls(self, f):
        cfg = self.bc_cfg.get('output_control', {})
        self._write(f, '! Output Controls')
        for key, items in cfg.items():
            for it in items:
                if key == 'OS':
                    unit = it.get('unit', 0)
                    self._write(f, f"OS {it['series']} {len(it['segments'])} {unit}")
                    for seg in it['segments']:
                        self._write(f, ' '.join(map(str, seg)))
                else:
                    sid = it if isinstance(it, (int, str)) else it.get('series')
                    self._write(f, f"{key} {sid}")
        self._write(f)

    def _write_friction(self, f):
        cfg = self.bc_cfg.get('friction', {})
        self._write(f, '! Friction Controls')
        for key, entries in cfg.items():
            for e in entries:
                self._write(f, f"FR {key} {e['region']} " + ' '.join(map(str, e['params'])))
        self._write(f)

    def _write_solution_controls(self, f):
        """
        Write Dirichlet (DB) and Neumann (NB) boundary conditions based on the config:
          db:
            <BC-type>:
              - string: <name>
                series: [<series-IDs>]
          nb:
            <BC-type>:
              - string: <name>
                series: [<series-IDs>]
        """
        cfg = self.bc_cfg
        self._write(f, '! Solution Controls')
        for sect in ['db','nb']:
            card = sect.upper()             # "DB" or "NB"
            for bc_type, specs in cfg.get(sect, {}).items():
                for spec in specs:
                    name = spec['string']
                    sid  = self.string_map.get(name)
                    if sid is None:
                        raise KeyError(f"No ID for '{name}', got map: {self.string_map}")
                    for series in spec['series']:
                        self._write(f, f"{card} {bc_type} {sid} {series}")



    def _write_time_controls(self, f):
        cfg = self.bc_cfg.get('time_controls', {})
        self._write(f, '! Time Controls')
        for key, vals in cfg.items():
            self._write(f, f"TC {key} " + ' '.join(map(str, vals)))
        self._write(f)
