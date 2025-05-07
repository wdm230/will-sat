import yaml
from pathlib import Path
import logging
import numpy as np

class BCBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def build(self, bc_path: str, boundary_loops: list) -> Path:
        out = Path(bc_path)
        with out.open('w') as f:
            self._write_operation(f)
            self._write_global_material(f)
            self._write_materials(f)
            self._write_string_structures(f, boundary_loops)
            self._write_time_series(f)
            self._write_iteration(f)
            self._write_friction(f)
            self._write_constituents(f)
            self._write_time_controls(f)
            self._write_solution_controls(f)
            self._write_output_controls(f)
            f.write('END\n')
        return out

    def _write_constituents(self, f):
        f.write('! Constituent Properties\n')
        for c in self.cfg.get('constituents', []):
            params = ' '.join(map(str, c['params']))
            f.write(f"CN {c['type']} \"{c['name']}\" {params}\n")
        f.write('\n')

    def _write_global_material(self, f):
        f.write('! Global Material Properties\n')
        for key, vals in self.cfg.get('global_material', {}).items():
            f.write(f"MP {key} {' '.join(map(str, vals))}\n")
        f.write('\n')
        
    def _write_materials(self, f):
        """
        Write out MP cards grouped by region, with a blank line
        between each region’s block.
        Expects cfg['materials'] as:
          region_id:
            - type: XXX
              params: [...]
            - type: YYY
              params: [...]
        """
        f.write('! Material Properties\n')
        mbR = self.cfg.get('materials', {})
        for region, mats in mbR.items():
            for m in mats:
                params = ' '.join(map(str, m['params']))
                f.write(f"MP {m['type']} {int(region)} {params}\n")
            # blank line after each region block
            f.write('\n')
        # extra blank line before next section
        f.write('\n')


    def _write_iteration(self, f):
        f.write('! Iteration Parameters\n')
        for key, vals in self.cfg.get('iteration', {}).items():
            f.write(f"IP {key} {' '.join(map(str, vals))}\n")
        f.write('\n')

    def _write_operation(self, f):
        f.write('! Operation Parameters\n')
        for key, vals in self.cfg.get('operation', {}).items():
            if vals is None:
                f.write(f"OP {key}\n")
            else:
                f.write(f"OP {key} {' '.join(map(str, vals))}\n")
        f.write('\n')
        
    def _write_string_structures(self, f, boundary_loops):
        """
        Emit:
          MTS <string_id> <material_region>
        for every entry in cfg['boundary_strings']['MTS'],
        then
          EGS <n1> <n2> <string_id>
        for each edge in each loop, starting string_id
        at (max_mts_id + 1), (max_mts_id + 2), …
        """
        f.write('! String Structures\n')

        # 1) write out all MTS cards
        mts_cfg = self.cfg.get('boundary_strings', {}).get('MTS', [])
        # ensure ints
        mts_entries = [(int(sid), int(region)) for sid, region in mts_cfg]
        for sid, region in mts_entries:
            f.write(f"MTS {sid} {region}\n")

        # find the highest string ID used so far
        max_mts_id = max((sid for sid, _ in mts_entries), default=0)

        # 2) now write each boundary‐loop as an EGS string,
        #    starting at max_mts_id + 1
        for loop_idx, (nodes, _) in enumerate(boundary_loops, start=1):
            string_id = max_mts_id + loop_idx
            arr = np.asarray(nodes, dtype=int)
            # pair arr[0]→arr[1], arr[1]→arr[2], … arr[-2]→arr[-1]
            for n1, n2 in zip(arr, arr[1:]):
                f.write(f"EGS {n1+1} {n2+1} {string_id}\n")

        f.write('\n')

    
    def _write_time_series(self, f):
        ts_list = self.cfg.get('time_series', [])
        if not ts_list:
            return
    
        f.write('! Time Series\n')
        for ts in ts_list:
            # pick up your BC-series identifiers
            stype    = ts.get('series_type',
                        ts.get('bc_type',
                        ts.get('type_id', 1)))
            sid      = ts.get('id',
                        ts.get('series_id'))
            in_units = ts.get('in_units',
                        ts.get('input_units', 0))
            out_units= ts.get('out_units',
                        ts.get('output_units', 0))
    
            # either a list of [t,v] pairs or a path to a text file
            raw = ts.get('points',
                  ts.get('data', []))
    
            # if it's a filename, read it in
            if isinstance(raw, str):
                file_path = Path(raw)
                pts = []
                with file_path.open() as fp:
                    for line in fp:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        t, v = map(float, parts[:2])
                        pts.append((t, v))
            else:
                # assume iterable of two-tuples
                pts = [(float(t), float(v)) for t, v in raw]
    
            npts = len(pts)
            f.write(f"XY1 {stype} {sid} {npts} {in_units} {out_units}\n")
            for t, v in pts:
                f.write(f"{t:.2f} {v:.2f}\n")
            
            f.write("\n")
        f.write('\n')



    def _write_solution_controls(self, f):
        f.write('! Solution Controls\n')
        for k, vals in self.cfg.get('db', {}).items():
            f.write(f"DB {k} {' '.join(map(str, vals))}\n")
        for k, vals in self.cfg.get('nb', {}).items():
            f.write(f"NB {k} {' '.join(map(str, vals))}\n")
        f.write('\n')

    def _write_friction(self, f):
        f.write('! Friction Controls\n')
        for key, entries in self.cfg.get('friction', {}).items():
            for e in entries:
                f.write(f"FR {key} {e['region']} " + ' '.join(map(str, e['params'])) + "\n")
        f.write('\n')

    def _write_time_controls(self, f):
        f.write('! Time Controls\n')
        for key, vals in self.cfg.get('time_controls', {}).items():
            f.write(f"TC {key} {' '.join(map(str, vals))}\n")
        f.write('\n')

    def _write_output_controls(self, f):
        f.write('! Output Control\n')
        for key, vals in self.cfg.get('output_control', {}).items():
            f.write(f"{key} {' '.join(map(str, vals))}\n")
        f.write('\n')
