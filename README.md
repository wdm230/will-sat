## MLC Pipeline

A modular command-line tool for generating river meshes from Sentinel-2 imagery, DEM data, classification masks, and more. It supports:

* Sentinel-2 MNDWI composite download and stitching
* Water/non-water classification
* DEM retrieval and smoothing
* Advancing-front and curvilinear meshing
* Hotstart file generation for ADH
* Boundary-condition deck creation

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/mlc_pipeline.git
   cd mlc_pipeline
   ```
2. **Create and activate the conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate mesh_maker
   ```
3. **Install the package**

   ```bash
   pip install .
   ```
4. **Authenticate with Google Earth Engine**

   ```bash
   run-mlc --config config.yaml
   ```

   Follow the prompt to log in.

---

## Configuration

All options are defined in `config.yaml`. Example:

```yaml
bbox: [-91.76, 31.19, -91.70, 31.23]
output_dir: output/river_project
loc_tag: river
project_name: earth_engine_project

dilation_iterations: 10

dem:
  n_subboxes: 10
  sigma: 2

sentinel:
  n_subboxes: 10
  num_samples: 10

classification:
  num_samples: 5000

meshing:
  curvi: true
  ni: 8
  nj: 150
  size: 20.0
  smoothing_iterations: 10
  chaikin_alpha: 0.5
  resample_len_frac: 0.10
  boundary_tol: 5.0
  interactive: true

hotstart:
  enabled: false
  # mode: wse | constant_depth | previous
  # value: <number>
  # previous_hot: path/to/file.hot
  # output: custom_name.hot

boundary:
  enabled: false
  # operation:
  #   START_TIME: [start, end]
  #   END_TIME:   [start, end]
  # global_material:
  #   DENSITY: [value, flag]
  # materials:
  #   1:
  #     - type: DENSITY
  #       params: [value, flag]
  # boundary_strings:
  #   MTS:
  #     - [string_id, region]
  # time_series:
  #   - series_type: id
  #     input_units: id
  #     output_units: id
  #     points: [[t0, v0], [t1, v1], ...]
  # iteration:
  #   MAX_IT: [max, flag]
  # friction:
  #   DRAG:
  #     - region: id
  #       params: [value]
  # time_controls:
  #   TIME_STEP: [step, flag]
  # solution_controls:
  #   db:
  #     SOLVER: [id, flag]
  #   nb:
  #     ADAPT: [id, flag]
  # output_control:
  #   PRINT: [interval, flag]
  # constituents:
  #   - type: id
  #     name: name
  #     params: [value, flag]
```

---

## Usage

```bash
python -m mlc_pipeline.pipeline \
  --config path/to/config.yaml \
  --log pipeline.log
```

### Command-line options

* `--config`: Path to YAML config (default: `config.yaml`)
* `--log`: Path for pipeline logs (default: `pipeline.log`)

---

## Boundary Conditions

Set `boundary.enabled: true` in your config to produce a `.bc` deck. The file `loc_tag.bc` will contain, in order:

```
OP  <operation parameters>
MP  <global and region materials>
CN  <constituent definitions>
MTS <boundary-string IDs>
EGS <edge-group entries>
XY1 <time-series definitions>
IP  <iteration settings>
FR  <friction entries>
TC  <time-control decks>
DB  <"db" solution-control decks>
NB  <"nb" solution-control decks>
<key>  <output-control cards>
END
```

### Example `boundary` config

```yaml
boundary:
  enabled: true

  operation:
    START_TIME: [0.0, 3600.0]
    END_TIME:   [3600.0, 7200.0]

  global_material:
    DENSITY: [1000.0, 0.0]

  materials:
    1:
      - type: DENSITY
        params: [998.2, 0.0]

  boundary_strings:
    MTS:
      - [1, 1]

  time_series:
    - series_type: 1
      id: 10
      input_units: 0
      output_units: 0
      points:
        - [0.0, 1.2]
        - [3600.0, 1.3]

  iteration:
    MAX_IT: [100, 10]

  friction:
    DRAG:
      - region: 1
        params: [0.01]

  time_controls:
    TIME_STEP: [10, 0]

  solution_controls:
    db:
      SOLVER: [2, 1]
    nb:
      ADAPT:  [1, 0]

  output_control:
    PRINT: [100, 0]

  constituents:
    - type: 1
      name: water_temp
      params: [273.15, 0.5]
```

Configure these sections in the `boundary:` block of `config.yaml` as shown above.

---

## Hotstart Files

Enable with `hotstart.enabled: true`. Options:

```yaml
hotstart:
  enabled: true
  mode: constant_depth     # wse | constant_depth | previous
  value: 2.5
  previous_hot: old.hot    # only for mode "previous"
  output: river_initial.hot
```
---

Enjoy reproducible mesh generation!

