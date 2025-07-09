# MLC Pipeline

A friendly, modular command-line tool for building river meshes from Sentinel-2 imagery, DEM data, classification masks, and more. It handles:

* Downloading and compositing Sentinel-2 MNDWI images
* Classification of water vs. non-water pixels
* DEM retrieval and smoothing
* Advancing-front (and optional curvilinear) meshing
* Generating hotstart files for ADH
* Writing out boundary-condition decks

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/mlc_pipeline.git
   cd mlc_pipeline
   ```

2. **Create the conda environment**
   (Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

   ```bash
   conda env create -f environment.yml
   conda activate mesh_maker
   ```

3. **Install the package**

   ```bash
   pip install .
   ```

4. **Authenticate Google Earth Engine**
   The first time you run, you’ll be prompted:

   ```bash
   run-mlc --config config.yaml
   ```

   Follow the on-screen instructions to log in.

---

## Configuration

All options live in your `config.yaml`. Here’s a minimal example:

```yaml
bbox: [-91.76, 31.19, -91.70, 31.23]        # [min_lon, min_lat, max_lon, max_lat]
output_dir: output/river_project            # where to save images, meshes, etc.
loc_tag: river                              # subfolder & filename prefix
project_name: earth_engine_project          # your GEE project ID

# How many dilation passes on the water mask before meshing
dilation_iterations: 10

dem:
  n_subboxes: 10    # subdivisions for DEM tiling
  sigma: 2          # Gaussian smoothing σ

sentinel:
  n_subboxes: 10    # splits of bbox for parallel downloads
  num_samples: 10   # how many images to sample per tile

classification:
  num_samples: 5000 # number of pixels to train your classifier

meshing:
  # Use curvilinear meshing?
  curvi: true        
  # node counts in U/V directions (ni, will be the shortest sides, nj will be the longest)
  ni: 8              
  nj: 150            
  size: 20.0         # target element size for advancing front meshing
  smoothing_iterations: 10
  chaikin_alpha: 0.5
  resample_len_frac: 0.10
  boundary_tol: 5.0  # snap points to image edges
  interactive: true  # pick corners by clicking

hotstart:
  enabled: false     # set to true to produce *.hot files
  # Optional settings (only if enabled):
  # mode: wse | constant_depth | previous
  # value: <number>          # water surface elevation or depth
  # previous_hot: path/to/file.hot
  # output: custom_name.hot

boundary:
  enabled: false     # set true to write out *.bc decks
  # If you turn this on, see "Boundary Conditions" below
```

---

## Usage

### Running the pipeline

```bash
python -m mlc_pipeline.pipeline \
  --config path/to/config.yaml \
  --log   pipeline.log
```

1. **Authenticate** with Earth Engine (only the first run).
2. **Download & stitch** Sentinel-2 MNDWI composites.
3. **Classify** water mask.
4. **Fetch & smooth** DEM, align to mask.
5. **Build** advancing-front mesh (and optional curvi mesh).
6. **Georeference** and save to `output_dir/loc_tag/loc_tag.3dm`.
7. **(Optional)** Generate hotstart (`.hot`) and boundary (`.bc`) files.

### Command-line options

* `--config`  Path to your YAML config (default: `../config.yaml`)
* `--log`     Path to write pipeline logs (default: `pipeline.log`)

---

## Boundary Conditions

If you flip `boundary.enabled: true`, the pipeline will:

1. Extract boundary loops from the final mesh
2. Feed them plus your `boundary:` settings into `bc_maker.BCBuilder`
3. Emit a deck file `loc_tag.bc` with cards like `OP`, `MP`, `MTS`, `EGS`, `XY1`, etc.

You can customize under `boundary:`:

```yaml
boundary:
  enabled: true

  operation:
    START_TIME: [0, 3600]
    END_TIME:   [3600, 7200]

  global_material:
    DENSITY: [1000, 0.0]

  materials:
    1:
      - type: DENSITY
        params: [998.2, 0.0]
    2:
      - type: VISCOSITY
        params: [1e-6]

  boundary_strings:
    MTS:
      - [1, 1]     # string_id, material_region

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

Once configured, re-run the pipeline and you’ll see `loc_tag.bc` in your output folder!

---

## Hotstart Files

Hotstart files (“.hot”) capture initial water elevations for ADH. Enable with:

```yaml
hotstart:
  enabled: true
  mode: constant_depth     # options: wse, constant_depth, previous
  value: 2.5               # water elevation/depth
  output: river_initial.hot
  previous_hot: old.hot    # only for mode “previous”
```


---


