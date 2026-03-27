# NaNDC

NaNDC is a Python-based workflow for focal mechanism and moment tensor inversion using nodal-array observations. The current code supports inversion driven by a combination of P-wave polarity, P-wave amplitude, and S/P amplitude ratio, and provides three inversion modes:

- `DC`: double-couple inversion
- `FMT`: full moment tensor inversion
- `DC2FMT`: constrained full moment tensor inversion initialized from acceptable DC solutions

The repository includes a compact example dataset under `examples/`, which makes it easy to run and verify the workflow directly.

If you use NaNDC in research, please cite: 
Lichun Yang, Ruijia Wang; NaNDC: Full Moment Tensor Inversion and Uncertainty Analysis for Large-N Monitored Small Earthquakes. Seismological Research Letters 2024; 96 (2A): 1150-1162. doi: https://doi.org/10.1785/0220240219

## What This Repository Does

For each event, NaNDC will:

1. Read the event catalog and station observations.
2. Build a temporary TauPy model from a TauP-compatible 1-D `.nd` file.
3. Estimate the best-fitting magnitude if amplitude data are enabled.
4. Search the source parameter space according to the selected inversion mode.
5. Save mechanism, decomposition, residual tables, lune-space results, diagnostic figures, and a structured log file.

## Main Features

- Supports mixed constraints from polarity, amplitude, and S/P ratio
- Supports `DC`, `FMT`, and `DC2FMT` inversion workflows
- Handles multiple events in one run
- Writes per-event outputs into separate directories
- Produces mechanism plots, data overlays, residual plots, and lune-space figures
- Includes a ready-to-run example dataset under `examples/`

## Installation

### Recommended: Conda

`pygmt` requires GMT, so using conda is the easiest way to get a working environment.

```bash
conda env create -f environment.yml
conda activate nandc
```

### Alternative: pip

If you already have a suitable Python environment and GMT installed on your system, you can install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Note:

- `pygmt` still needs the GMT system package.
- If lune plotting fails while the inversion itself works, your GMT / PyGMT setup is the first thing to check.

## Quick Start

Run the script directly from the repository root:

```bash
python src/nandc/runNaNDC.py \
  --event_list=examples/events.list \
  --data_dir=examples \
  --vmodel=examples/velocity_example.nd \
  --save_dir=results/example_fmt \
  --inv_method=FMT \
  --weight_pol=1 \
  --weight_amp=1 \
  --weight_spr=1 \
  --normalization \
  --skipzero
```

Outputs will be written under:

```text
results/example_fmt/<event_id>/
```

For example, one event directory may contain:

```text
<event_id>.log
<event_id>.png
<event_id>_data.png
<event_id>_data_residual.png
<event_id>_data.csv
<event_id>_meca.txt
<event_id>_decom.txt
<event_id>_lune.txt
<event_id>_lune.png
```

`_lune.txt` and `_lune.png` are generated for `FMT` and `DC2FMT`. Pure `DC` runs do not produce lune-space outputs.

## Input Data Requirements

NaNDC expects three main inputs:

- an event list file
- a directory containing per-event observation files
- a 1-D velocity model in TauP-compatible `.nd` format

### 1. Event List

The event list is passed through `--event_list`.

Supported formats:

- 4 columns: `evid evlo evla evdp`
- 5 columns: `evid evlo evla evdp mag`

The parser accepts whitespace-separated or comma-separated input.

Example from `examples/events.list`:

```text
20161128065337.920, -117.248145, 54.343429, 3.269
clvd-69, -117.237747, 54.343700, 3.5
```

Column meanings:

- `evid`: event ID, also used to locate the observation file `<evid>.dat`
- `evlo`: event longitude in degrees
- `evla`: event latitude in degrees
- `evdp`: event depth in km
- `mag`: optional reference magnitude

Important notes:

- Event IDs must not contain `/` or `\`.
- If `mag` is present, NaNDC searches magnitude around `mag ± rmag`.
- If `mag` is absent, NaNDC performs a broader two-step search between `min_mag` and `max_mag`.

### 2. Observation Files

The observation directory is passed through `--data_dir`.

For each event in the event list, NaNDC looks for:

```text
<data_dir>/<event_id>.dat
```

Expected row format:

- 9 columns: `stid stlo stla tau1 pol tau2 amp tau3 spr`

Example from `examples/20161128065337.920.dat`:

```text
5B.1107, -117.254799, 54.310699, 1, -1.0, 1, -3.5544e-07, 1, 0.8193
5B.1108, -117.241501, 54.310299, 1, 1.0, 1, 4.3050e-07, 1, 0.8229
5B.1109, -117.203171, 54.317829, 1, 1.0, 1, 1.1456e-06, 1, -0.0072
```

Column meanings:

- `stid`: station ID
- `stlo`, `stla`: station longitude / latitude in degrees
- `tau1`: weight or availability flag for polarity
- `pol`: observed P-wave polarity, commonly `-1`, `0`, or `1`
- `tau2`: weight or availability flag for amplitude
- `amp`: observed P-wave amplitude
- `tau3`: weight or availability flag for S/P ratio
- `spr`: observed S/P amplitude ratio

Practical notes:

- Zero-valued observations are allowed.
- If all observations for an event are effectively unusable, the event will be skipped.
- Magnitude search is skipped if amplitude weight is zero or no amplitude data are available.

### 3. Velocity Model

The velocity model is passed through `--vmodel`.

The `.nd` file must satisfy TauP / TauPy formatting requirements, because NaNDC first uses ObsPy to build a temporary TauPy model for ray-parameter calculations. NaNDC then reads the first four numeric columns of each valid row:

```text
depth vp vs rho
```

Any extra columns in the file are ignored by the reader.

Example from `examples/velocity_example.nd`:

```text
0.00 3.2000 1.5900 2.2650 79.0 36.0
1.00 3.5370 1.7855 2.3620 131.0 58.0
2.00 3.8740 1.9809 2.3620 131.0 58.0
```

The code uses this model in two ways:

- it reads `depth`, `vp`, `vs`, and `rho` for internal calculations
- it builds a temporary TauPy model via ObsPy for ray-parameter calculations

## Running Different Inversion Modes

Most runs use the same command structure; in practice, you mainly switch inversion mode by changing `--inv_method`:

```bash
python src/nandc/runNaNDC.py \
  --event_list=examples/events.list \
  --data_dir=examples \
  --vmodel=examples/velocity_example.nd \
  --save_dir=results/example_run \
  --inv_method=FMT \
  --normalization \
  --skipzero
```

Mode summary:

- `DC`: simplest search space, no lune outputs
- `FMT`: full search in moment tensor space, produces lune-space products
- `DC2FMT`: first searches acceptable DC solutions, then restricts FMT search around them

## Output Files

NaNDC writes results for each event into:

```text
<save_dir>/<event_id>/
```

### Structured log

`<event_id>.log`

Contains sections such as:

- event metadata
- number of active observations
- coverage statistics
- active weights and normalization settings
- best-fit misfit values
- quality evaluation
- runtime

This file is the best starting point when checking whether a run behaved as expected.

### Result tables

`<event_id>_meca.txt`

- best-fitting moment tensor components
- event coordinates
- estimated magnitude and scalar moment
- total misfit

`<event_id>_decom.txt`

- nodal planes
- lune longitude / latitude
- percent DC / CLVD / ISO decomposition

`<event_id>_data.csv`

- observed versus synthetic polarity
- observed versus synthetic amplitude
- observed versus synthetic S/P ratio
- residual columns for each station

`<event_id>_lune.txt`

- lune-space grid of best solutions by source type
- generated for `FMT` and `DC2FMT`

### Figures

`<event_id>.png`

- focal mechanism plot

`<event_id>_data.png`

- mechanism plot overlaid with observations

`<event_id>_data_residual.png`

- observation residual view for amplitude and S/P ratio

`<event_id>_lune.png`

- lune-space misfit plot for `FMT` and `DC2FMT`

The image suffix depends on `--figtype`, which defaults to `png`.

## Important Command-Line Arguments

Below is a practical grouping of the most relevant parameters defined in `runNaNDC.py`.

### Input / output

- `--event_list`: event catalog file
- `--data_dir`: directory containing `<event_id>.dat`
- `--save_dir`: output directory, default `results`
- `--vmodel`: TauP-compatible 1-D `.nd` velocity model file

### Data weights

- `--weight_pol`: weight for polarity misfit
- `--weight_amp`: weight for P-wave amplitude misfit
- `--weight_spr`: weight for S/P ratio misfit

At least one weight must be non-zero.

### Misfit norm and normalization

- `--norm_pol`: `L1` or `L2`
- `--norm_amp`: `L1` or `L2`
- `--norm_spr`: `L1` or `L2`
- `--normalization`: enable misfit normalization (recommended)

### Inversion mode

- `--inv_method`: one of `DC`, `FMT`, `DC2FMT`

### Parallel settings

- `--n_jobs`: number of parallel workers, default `4`
- `--chunk_size`: grid batch size per worker, default `30000`

### Magnitude search

- `--rmag`: half-width around the reference magnitude
- `--dmag`: magnitude interval in the refined search
- `--min_mag`, `--max_mag`: bounds when no input magnitude is provided

### FMT grid settings

- `--npts_v`
- `--npts_w`
- `--npts_kappa`
- `--npts_sigma`
- `--npts_h`
- `--tightness`
- `--uniformity`

These determine the density of the search grid and can strongly affect runtime.

### Plotting

- `--figtype`: output image type, default `png`
- `--skipzero`: hide zero observations in plotted views
- `--plot_station_names`: annotate station names on the beachball plot

## Practical Tuning Advice

- Start with the provided example settings before increasing grid density.
- Use smaller grids and fewer active data types for quick smoke tests.
- If runtime is high, first try lowering `npts_v`, `npts_w`, `npts_kappa`, `npts_sigma`, or `npts_h`.
- If your machine has enough CPU cores, increase `n_jobs`.

## Example Datasets Included

The repository includes the following example input files:

- `examples/events.list`
- `examples/20161128065337.920.dat`
- `examples/clvd-69.dat`
- `examples/velocity_example.nd`

These files are useful for:

- validating that the environment is working
- understanding the expected input file organization
- running a quick end-to-end example

## Tests and Demo Scripts

The current `tests/` directory contains a simple runnable smoke test:

- `tests/test_run.py`: runs the main inversion script against the example files in `examples/`

Because `tests/test_run.py` uses relative paths, it is best run from inside the `tests/` directory or after adjusting the paths.

## Common Issues

### 1. PyGMT or GMT errors

Symptoms:

- inversion may run, but lune plotting fails
- PyGMT import or plotting raises an environment error

What to check:

- whether GMT is installed
- whether the conda environment was created from `environment.yml`

### 2. Event file not found

NaNDC looks for:

```text
<data_dir>/<event_id>.dat
```

If the event list contains an ID that does not match a file name exactly, that event will be skipped.

### 3. Empty or ineffective observations

If an event has no usable polarity, amplitude, or S/P ratio information after parsing, it will not produce a meaningful inversion result.

### 4. Runtime is too long

The largest cost usually comes from:

- dense search grids
- many events
- multiple active data types
- full FMT searches

Try smaller grids first, then scale up.

### 5. No lune output in DC mode

This is expected. Lune-space products are only created for `FMT` and `DC2FMT`.
