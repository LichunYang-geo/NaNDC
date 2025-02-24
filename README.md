# NaNDC-v0.0.1
NaNDC is a full moment tensor inversion package for small earthquakes monitored by densy array.

## Quick Installation
The NaNDC can be installed by conda:

`conda env create -f envs.yml`

Then activate codna environment:

`conda activate nandc`

## Example
`cd ./example`
For Linux:
`bash ./run_example.sh`
or:
`python ./run_example.py`
For windows:
`python run_example.py`

## Input parameters
#must specified parameters
--event_list: 'catalog file'
--data_fir: 'folders contain observations'
--vmodel: '1-D velocity model'
# default
--save_dir: './inversion_results'
--weight: '1/1/1'
--npts_v: '21'
--npts_w: '45'
--npts_kappa: '73'
--npts_sigma: '37'
--npts_h: '21'
--tightness: '0.9'
--uniformity: '0.9'

## Citation
Lichun Yang, Ruijia Wang; NaNDC: Full Moment Tensor Inversion and Uncertainty Analysis for Large‐N Monitored Small Earthquakes. Seismological Research Letters 2024; doi: https://doi.org/10.1785/0220240219
