import numpy as np
from util.methods import get_best_fullspace
from moment_tensor import mt_to_magnitude
from grid.grid_create import FullMomentTensorGridSegment
from grid.grid_search import GridSearchOptimalParallel

def searching_magnitude(event, weight_list, rmag, dmag, min_mag, max_mag, pcpu, ncpu):
    '''
    First-step Grid search for best-fitting magnitude (seismic moment M0)
    If reference magnitude (like local magnitude) are provided, best-fitting magnitude will be search in range `[mag-rmag, mag+rmag]` with inverval dmag.
    Otherwise, best-fitting magnitude will be search in range `[min_mag, max_mag]` with inverval 1 first and
    then search in range `[mag-1, mag+1]` with inverval dmag.
    :param event: 
    :type event: NaNDC.DataIN
    :param weight_list: 
    :type weight_list: list or np.ndarray
    :param weight_list: 
    :type weight_list: list or np.ndarray
    :return magnitude: moment magnitude
    :rtype magnitude: float
    '''
    if weight_list[1] == 0:
        magnitude = None
        print("\n ----  Skipping magnitude searching (zero weight for P-wave amplitude) ----\n")
    else:
        if hasattr(event, 'evmag'):
            print("\n ----  Searching Best-Fitting Magnitude between {} - {} ----\n".format(event.evmag-rmag, event.evmag+rmag))
            ## create grid searching space
            mag_list = np.arange(event.evmag-rmag, event.evmag+rmag, dmag)
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11)
            gs, _ = GridSearchOptimalParallel(rawdata=event, grid=grid, weight=[0,1,0], pcpu=pcpu, ncpu=ncpu)
            _, best_mt = get_best_fullspace(gs)
            magnitude = round(mt_to_magnitude(best_mt), 2)
            print('\n Best fitting magnitude -- {} \n'.format(magnitude))
        else:
            # magnitude search step one #
            print("\n ----  1-1 Searching Best-Fitting Magnitude between {} - {}  ----\n".format(min_mag, max_mag))
            mag_list = np.arange(min_mag, max_mag, 1)
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11)
            gs, _ = GridSearchOptimalParallel(rawdata=event, grid=grid, weight=[0,1,0], pcpu=pcpu, ncpu=ncpu)
            _, best_mt = get_best_fullspace(gs)
            magnitude = round(mt_to_magnitude(best_mt), 2)
            # magnitude search step two #
            print("\n ----  1-2 Searching Best-Fitting Magnitude between {} - {}  ----\n".format(magnitude-1, magnitude+1))
            mag_list = np.arange(magnitude-1, magnitude+1, dmag)                                                                                                                                                                                  
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11)
            gs, _ = GridSearchOptimalParallel(rawdata=event, grid=grid, weight=[0,1,0], pcpu=pcpu, ncpu=ncpu)
            _, best_mt = get_best_fullspace(gs)
            magnitude = round(mt_to_magnitude(best_mt), 2)
            print('\n Best fitting magnitude -- {} \n'.format(magnitude))
    return magnitude
