import numpy as np
import xarray as xr
from tqdm import tqdm
import multiprocessing as mp
from util.methods import get_optimal_misfit

def GridSearchOptimalParallel(rawdata, grid: xr.DataArray, weight: list, pcpu: int, ncpu: int) -> xr.DataArray:
    '''
    Inversion pool
    :param rawdata: obervation class
    :type: obseravtion
    :param grid:
    :type: xr.DataArray
    :param weight:
    :type: list
    :return gs: grid searching results contains misfit
    :rtype gs: xr.DataArray
    :return misfit_all: 
    :rtype misfit_all: np.ndarray
    '''
    ## Numpy array 
    # rho v w kappa sigma h
    Grid  = np.ascontiguousarray(grid.stack(z=grid.dims).coords.to_index().to_frame(index=False).to_numpy()) # numpy array N*7
    order = _sort_multi_(pcpu, len(Grid))
    
    print('Searching Space size: {}\n'.format(len(Grid)))
    pbar = tqdm(total=len(order))
    pbar.set_description('Searching')
    update = lambda *args: pbar.update()
    
    # open parallel pool
    maxcpu = mp.cpu_count()
    if ncpu > maxcpu: ncpu = maxcpu
    pool = mp.Pool(processes=ncpu)

    # parallel begin
    mp_results = []
    for _i in range(0, len(order)):
        _start_ = int(order[_i,0])
        _end_ = int(order[_i,1])
        result = pool.apply_async(get_optimal_misfit, (Grid[_start_:_end_,:], rawdata, weight), callback=update)
        mp_results.append(result)
    pool.close() 
    pool.join()
    # parallel end

    # extract results
    misfit = np.empty((len(Grid),3))
    for _i in range(0, len(order)):
        _start_ = int(order[_i,0])
        _end_ = int(order[_i,1])
        misfit[_start_:_end_,:] = mp_results[_i].get()

    # # Linear Normalization (min-max)
    norm_pol_misfit     = (misfit[:,0]-np.min(misfit[:,0]))/(np.max(misfit[:,0])-np.min(misfit[:,0])) if weight[0] !=0 else misfit[:,0]
    norm_pamp_misfit    = (misfit[:,1]-np.min(misfit[:,1]))/(np.max(misfit[:,1])-np.min(misfit[:,1])) if weight[1] !=0 else misfit[:,1]
    norm_spratio_misfit = (misfit[:,2]-np.min(misfit[:,2]))/(np.max(misfit[:,2])-np.min(misfit[:,2])) if weight[2] !=0 else misfit[:,2]
    misfit_sum = weight[0]*norm_pol_misfit + weight[1]*norm_pamp_misfit + weight[2]*norm_spratio_misfit
    misfit_all = np.concatenate((misfit_sum.reshape(-1,1), norm_pol_misfit.reshape(-1,1), norm_pamp_misfit.reshape(-1,1), norm_spratio_misfit.reshape(-1,1)), axis=1)

    misfit = misfit_sum.reshape(grid.sizes['rho'],grid.sizes['v'],grid.sizes['w'],grid.sizes['kappa'],grid.sizes['sigma'],grid.sizes['h'])
    gs = xr.DataArray(data=misfit, dims=grid.dims, coords=grid.coords)
    return gs, misfit_all

def _sort_multi_(pcpu: int, Grid_num: int) -> np.ndarray: 
    '''
    param inverval:
    param Grid_num:
    '''
    order = []
    _i = 0
    while 1:
        if (Grid_num) - (_i+1)*pcpu < 0:  # from 0 to Grid_num-1
            order.append([_i*pcpu, Grid_num])
            break
        else:
            order.append([_i*pcpu, (_i+1)*pcpu])
        _i = _i + 1
    return np.array(order)