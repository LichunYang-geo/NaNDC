import numpy as np

from tqdm import tqdm
from xarray import DataArray
from joblib import Parallel, delayed

from nandc_mt import tt_to_mij
from nandc_forward import calc_synthetic

from util.nandc_method import calc_optimal_misfit, misfit_normalization

def GridSearchOptimalParallel(event, grid, weight, norm_list, normalization, n_jobs, chunk_size):

    # 1. 构造任务
    # convert xr.DataArray to numpy array
    grid_array = np.ascontiguousarray(grid.stack(z=grid.dims).coords.to_index().to_frame(index=False).to_numpy()) # numpy array N*7
    N_models = grid_array.shape[0]
    print(f" Space size: {N_models} ")
    
    # 2. 单个 chunk 的计算函数
    def _process_chunk_(start, end):
        '''

        '''

        mts = tt_to_mij(grid_array[start:end, 0], grid_array[start:end, 1], grid_array[start:end, 2],
                        grid_array[start:end, 3], grid_array[start:end, 4], grid_array[start:end, 5])
        amp_chunk, spr_chunk = calc_synthetic(mts, event.ray_array)
        pol_chunk = np.sign(amp_chunk)

        syn_pol_chunk = np.delete(pol_chunk, event.zero_pol_index, axis=1)
        syn_amp_chunk = np.delete(amp_chunk, event.zero_amp_index, axis=1)
        syn_spr_chunk = np.delete(spr_chunk, event.zero_spr_index, axis=1)

        pol_misfit_chunk, amp_misfit_chunk, spr_misfit_chunk = calc_optimal_misfit(syn_pol_chunk, syn_amp_chunk, syn_spr_chunk, 
                                                                                   event.pol, event.amp, event.spr, 
                                                                                   event.tau1, event.tau2, event.tau3,
                                                                                   weight[0], weight[1], weight[2], norm_list)
        
        misfit[start:end, 0] = pol_misfit_chunk
        misfit[start:end, 1] = amp_misfit_chunk
        misfit[start:end, 2] = spr_misfit_chunk

        pbar.update(1)

    # 3. 并行执行（多线程）
    misfit = np.empty((N_models, 3), dtype=np.float64)
    task_list = [(i, min(i+chunk_size, N_models)) for i in range(0, N_models, chunk_size)]
    with tqdm(total=len(task_list), desc="Grid Searching") as pbar:
        Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(delayed(_process_chunk_)(s, e) for (s, e) in task_list)
    
    # 4. 组合误差
    misfit_sum, misfit_all = misfit_normalization(event, misfit[:,0], misfit[:,1], misfit[:,2], weight, normalization, p=99)

    gs = DataArray(data=misfit_sum.reshape(grid.sizes['rho'], grid.sizes['v'], grid.sizes['w'], grid.sizes['kappa'], grid.sizes['sigma'], grid.sizes['h']), 
                   dims=grid.dims, coords=grid.coords)
    
    return gs, misfit_all

def GridSearchOptimalParallelFromTable(event, model_table, weight, norm_list, normalization, n_jobs, chunk_size):

    model_table = np.ascontiguousarray(model_table, dtype=np.float64)
    if model_table.ndim != 2 or model_table.shape[1] != 6:
        raise ValueError("`model_table` must be a numpy array with shape (N, 6)")

    N_models = model_table.shape[0]
    print(f" Space size: {N_models} ")

    if N_models == 0:
        empty_result = np.empty((0, 10), dtype=np.float64)
        empty_misfit = np.empty((0, 4), dtype=np.float64)
        return empty_result, empty_misfit

    def _process_chunk_(start, end):
        mts = tt_to_mij(
            model_table[start:end, 0], model_table[start:end, 1], model_table[start:end, 2],
            model_table[start:end, 3], model_table[start:end, 4], model_table[start:end, 5]
        )
        amp_chunk, spr_chunk = calc_synthetic(mts, event.ray_array)
        pol_chunk = np.sign(amp_chunk)

        syn_pol_chunk = np.delete(pol_chunk, event.zero_pol_index, axis=1)
        syn_amp_chunk = np.delete(amp_chunk, event.zero_amp_index, axis=1)
        syn_spr_chunk = np.delete(spr_chunk, event.zero_spr_index, axis=1)

        pol_misfit_chunk, amp_misfit_chunk, spr_misfit_chunk = calc_optimal_misfit(
            syn_pol_chunk, syn_amp_chunk, syn_spr_chunk,
            event.pol, event.amp, event.spr,
            event.tau1, event.tau2, event.tau3,
            weight[0], weight[1], weight[2], norm_list
        )

        misfit[start:end, 0] = pol_misfit_chunk
        misfit[start:end, 1] = amp_misfit_chunk
        misfit[start:end, 2] = spr_misfit_chunk

        pbar.update(1)

    misfit = np.empty((N_models, 3), dtype=np.float64)
    task_list = [(i, min(i+chunk_size, N_models)) for i in range(0, N_models, chunk_size)]
    with tqdm(total=len(task_list), desc="Grid Searching") as pbar:
        Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(delayed(_process_chunk_)(s, e) for (s, e) in task_list)

    _, misfit_all = misfit_normalization(event, misfit[:,0], misfit[:,1], misfit[:,2], weight, normalization, p=99)

    result_table = np.column_stack([model_table, misfit_all])
    return result_table, misfit_all
