import numpy as np
from xarray import DataArray
from pandas import DataFrame, concat
from moment_tensor import para_to_mij
from forward import calc_synthetic
from util.misfit import calc_optimal_misfit

def get_best_lunespace(gs: DataArray, asnumpy: bool=True, mtype: str="NED"):
    '''
    extract each source types' minimum fault oritation and its misfit from full moment tensor space
    '''
    if type(gs) is not DataArray:
        raise ValueError("`gs` must be a `DataArray`")
    
    ## asnumpy only for min_vw, min_mt always is numpy array with size 
    values = gs.min(dim=('rho','kappa','sigma','h'))  # size  v*w
    vw_misfit = gs.where(gs==values).stack(z=gs.dims)  # after stack contans Nan
    coords = vw_misfit[vw_misfit.notnull()].coords['z'].to_index().to_frame(index=False)  # v*w coords
    values = DataFrame(vw_misfit[vw_misfit.notnull()].data, columns=['misfit'])  #v*w misfit values

    vw_lune = concat([coords, values], axis=1)  # DataFrame N*7 rho, v, w, kappa, sigma, h, misfit
    mt_lune = para_to_mij(
        vw_lune['rho'].to_numpy(), vw_lune['v'].to_numpy(), vw_lune['w'].to_numpy(),
        vw_lune['kappa'].to_numpy(), vw_lune['sigma'].to_numpy(), vw_lune['h'].to_numpy(), mtype)
    
    if asnumpy:
        return np.array(vw_lune), mt_lune # np.ndarray
    else:
        return vw_lune, mt_lune  # pd.DataFrame, np.ndarray

def get_specified_lunespace(gs: DataArray, tolerance: float, asnumpy: bool=True, mtype: str="NED"):
    '''
    extract specified source types' minimum fault oritation and its misfit from full moment tensor space
    '''
    # first extract each source type and its misfit
    vw_lune, mt_lune = get_best_lunespace(gs, asnumpy=True, mtype=mtype)
    indices = np.where(vw_lune[:,6]<=tolerance)[0]
    vw_spec = vw_lune[indices,:]
    mt_spec = mt_lune[indices,:]

    vw_spec = DataFrame(vw_spec, columns=('rho','v','w','kappa','sigma','h','misfit'))

    if asnumpy:
        return np.array(vw_spec), mt_spec
    else:
        return vw_spec, mt_spec
    
def get_best_fullspace(gs: DataArray, asnumpy: bool=True, mtype: str="NED"):
    """
    extract the moment tensor with minimum misfit from full moment tensor space
    :param gs:
    :dtype gs: xarray DataArray
    :param asnumpy: 
    :dtype asnumpy: bool
    """    
    values = gs.where(gs==gs.min(), drop=True)  # DataArray
    coords = values.coords.to_index().to_frame(index=False)  # DataFrame
    vw_best = concat([coords, DataFrame(values.values.squeeze().reshape(-1,1), columns=['misfit'])], axis=1)  # DataFrame 1*7 rho,v,w,kappa,sigma,h,misfit
    mt_best = para_to_mij(
        vw_best['rho'].to_numpy(), vw_best['v'].to_numpy(), vw_best['w'].to_numpy(),
        vw_best['kappa'].to_numpy(), vw_best['sigma'].to_numpy(), vw_best['h'].to_numpy(), mtype)
    
    if asnumpy:
        return np.array(vw_best), mt_best # np.ndarray
    else:
        return vw_best, mt_best # pd.DataFrame, np.ndarray

def get_optimal_misfit(grid: np.ndarray, rawdata, weight: list) -> list:
    '''
    '''
    lambd1, lambd2, lambd3 = weight[0], weight[1], weight[2]
    mts = para_to_mij(grid[:,0],grid[:,1],grid[:,2],grid[:,3],grid[:,4],grid[:,5]) # N*6
    pol, pamp, spratio = calc_synthetic(mts=mts, supplement=rawdata.ray_array) #ndarray N*2 Pamp&SPratio
    syn_pol = np.delete(pol, rawdata.zero_pol, axis=1)
    syn_pamp = np.delete(pamp, rawdata.zero_pamp, axis=1)
    syn_spratio = np.delete(spratio, rawdata.zero_spratio, axis=1)
    misfit = calc_optimal_misfit(syn_pol, syn_pamp, syn_spratio, 
                         rawdata.pol_array, rawdata.pamp_array, rawdata.spratio_array, 
                         rawdata.tau1, rawdata.tau2, rawdata.tau3,
                         lambd1, lambd2, lambd3)
    return misfit