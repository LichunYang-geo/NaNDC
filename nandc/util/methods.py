import numpy as np
from xarray import DataArray
from pandas import DataFrame, concat
from moment_tensor import tt_to_mij
from forward import calc_synthetic
from util.misfit import calc_optimal_misfit

def get_best_lunespace(gs, asnumpy=True, mtype="NED"):
    '''
    extract each source types' minimum fault oritation and its misfit from full moment tensor space
    :param gs: inversion DataArray
    :type gs: xarray.DataArray
    :param asnumpy: 
    '''
    if type(gs) is not DataArray: raise ValueError("`gs` must be a `DataArray`")
    ## asnumpy only for min_vw, min_mt always numpy array
    min_value = gs.min(dim=('rho','kappa','sigma','h')) # DataArray with size (v, w), min misfit values for each source type
    index = gs.where(gs==min_value).stack(z=gs.dims) # DataArray after stack contans Nan with size (len(grid), 1)
    coords = index[index.notnull()].coords['z'].to_index().to_frame(index=False) # DataFrame with size (v*w, 6)
    values = DataFrame(index[index.notnull()].data, columns=['misfit']) # DataFrame with size (v*w, 1)       

    tt_lune = concat([coords, values], axis=1)  # DataFrame N*7 rho, v, w, kappa, sigma, h, misfit
    mt_lune = tt_to_mij(
        tt_lune['rho'].to_numpy(),   tt_lune['v'].to_numpy(),     tt_lune['w'].to_numpy(),
        tt_lune['kappa'].to_numpy(), tt_lune['sigma'].to_numpy(), tt_lune['h'].to_numpy(), mtype)
    
    if asnumpy:
        return np.array(tt_lune), mt_lune # np.ndarray
    else:
        return tt_lune, mt_lune  # pd.DataFrame, np.ndarray

def get_specified_lunespace(gs, tolerance, asnumpy=True, mtype="NED"):
    '''
    extract specified source types' minimum fault oritation and its misfit from full moment tensor space
    :param gs: 
    :type gs: xarray.DataArray
    '''
    if type(gs) is not DataArray: raise ValueError("`gs` must be a `DataArray`")
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
    
def get_best_fullspace(gs, asnumpy=True, mtype="NED"):
    """
    extract the moment tensor with minimum misfit from full moment tensor space
    :param gs:
    :dtype gs: xarray.DataArray
    :param asnumpy: 
    :dtype asnumpy: bool
    """
    if type(gs) is not DataArray: raise ValueError("`gs` must be a `DataArray`")
    values = gs.where(gs==gs.min(), drop=True)  # DataArray
    coords = values.coords.to_index().to_frame(index=False)  # DataFrame
    tt_best = concat([coords, DataFrame(values.values.squeeze().reshape(-1,1), columns=['misfit'])], axis=1)  # DataFrame 1*7 rho,v,w,kappa,sigma,h,misfit
    mt_best = tt_to_mij(
        tt_best['rho'].to_numpy(), tt_best['v'].to_numpy(), tt_best['w'].to_numpy(),
        tt_best['kappa'].to_numpy(), tt_best['sigma'].to_numpy(), tt_best['h'].to_numpy(), mtype)
    
    if asnumpy:
        return np.array(tt_best), mt_best # np.ndarray
    else:
        return tt_best, mt_best # pd.DataFrame, np.ndarray

def get_optimal_misfit(grid, rawdata, weight):
    '''
    '''
    lambd1, lambd2, lambd3 = weight[0], weight[1], weight[2]
    mts = tt_to_mij(grid[:,0],grid[:,1],grid[:,2],grid[:,3],grid[:,4],grid[:,5]) # N*6
    pol, pamp, spratio = calc_synthetic(mts=mts, supplement=rawdata.ray_array) #ndarray N*2 Pamp&SPratio
    syn_pol = np.delete(pol, rawdata.zero_pol_index, axis=1)
    syn_pamp = np.delete(pamp, rawdata.zero_pamp_index, axis=1)
    syn_spratio = np.delete(spratio, rawdata.zero_spratio_index, axis=1)
    misfit = calc_optimal_misfit(syn_pol, syn_pamp, syn_spratio, 
                         rawdata.pol, rawdata.pamp, rawdata.spratio, 
                         rawdata.tau1, rawdata.tau2, rawdata.tau3,
                         lambd1, lambd2, lambd3)
    return misfit