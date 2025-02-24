import numpy as np
from xarray import DataArray
from pandas import DataFrame
from forward import calc_ray_paras, calc_synthetic
from util.methods import get_best_lunespace, get_specified_lunespace, get_best_fullspace
from moment_tensor import v_w_to_gamma_delta, w_to_delta, v_to_gamma, standard_decomposition, mom2other, magnitude_to_moment

class DataIN():
    '''
    class DataIn contains information about events, stations and ray parameters
    :param data: stations locations and observations [stid, stlo, stla, tao1, pol, tau2, pamp, tau3, spratio]
    :type data: pd.DataFrame
    :param evinfo: event information, contains evid, evlo, evla, evdp and local magnitude (if has)
    :type evinfo: Optional[Union[list, np.ndarray]]
    :param vmodel: velocity model path
    :type vmodel: str
    '''
    def __init__(self, data, evinfo, vmodel):
        if type(data) is not DataFrame:
            raise TypeError("Type of input data should be DataFrame.")
        
        # events info
        if len(evinfo) == 5:
            self.evid, self.evlo, self.evla, self.evdp, self.evmag = evinfo[0], evinfo[1], evinfo[2], evinfo[3], evinfo[4]
        elif len(evinfo) == 4:
            self.evid, self.evlo, self.evla, self.evdp = evinfo[0], evinfo[1], evinfo[2], evinfo[3]
        else:
            raise ValueError(" Check format of event list 'evid evlo evla evdp (evmag)' ")
        
        # stations info and parameters
        # calculate ray parameters for further inversion 
        self.stations = data.iloc[:,0:3] # DataFrame
        self.ray = calc_ray_paras(self.evlo, self.evla, self.evdp, self.stations, vmodel) # DataFrame
        self.ray_array = self.ray.to_numpy(dtype='float') # should not only have one station numpy array N*7
        # observation weight
        self.tau1_raw = data['tau1'].to_numpy(dtype=float)
        self.tau2_raw = data['tau2'].to_numpy(dtype=float)
        self.tau3_raw = data['tau3'].to_numpy(dtype=float)
        # observations
        self.pol_raw = data['pol'].to_numpy(dtype=float)
        self.pamp_raw = data['pamp'].to_numpy(dtype=float)
        self.spratio_raw = data['spratio'].to_numpy(dtype=float)
        # remove None (marked as zero) observations
        self.zero_pol_index = np.where(self.pol_raw==0)[0]
        self.zero_pamp_index = np.where(self.pamp_raw==0)[0]
        self.zero_spratio_index = np.where(self.spratio_raw==0)[0]
        self.tau1 = np.delete(self.tau1_raw, self.zero_pol_index)
        self.tau2 = np.delete(self.tau2_raw, self.zero_pamp_index)
        self.tau3 = np.delete(self.tau3_raw, self.zero_spratio_index)
        self.pol = np.delete(self.pol_raw, self.zero_pol_index)
        self.pamp = np.delete(self.pamp_raw, self.zero_pamp_index)
        self.spratio = np.delete(self.spratio_raw, self.zero_spratio_index)
        # # for graphics (beachball)
        # self.polar_proj   = np.column_stack((self.stations.stid, self.ray.azi, self.ray.tko, self.pol_raw))
        # self.pamp_proj    = np.column_stack((self.stations.stid, self.ray.azi, self.ray.tko, self.pamp_raw))
        # self.spratio_proj = np.column_stack((self.stations.stid, self.ray.azi, self.ray.tko, self.spratio_raw))

class DataOT():
    '''
    class contains inversion results
    '''    
    def __init__(self, gs: DataArray, event, misfit, magnitude):
        self.magnitude = magnitude
        self.moment = magnitude_to_moment(self.magnitude) if magnitude is not None else None
        
        ## for each source type on eigenvalue lune, get the result corresponding to minumum misfit ##
        self.tt_lune, self.mt_lune = get_best_lunespace(gs, asnumpy=False)
        lon = v_to_gamma(self.tt_lune['v'].to_numpy())
        lat = w_to_delta(self.tt_lune['w'].to_numpy())
        values = self.tt_lune['misfit'].to_numpy()
        self.misfit_lune = np.column_stack([lon, lat, values])
        ## lune space uncertainty analysis ##
        minval, maxval = np.min(self.tt_lune['misfit']), np.max(self.tt_lune['misfit'])
        stepval = ((maxval-minval)/100)
        self.vw_spec, self.mt_spec = get_specified_lunespace(gs, minval+2*stepval)
        ## source mechanism - minumum misfit ##
        self.vw_best, self.mt_best = get_best_fullspace(gs, asnumpy=False) # best_vw 1*7 best_mt 1*6 [Mxx,Myy,Mzz,Mxy,Mxz,Myz]
        if len(self.vw_best) == 1:
            min_misfit_index = np.argmin(misfit[:,0])
            self.min_misfit = misfit[min_misfit_index, :]
            self.gamma, self.delta = v_w_to_gamma_delta(self.vw_best['v'].to_numpy(), self.vw_best['w'].to_numpy())
            self.mt_iso, self.mt_clvd, self.mt_dc, self.p_iso, self.p_clvd, self.p_dc, self.eps = standard_decomposition(self.mt_best)
            self.NP1, self.NP2, self.Ptrpl, self.Ttrpl, self.Btrpl = mom2other(self.mt_best)
            self.syn_pol, self.syn_pamp, self.syn_spratio = calc_synthetic(mts=self.mt_best, supplement=event.ray_array)
        else:
            print("{} results have same minimum misfit values!".format(len(self.vw_best)))
