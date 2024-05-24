import os
import numpy as np
from typing import Optional, Union
from xarray import DataArray
from pandas import DataFrame
from forward import calc_ray_paras, calc_synthetic
from util.methods import get_best_lunespace, get_specified_lunespace, get_best_fullspace
from moment_tensor import para_to_delta_gamma, para_to_delta, para_to_gamma, decom, mom2other, magnitude_to_moment

class DataIN():
    def __init__(self, data: DataFrame, evinfo: Optional[Union[list, np.ndarray]], vmodel: str):
        '''
        information about events, stations and ray parameters
        :param data: stations locations and observations
        :type data: pd.DataFrame
        :param evinfo: event information, contains evid, evlo, evla, evdp and local magnitude (if has)
        :type evinfo: Optional[Union[list, np.ndarray]]
        :param vmodel: velocity model path
        :type vmodel: str
        '''
        # events info
        if len(evinfo) == 5:
            self.evid = evinfo[0]
            self.evlo = evinfo[1]
            self.evla = evinfo[2]
            self.evdp = evinfo[3]
            self.evmag = evinfo[4]
        elif len(evinfo) == 4:
            self.evid = evinfo[0]
            self.evlo = evinfo[1]
            self.evla = evinfo[2]
            self.evdp = evinfo[3]
        # stations info and parameters
        # calculate ray parameters for further inversion 
        self.stations = data.iloc[:,0:3] # DataFrame
        self.ray = calc_ray_paras(self.evlo, self.evla, self.evdp, self.stations, vmodel) # DataFrame
        self.ray_array = self.ray.to_numpy(dtype='float') # should not only have one station numpy array N*7
        # observation weight
        tau1 = data['tau1'].to_numpy(dtype=float)
        tau2 = data['tau2'].to_numpy(dtype=float)
        tau3 = data['tau3'].to_numpy(dtype=float)
        # observations
        self.pol = data['pol'].to_numpy(dtype=float)
        self.pamp = data['pamp'].to_numpy(dtype=float)
        self.spratio = data['spratio'].to_numpy(dtype=float)
        # remove zero observations
        self.zero_pol = np.where(self.pol==0)[0]
        self.zero_pamp = np.where(self.pamp==0)[0]
        self.zero_spratio = np.where(self.spratio==0)[0]
        self.pol_array = np.delete(self.pol, self.zero_pol)
        self.pamp_array = np.delete(self.pamp, self.zero_pamp)
        self.spratio_array = np.delete(self.spratio, self.zero_spratio)
        self.tau1 = np.delete(tau1, self.zero_pol)
        self.tau2 = np.delete(tau2, self.zero_pamp)
        self.tau3 = np.delete(tau3, self.zero_spratio)
        # for graphics (beachball)
        self.polar_info = np.column_stack((self.stations.stid, self.ray.azi, self.ray.tko, self.pol))
        self.pamp_info = np.column_stack((self.stations.stid, self.ray.azi, self.ray.tko, self.pamp))
        self.spratio_info = np.column_stack((self.stations.stid, self.ray.azi, self.ray.tko, self.spratio))

class DataOT():
    def __init__(self, gs: DataArray, event, misfit, magnitude):
        '''
        class contains inversion results
        '''
        self.magnitude = magnitude
        self.moment = magnitude_to_moment(self.magnitude) if magnitude is not None else None
        ## for each source types on eigenvalue lune, get the result corresponding to minumum misfit ##
        self.vw_lune, self.mt_lune = get_best_lunespace(gs, asnumpy=False)
        lon = para_to_gamma(self.vw_lune['v'].to_numpy())
        lat = para_to_delta(self.vw_lune['w'].to_numpy())
        values = self.vw_lune['misfit'].to_numpy()
        self.misfit_lune = np.column_stack([lon, lat, values])
        ## lune space uncertainty analysis ##
        minval, maxval = np.min(self.vw_lune['misfit']), np.max(self.vw_lune['misfit'])
        stepval = ((maxval-minval)/100)
        self.vw_spec, self.mt_spec = get_specified_lunespace(gs, minval+2*stepval)
        ## source mechanism - minumum misfit ##
        self.vw_best, self.mt_best = get_best_fullspace(gs, asnumpy=False) # best_vw 1*7 best_mt 1*6 [Mxx,Myy,Mzz,Mxy,Mxz,Myz]
        if len(self.vw_best) == 1:
            min_misfit_index = np.argmin(misfit[:,0])
            self.min_misfit = misfit[min_misfit_index,:]
            self.delta, self.gamma = para_to_delta_gamma(self.vw_best['v'].to_numpy(), self.vw_best['w'].to_numpy())
            self.percent, self.mt_iso, self.mt_dc, self.mt_clvd = decom(self.mt_best)
            self.NP1, self.NP2, self.Ptrpl, self.Ttrpl, self.Btrpl = mom2other(self.mt_best)
            self.syn_pol, self.syn_pamp, self.syn_spratio = calc_synthetic(mts=self.mt_best, supplement=event.ray_array)
        else:
            print("{} results have same minimum misfit values!".format(len(self.vw_best)))
