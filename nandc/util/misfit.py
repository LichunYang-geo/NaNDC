import numpy as np

def calc_optimal_misfit(
                syn_pol: np.ndarray, syn_pamp: np.ndarray, syn_spratio: np.ndarray, 
                obs_pol: np.ndarray, obs_pamp: np.ndarray, obs_spratio: np.ndarray, 
                tau1: np.ndarray, tau2: np.ndarray, tau3: np.ndarray, 
                lambd1, lambd2, lambd3) -> np.ndarray:
    '''
    Calculate misfit values for theoretical and observed data by L1 norm
    :param syn_pol: theoretical P-wave polarity
    :type syn_pol: np.ndarray
    :param syn_pamp: theoretical P-wave amplitude
    :type syn_pamp: np.ndarray
    :param syn_spratio: theoretical S/P amplitude ratio
    :type syn_spratio: np.ndarray
    :param obs_pol: observed P-wave polarity
    :type obs_pol: np.ndarray
    :param obs_pamp: observed P-wave amplitude
    :type obs_pamp: np.ndarray
    :param obs_spratio: observed S/P amplitude ratio
    :type obs_spratio: np.ndarray
    :param w1: weighting for P-wave polarity in each station
    :type w1: np.ndarray
    :param w2: weighting for P-wave amplitude in each station
    :type w2: np.ndarray
    :param w3: weighting for S/P amplitude ratio in each station
    :type w3: np.ndarray
    :return misfit: misfit values of each section
    :rtype misfit: np.ndarray
    '''
    
    pol_mis = np.sum(np.abs(tau1*0.5*(syn_pol-obs_pol)), axis=1)/np.sum(tau1*np.abs(obs_pol)) if lambd1 !=0 else np.zeros((np.size(syn_pol,0)))
    pamp_mis = np.sum(np.abs(tau2*(syn_pamp-obs_pamp)), axis=1)/np.sum(tau2*np.abs(obs_pamp)) if lambd2 !=0 else np.zeros((np.size(syn_pamp,0)))
    ratio_mis = np.sum(np.abs(tau3*(syn_spratio-obs_spratio)), axis=1)/np.sum(tau3*np.abs(obs_spratio)) if lambd3 !=0 else np.zeros((np.size(syn_spratio,0)))
    misfit  = np.concatenate((pol_mis.reshape(-1,1), pamp_mis.reshape(-1,1), ratio_mis.reshape(-1,1)), axis=1)

    return misfit