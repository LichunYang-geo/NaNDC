import os
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from util.base import DataIN, DataOT
from graphics.datacompare import plot_data
from graphics.beachball import plot_mech, plot_mech_data
from graphics.lune import plot_misfit_lune

def read_data(event_list: str, data_dir: str, vmodel: str) -> list:
    '''
    read data file 
    :param event_list: events file for inversion (evid, evlo, evla, evdp, magnitude)
    :type event_list: str
    :param data_dir: data folder path
    :type data_dir: str
    :param vmodel: velocity model file
    :type vmodel: str
    :return rawdatas: observation class list
    :rtype rawdatas: list
    '''
    rawdatas = []
    # open event list and read event information about evid evlo evla evdp and mag(if has)
    with open(event_list, 'r') as g:
        for gline in tqdm(g.readlines()):
            # '#' in event list file means skip this event
            if gline.strip('\n').split()[0] == '#':
                continue
            else:
                if len(gline.strip('\n').split()) == 5:  # evid evlo evla evdp mag
                    evid, evlo, evla, evdp, mag = gline.strip('\n').split()
                    evinfo = [evid, float(evlo), float(evla), float(evdp), float(mag)]
                elif len(gline.strip('\n').split()) == 4: # evid evlo evla evdp
                    evid, evlo, evla, evdp = gline.strip('\n').split()
                    evinfo = [evid, float(evlo), float(evla), float(evdp)]
                # read data(amplitude or S/P ratio) by evid and data_dir
                filepath = os.path.join(data_dir, evid+'.dat')
                try:
                    with open(filepath,'r') as f:
                        lines = f.readlines()
                        data = DataFrame(columns=['stid','stlo','stla','tau1','pol','tau2','pamp','tau3','spratio'],index=range(len(lines)))
                        for i in range(0, len(lines)):
                            try:
                                stid, stlo, stla, tau1, pol, tau2, pamp, tau3, spratio = lines[i].strip('\n').split(',')
                                data.iloc[i] = [stid, float(stlo), float(stla), float(tau1), float(pol), float(tau2), float(pamp), float(tau3), float(spratio)]
                            except ValueError:
                                print("{}: line {} length error\n".format(evid, i+1))
                                continue
                except FileNotFoundError:
                    print("Can not open file: {}\n".format(filepath))
                    continue
                rawdatas.append(DataIN(data, evinfo, vmodel))
    return rawdatas

def save_results(savepath: str, results: DataOT, event: DataIN):
    '''
    Save results
    param savepath: path to save inversion results
    type savepath: str
    param results: result dict in DataOT to save
    type results: dict
    param evID: event number
    type evID: str
    '''
    magnitude = str(round(results.magnitude,2)) if results.magnitude else 'NAN'
    moment = "{:.3e}".format(results.moment) if results.moment else 'NAN'
    mt_best = results.mt_best
    nodal_plane_1 = '/'.join((str(round(results.NP1[0],0)), str(round(results.NP1[1],0)), str(round(results.NP1[2],0))))
    nodal_plane_2 = '/'.join((str(round(results.NP2[0],0)), str(round(results.NP2[1],0)), str(round(results.NP2[2],0))))
    pdc, pclvd, piso = results.percent[0], results.percent[2], results.percent[1]
    lune_longitude = results.gamma[0]
    lune_latitude = results.delta[0]
    min_misfit = results.min_misfit
    
    fp = open(os.path.join(savepath, event.evid+'_meca.txt'), 'w')
    tplt1 = "{:>20} {:>12} {:>12} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}\n"
    fp.write(tplt1.format('evid', 'evlo', 'evla', 'evdp', 'Mw', 'moment', 'Mxx', 'Myy', 'Mzz', 'Mxy', 'Mxz', 'Myz', 'misfit'))
    fp.write(tplt1.format(' ', ' ', ' ', 'km', ' ', 'N-m', 'N-m', 'N-m', 'N-m', 'N-m', 'N-m', 'N-m', ' '))
    tplt2 = "{:>20s} {:>12.6f} {:>12.6f} {:>6.3f} {:>6s} {:>10s} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.3e} {:>8.3e} {:>8.3e} {:>8.3e}\n"
    fp.write(tplt2.format(
                event.evid, event.evlo, event.evla, event.evdp, 
                magnitude, moment,
                mt_best[0,0], mt_best[0,1], mt_best[0,2],
                mt_best[0,3], mt_best[0,4], mt_best[0,5], 
                min_misfit[0], min_misfit[1], min_misfit[2], min_misfit[3]))
    fp.close()

    ft = open(os.path.join(savepath, event.evid+'_decom.txt'), 'w')
    tplt3 = "{:>20} {:>12} {:>12} {:>6} {:>6} {:>18s} {:>18s} {:>9} {:>9} {:>6} {:>6} {:>6}\n"
    ft.write(tplt3.format('evid', 'evlo', 'evla', 'evdp', 'Mw', 'NP1', 'NP2', 'lune-lon.', 'lune-lat.', 'pdc', 'pclvd', 'piso'))
    tplt4 = "{:>20s} {:>12.6f} {:>12.6f} {:>6.3f} {:>6s} {:>18s} {:>18s} {:>9.1f} {:>9.1f} {:>6.1f} {:>6.1f} {:>6.1f}\n"
    ft.write(tplt4.format(
                event.evid, event.evlo, event.evla, event.evdp, 
                magnitude,
                nodal_plane_1, nodal_plane_2,
                lune_longitude, lune_latitude,
                pdc, pclvd, piso))
    ft.close()

def plot_results(savepath: str, filetype: str, results: DataOT, event: DataIN, weight: list):

    plot_mech(savepath=savepath, filename=event.evid, filetype=filetype, moment_tensor=results.mt_best, 
              best_dc=True, additional_info=False, PTaxis=True, 
              beachball_type='full', edgecolor='black', color_t='#f1939c')
    
    azimuth = event.ray['azi'].to_numpy()
    takeoff = event.ray['tko'].to_numpy()    
    pol=None if weight[0]==0 else event.pol
    pamp=None if weight[1]==0 else event.pamp
    spratio=None if weight[2]==0 else event.spratio 

    plot_mech_data(savepath=savepath, filename=event.evid+'_data', filetype=filetype, moment_tensor=results.mt_best, 
                   best_dc=True, PTaxis=True, 
                    takeoff=takeoff, azimuth=azimuth,
                    pol=pol,pamp=pamp,spratio=spratio,
                    beachball_type='full', edgecolor='black', color_t='#f1939c')
    
    plot_misfit_lune(savepath=savepath, filename=event.evid+'_lune'+'.'+filetype, lune_value=results.misfit_lune, lune_marker=[results.gamma, results.delta], colormap='no_green')

