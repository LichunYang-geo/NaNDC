import os
import numpy as np
from pandas import DataFrame
from util.base import DataIN
from graphics.beachball import plot_mech, plot_mech_data
from graphics.lune import plot_misfit_lune_pygmt

def read_data(event_list, data_dir, vmodel):
    '''
    read observations
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
        for gline in g.readlines():
            if len(gline.strip('\n').split(',')) == 5:  # evid evlo evla evdp mag
                evid, evlo, evla, evdp, mag = gline.strip('\n').split(',')
                evinfo = [evid, float(evlo), float(evla), float(evdp), float(mag)]
            elif len(gline.strip('\n').split(',')) == 4: # evid evlo evla evdp
                evid, evlo, evla, evdp = gline.strip('\n').split(',')
                evinfo = [evid, float(evlo), float(evla), float(evdp)]
            else:
                print(f"Wrong format {gline}")
                continue
            # read data (amplitude or S/P ratio) by evid and data_dir
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

def save_results(savepath, results, event):
    '''
    Save results
    param savepath: path to save inversion results
    type savepath: str
    param results: result dict in DataOT to save
    type results: DataOT
    param event:
    type evID: DataIN
    '''
    magnitude = str(round(results.magnitude,2)) if results.magnitude else 'NAN'
    moment = "{:.3e}".format(results.moment) if results.moment else 'NAN'
    mt_best = results.mt_best
    nodal_plane_1 = '/'.join((str(round(results.NP1[0],0)), str(round(results.NP1[1],0)), str(round(results.NP1[2],0))))
    nodal_plane_2 = '/'.join((str(round(results.NP2[0],0)), str(round(results.NP2[1],0)), str(round(results.NP2[2],0))))
    pdc, pclvd, piso = results.p_dc, results.p_clvd, results.p_iso
    lune_longitude, lune_latitude = results.gamma[0], results.delta[0]
    min_misfit = results.min_misfit
    # meca
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
    # decom
    ft = open(os.path.join(savepath, event.evid+'_decom.txt'), 'w')
    tplt3 = "{:>20} {:>12} {:>12} {:>6} {:>6} {:>18s} {:>18s} {:>9} {:>9} {:>6} {:>6} {:>6}\n"
    ft.write(tplt3.format('evid', 'evlo', 'evla', 'evdp', 'Mw', 'NP1', 'NP2', 'lune-lon.', 'lune-lat.', 'pdc', 'pclvd', 'piso'))
    tplt4 = "{:>20s} {:>12.6f} {:>12.6f} {:>6.3f} {:>6s} {:>18s} {:>18s} {:>9.1f} {:>9.1f} {:>6.1f} {:>6.1f} {:>6.1f}\n"
    ft.write(tplt4.format(
                event.evid, event.evlo, event.evla, event.evdp, 
                magnitude, nodal_plane_1, nodal_plane_2,
                lune_longitude, lune_latitude, pdc, pclvd, piso))
    ft.close()
    # data
    fo = open(os.path.join(savepath, event.evid+'_data.csv'), 'w')
    fo.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        'stid', 'stlo', 'stla', 'azi', 'tko', 'aoi', 'dist (m)', 'alpha (m/s)', 'beta (m/s)', 'rho (kg/m3)',
        'tau1', 'obs_pol', 'syn_pol', 'tau2', 'obs_pamp', 'syn_pamp', 'tau3', 'obs_spratio', 'syn_spratio'))
    for i in range(0, len(event.stations)):
        fo.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            event.stations.loc[i]['stid'], event.stations.loc[i]['stlo'], event.stations.loc[i]['stla'],
            event.ray.loc[i]['azi'], event.ray.loc[i]['tko'], event.ray.loc[i]['aoi'], event.ray.loc[i]['dist'],
            event.ray.loc[i]['alpha'], event.ray.loc[i]['beta'], event.ray.loc[i]['rho'],
            event.tau1_raw[i], event.pol_raw[i],     results.syn_pol[0,i],
            event.tau2_raw[i], event.pamp_raw[i],    results.syn_pamp[0,i],
            event.tau3_raw[i], event.spratio_raw[i], results.syn_spratio[0,i] ))
    fo.close()
    # lune points
    fl = open(os.path.join(savepath, event.evid+'_lune.txt'), 'w')
    for i in range(0, len(results.misfit_lune)):
        fl.write("{:3.2f} {:3.2f} {:2.2e}\n".format(results.misfit_lune[i,0],results.misfit_lune[i,1],results.misfit_lune[i,2]))
    fl.close()

def plot_results(savepath, filetype, results, event, weight):
    '''
    Plot function to show beachball for best-fitting moment tensor and the misfit variation on eigenvalue lune projection.
    :param savepath: 
    '''
    azimuth = event.ray['azi'].to_numpy()
    takeoff = event.ray['tko'].to_numpy()    
    pol=None if weight[0]==0 else event.pol_raw
    pamp=None if weight[1]==0 else event.pamp_raw
    spratio=None if weight[2]==0 else event.spratio_raw

    plot_mech(savepath=savepath, filename=event.evid, filetype=filetype, moment_tensor=results.mt_best, 
              best_dc=True, additional_info=False, PTaxis=True,
              beachball_type='full', edgecolor='black', color_t='#f1939c')
    
    plot_mech_data(savepath=savepath, filename=event.evid+'_data', filetype=filetype, moment_tensor=results.mt_best, 
                best_dc=True, PTaxis=True, 
                takeoff=takeoff, azimuth=azimuth,
                pol=pol, pamp=pamp, spratio=spratio,
                beachball_type='full', edgecolor='black', color_t='#f1939c')
    
    plot_misfit_lune_pygmt(savepath=savepath, filename=event.evid+'_lune'+'.'+filetype, 
                           lune_value=results.misfit_lune, lune_marker=[results.gamma, results.delta], 
                           colormap='no_green')
