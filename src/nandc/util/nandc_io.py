import re
import warnings
import traceback
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pandas import concat, DataFrame

from util.nandc_base import DataIN
from nandc_forward import calc_synthetic
from nandc_mt import (
    magnitude_to_moment, 
    mt_to_gamma_delta, 
    mom2other, 
    standard_decomposition, 
    v_w_to_gamma_delta,
)

warnings.filterwarnings("ignore")

def sanitize_evid(evid):
    evid = str(evid).strip()
    if not evid:
        raise ValueError("event id is empty")
    if evid in {".", ".."}:
        raise ValueError(f"event id is not allowed: {evid}")
    if "/" in evid or "\\" in evid:
        raise ValueError(f"event id contains path separators: {evid}")
    return evid

def read_data(event_list, data_dir, vmodel_TauPy, vmodel_DataFrame, alpha, beta, rho):

    event_list = Path(event_list)
    data_dir = Path(data_dir)

    if not event_list.exists():
        raise FileNotFoundError(f"Can not find {event_list}")
    if not event_list.is_file():
        raise FileNotFoundError(f"Event list is not a file: {event_list}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Can not find data directory: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory is not a directory: {data_dir}")

    # open event list and read event information about evid evlo evla evdp and [mag] (if has)
    rawdatas = []

    # read catalog
    with event_list.open('r') as g:
        glines = g.readlines()

    for gline in tqdm(glines):
        # skip comment line
        if gline.startswith('#'):
            continue

        line = gline.strip()
        if not line:
            continue

        parts = [x for x in re.split(r'[\s,]+', line) if x]

        # catalog format: evid evlo evla evdp [mag]
        if len(parts) == 5:   # evid evlo evla evdp mag
            evid, evlo, evla, evdp, mag = parts
        elif len(parts) == 4: # evid evlo evla evdp
            evid, evlo, evla, evdp = parts
        else:
            print(f" Wrong columns in {gline}, please check it! ")
            continue

        try:
            evid = sanitize_evid(evid)
        except ValueError as e:
            print(f" Skipping invalid event id {evid!r}: {e} ")
            continue

        if len(parts) == 5:
            evinfo = [evid, float(evlo), float(evla), float(evdp), float(mag)]
        else:
            evinfo = [evid, float(evlo), float(evla), float(evdp)]

        # read observations
        filepath = data_dir / f"{evid}.dat"
        if not filepath.exists():
            print(f" {evid}: File {filepath} does not exist! ")
            continue

        try: # avoid open file error to break running
            data = DataFrame(columns=['stid', 'stlo', 'stla', 'stal', 'tau1', 'pol', 'tau2', 'amp', 'tau3', 'spr'])

            with filepath.open('r') as f:
                flines = f.readlines()

            for fline in flines:
                try: # avoid read line error to break running
                    fline_strip = fline.strip()
                    if not fline_strip:
                        continue

                    fparts = [x for x in re.split(r'[\s,]+', fline_strip) if x]

                    if len(fparts) == 10:
                        stid, stlo, stla, stal, tau1, pol, tau2, amp, tau3, spr = fparts
                    elif len(fparts) == 9:
                        stid, stlo, stla, tau1, pol, tau2, amp, tau3, spr = fparts
                        stal = '0.0'
                    else:
                        print(f" {evid}: Wrong columns in {fline}, please check it! ")
                        continue

                    data = concat([data, DataFrame(
                                    {'stid': stid, 'stlo': [float(stlo)], 'stla': [float(stla)], 'stal': [float(stal)],
                                     'tau1': [float(tau1)], 'pol': [int(float(pol))],
                                     'tau2': [float(tau2)], 'amp': [float(amp)],
                                     'tau3': [float(tau3)], 'spr': [float(spr)]})], ignore_index=True)

                except (ValueError, TypeError) as e:
                    print(f" {evid}: {fline}, {e} ")
                    continue

            # check data
            if len(data) == 0:
                print(f" {evid}: No data found in file {filepath}! ")
                continue

            rawdata = DataIN(data, evinfo, vmodel_TauPy, vmodel_DataFrame, alpha, beta, rho)
            if len(rawdata.pol) == 0 and len(rawdata.amp) == 0 and len(rawdata.spr) == 0:
                print(f" {evid}: All observations in {filepath} are zero! ")
                continue

            rawdatas.append(rawdata)
        except (OSError, UnicodeDecodeError) as e:
            print(f"{evid}: Failed to read file {filepath}, {e}")
            traceback.print_exc()
            continue
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"{evid}: Failed to build event from {filepath}, {e}")
            traceback.print_exc()
            continue

    return rawdatas

def read_nd_model(vmodel):
    vmodel = Path(vmodel)
    rows = []
    with vmodel.open('r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                z   = float(parts[0])
                vp  = float(parts[1])
                vs  = float(parts[2])
                rho = float(parts[3])
                rows.append([z, vp, vs, rho])
            except ValueError:
                continue

    return DataFrame(rows, columns=['depth', 'vp', 'vs', 'rho'])

def write_results(savepath, event, mt_best, magnitude, min_misfit, weight_list):

    write_file_data(savepath, event.evid+'_data.csv', event, mt_best, weight_list)
    write_file_meca(savepath, event.evid+'_meca.txt', event, mt_best, magnitude, min_misfit)
    write_file_decomposition(savepath, event.evid+'_decom.txt', event, mt_best, magnitude)

def write_file_data(savepath, filename, event, mt_best, weight_list):

    syn_amp, syn_spr = calc_synthetic(mts=mt_best, supplement=event.ray_array)
    syn_pol = np.sign(syn_amp)

    # data
    with (savepath / filename).open('w') as fo:
        fo.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            'stid', 'stlo', 'stla', 'azi', 'tko', 'aoi', 'dist (m)', 'alpha (m/s)', 'beta (m/s)', 'rho (kg/m3)',
            'tau1', 'obs_pol', 'syn_pol', 'pol_mismatch',
            'tau2', 'obs_amp', 'syn_amp', 'amp_pct_res',
            'tau3', 'obs_spr', 'syn_spr', 'spr_pct_res'))

        for i in range(0, len(event.stations)):
            obs_pol = event.pol_raw[i]
            obs_amp = event.amp_raw[i]
            obs_spr = event.spr_raw[i]

            this_syn_pol = syn_pol[0, i]
            this_syn_amp = syn_amp[0, i]
            this_syn_spr = syn_spr[0, i]

            if weight_list[0] == 0 or obs_pol == 0:
                pol_mismatch = 0
            else:
                pol_mismatch = int(obs_pol != this_syn_pol)

            if weight_list[1] == 0 or obs_amp == 0:
                amp_pct_res = 0.0
            else:
                amp_pct_res = 100.0 * (obs_amp - this_syn_amp) / abs(obs_amp)

            if weight_list[2] == 0 or obs_spr == 0:
                spr_pct_res = 0.0
            else:
                spr_pct_res = 100.0 * (obs_spr - this_syn_spr) / abs(obs_spr)

            fo.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                event.stations.loc[i]['stid'], event.stations.loc[i]['stlo'], event.stations.loc[i]['stla'],
                event.ray.loc[i]['azi'], event.ray.loc[i]['tko'], event.ray.loc[i]['aoi'], event.ray.loc[i]['dist'],
                event.ray.loc[i]['alpha'], event.ray.loc[i]['beta'], event.ray.loc[i]['rho'],
                event.tau1_raw[i], obs_pol, this_syn_pol, pol_mismatch,
                event.tau2_raw[i], obs_amp, this_syn_amp, amp_pct_res,
                event.tau3_raw[i], obs_spr, this_syn_spr, spr_pct_res ))

def write_file_meca(savepath, filename, event, mt_best, magnitude, min_misfit):

    if magnitude:
        moment, magnitude = "{:.3e}".format(magnitude_to_moment(magnitude)), str(round(magnitude, 2))
    else:
        moment, magnitude = "NAN", "NAN"

    # meca results
    with (savepath / filename).open('w') as fm:
        tplt1 = "{:>20} {:>12} {:>12} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}\n"
        fm.write(tplt1.format('evid', 'evlo', 'evla', 'evdp', 'Mw', 'moment', 'Mxx', 'Myy', 'Mzz', 'Mxy', 'Mxz', 'Myz', 'misfit'))
        fm.write(tplt1.format(' ', ' ', ' ', 'km', ' ', 'N-m', 'N-m', 'N-m', 'N-m', 'N-m', 'N-m', 'N-m', ' ', ' ', ' '))

        tplt2 = "{:>20s} {:>12.6f} {:>12.6f} {:>6.3f} {:>6s} {:>10s} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.3e}\n"
        fm.write(tplt2.format(event.evid, event.evlo, event.evla, event.evdp, magnitude, moment,
                            mt_best[0,0], mt_best[0,1], mt_best[0,2], mt_best[0,3], mt_best[0,4], mt_best[0,5],
                            min_misfit))

def write_file_decomposition(savepath, filename, event, mt_best, magnitude):

    magnitude = str(round(magnitude, 2)) if magnitude else "NAN"

    best_gamma, best_delta = mt_to_gamma_delta(mt_best)
    mt_iso, mt_clvd, mt_dc, piso, pclvd, pdc, eps = standard_decomposition(mt_best)
    np1, np2, Ptrpl, Ttrpl, Btrpl = mom2other(mt_best)
    nodal_plane_1 = '/'.join((str(round(np1[0],0)), str(round(np1[1],0)), str(round(np1[2],0))))
    nodal_plane_2 = '/'.join((str(round(np2[0],0)), str(round(np2[1],0)), str(round(np2[2],0))))

    # decomposition results
    with (savepath / filename).open('w') as fd:
        fd.write("{:>20} {:>12} {:>12} {:>6} {:>6} {:>18s} {:>18s} {:>9} {:>9} {:>6} {:>6} {:>6}\n".format(
            'evid', 'evlo', 'evla', 'evdp', 'Mw', 'NP1(str/dip/rake)', 'NP2', 'lune-lon.', 'lune-lat.', 'pdc', 'pclvd', 'piso'))

        tplt = "{:>20s} {:>12.6f} {:>12.6f} {:>6.3f} {:>6s} {:>18s} {:>18s} {:>9.1f} {:>9.1f} {:>6.1f} {:>6.1f} {:>6.1f}\n"
        fd.write(tplt.format(event.evid, event.evlo, event.evla, event.evdp, magnitude, nodal_plane_1, nodal_plane_2, best_gamma, best_delta, pdc, pclvd, piso))

def write_file_lune(savepath, filename, lune_values):

    # results within misfit thershold
    with (savepath / filename).open('w') as fl:
        fl.write("{:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12}\n".format(
            'rho', 'v', 'w', 'kappa', 'sigma', 'h', 'lune_lon', 'lune_lat', 'misfit'))

        tplt = "{:>10.4e} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>10.2f} {:>10.2f} {:>12.4e}\n"
        for i in range(len(lune_values)):
            gamma, delta = v_w_to_gamma_delta(lune_values[i,1], lune_values[i,2])
            fl.write(tplt.format(lune_values[i,0], lune_values[i,1], lune_values[i,2],
                                lune_values[i,3], lune_values[i,4], lune_values[i,5],
                                gamma, delta, lune_values[i,6]))
