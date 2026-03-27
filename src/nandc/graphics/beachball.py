import warnings
import matplotlib
matplotlib.use('Agg') # do not show figures, only save
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pyrocko.plot import beachball
from pyrocko import moment_tensor as pmt

from nandc_mt import mom2other, mt_to_moment, mt_to_magnitude, mt_to_gamma_delta
from nandc_forward import calc_synthetic

warnings.filterwarnings('ignore') # ignore scatter warning about x marker do not have edgecolor

def _project_station_points(azimuth, takeoff, projection='lambert'):

    azimuth = np.asarray(azimuth, dtype=float).reshape(-1)
    takeoff = np.asarray(takeoff, dtype=float).reshape(-1)
    if azimuth.size != takeoff.size:
        raise ValueError("Error: azimuth and takeoff must have the same length.")

    invalid_takeoff = np.isfinite(takeoff) & ((takeoff < 0.0) | (takeoff > 180.0))
    if invalid_takeoff.any():
        raise ValueError("Error: takeoff angles must be within [0, 180] degrees.")

    radius = np.where(takeoff <= 90., 1.0, -1.0)
    rtp = np.column_stack((
        radius,
        np.deg2rad(takeoff),
        np.deg2rad(90. - azimuth),
    ))
    points = beachball.numpy_rtp2xyz(rtp)
    x, y = beachball.project(points, projection=projection).T
    return x, y

def _prepare_station_overlay_data(azimuth, takeoff, values, stid=None, value_name='values'):

    azimuth = np.asarray(azimuth, dtype=float).reshape(-1)
    takeoff = np.asarray(takeoff, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size != azimuth.size:
        raise ValueError(f"Error: {value_name} must have the same length as azimuth and takeoff.")
    if stid is not None and len(stid) != azimuth.size:
        raise ValueError("Error: stid must have the same length as azimuth and takeoff.")

    return azimuth, takeoff, values

def _build_relative_residual_norm(residual):

    abs_residual = np.abs(residual[np.isfinite(residual)])
    vmax = np.percentile(abs_residual, 95) if abs_residual.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax < 1e-20:
        vmax = 1.0

    return Normalize(vmin=-vmax, vmax=vmax)

def _add_pol(axes, azimuth, takeoff, pol, skipzero, stid=None, 
             station_name=False, projection='lambert', labelsize=50):

    azimuth, takeoff, pol = _prepare_station_overlay_data(
        azimuth, takeoff, pol, stid=stid, value_name='pol')
    x, y = _project_station_points(azimuth, takeoff, projection=projection)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(pol)
    positive = finite & (pol > 0.0)
    negative = finite & (pol < 0.0)
    zero_obs = finite & (pol == 0.0)

    if zero_obs.any() and not skipzero:
        axes.scatter(x[zero_obs], y[zero_obs], s=labelsize, c='black', marker='x')

    if positive.any():
        axes.scatter(x[positive], y[positive], s=labelsize, c='black',
                     marker='o', edgecolors='black')

    if negative.any():
        axes.scatter(x[negative], y[negative], s=labelsize, c='white',
                     marker='o', edgecolors='black')

    if station_name and (stid is not None):
        plotted = positive | negative | (zero_obs & ~skipzero)
        for idx in np.flatnonzero(plotted):
            axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

def _add_pamp(axes, azimuth, takeoff, pamp, skipzero, stid=None, 
              station_name=False, projection='lambert', labelsize=40):

    azimuth, takeoff, pamp = _prepare_station_overlay_data(
        azimuth, takeoff, pamp, stid=stid, value_name='pamp')
    x, y = _project_station_points(azimuth, takeoff, projection=projection)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(pamp)
    nonzero = finite & (pamp != 0.0)
    positive = finite & (pamp > 0.0)
    negative = finite & (pamp < 0.0)
    zero_obs = finite & (pamp == 0.0)

    nonzero_amp = np.abs(pamp[nonzero])
    scale = np.mean(nonzero_amp) if nonzero_amp.size > 0 else 1.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    sizes = np.zeros_like(pamp, dtype=float)
    sizes[nonzero] = np.abs(labelsize * pamp[nonzero] / scale)
    positive = positive & np.isfinite(sizes) & (sizes > 0.0)
    negative = negative & np.isfinite(sizes) & (sizes > 0.0)

    if zero_obs.any() and not skipzero:
        axes.scatter(x[zero_obs], y[zero_obs], s=labelsize, c='black', marker='x')

    if positive.any():
        axes.scatter(x[positive], y[positive], s=sizes[positive], c='black',
                     marker='o', edgecolors='black')

    if negative.any():
        axes.scatter(x[negative], y[negative], s=sizes[negative], c='white',
                     marker='o', edgecolors='black')

    if station_name and stid is not None:
        plotted = positive | negative | (zero_obs & ~skipzero)
        for idx in np.flatnonzero(plotted):
            axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

def _add_spratio(axes, azimuth, takeoff, spratio, skipzero, stid=None, 
                 station_name=False, projection='lambert', labelsize=40):

    azimuth, takeoff, spratio = _prepare_station_overlay_data(
        azimuth, takeoff, spratio, stid=stid, value_name='spratio')
    x, y = _project_station_points(azimuth, takeoff, projection=projection)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(spratio)
    nonzero = finite & (spratio != 0.0)
    zero_obs = finite & (spratio == 0.0)

    sizes = np.zeros_like(spratio, dtype=float)
    sizes[nonzero] = labelsize * (10.0 ** spratio[nonzero])
    nonzero = nonzero & np.isfinite(sizes) & (sizes > 0.0)

    if zero_obs.any() and not skipzero:
        axes.scatter(x[zero_obs], y[zero_obs], s=labelsize, c='black', marker='x')

    if nonzero.any():
        axes.scatter(x[nonzero], y[nonzero], s=sizes[nonzero], marker='o',
                     color='none', edgecolors='black')

    if station_name and stid is not None:
        plotted = nonzero | (zero_obs & ~skipzero)
        for idx in np.flatnonzero(plotted):
            axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

def _add_pamp_residual(axes, azimuth, takeoff, pamp, pamp_residual, skipzero, stid=None,
                       station_name=False, projection='lambert', labelsize=40,
                       cmap='bwr'):

    azimuth, takeoff, pamp = _prepare_station_overlay_data(
        azimuth, takeoff, pamp, stid=stid, value_name='pamp')
    x, y = _project_station_points(azimuth, takeoff, projection=projection)

    finite = np.isfinite(pamp) & np.isfinite(pamp_residual)
    nonzero = finite & (pamp != 0.0)
    zero_obs = finite & (pamp == 0.0)

    nonzero_amp = np.abs(pamp[nonzero])
    scale = np.mean(nonzero_amp) if nonzero_amp.size > 0 else 1.0
    if not np.isfinite(scale):
        scale = 1.0

    if zero_obs.any() and not skipzero:  # 如果有零值且skipzero不为True
        axes.scatter(x[zero_obs], y[zero_obs], s=labelsize, c='black', marker='x')
        if station_name and stid is not None:
            for idx in np.where(zero_obs)[0]:
                axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

    if not nonzero.any():
        return None

    sizes = np.abs(labelsize * pamp[nonzero] / scale)
    scatter = axes.scatter(x[nonzero], y[nonzero], s=sizes, c=pamp_residual[nonzero], marker='o',
                           cmap=cmap, norm=_build_relative_residual_norm(pamp_residual[nonzero]),
                           edgecolors='black')

    if station_name and stid is not None:
        plotted = nonzero | (zero_obs & ~skipzero)
        for idx in np.flatnonzero(plotted):
            axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

    return scatter

def _add_spratio_residual(axes, azimuth, takeoff, spratio, spratio_residual, skipzero, stid=None,
                          station_name=False, projection='lambert', labelsize=40,
                          cmap='bwr'):

    azimuth, takeoff, spratio = _prepare_station_overlay_data(
        azimuth, takeoff, spratio, stid=stid, value_name='spratio')
    x, y = _project_station_points(azimuth, takeoff, projection=projection)

    finite = np.isfinite(spratio) & np.isfinite(spratio_residual)
    nonzero = finite & (spratio != 0.0)
    zero_obs = finite & (spratio == 0.0)

    if zero_obs.any() and not skipzero:  
        axes.scatter(x[zero_obs], y[zero_obs], s=labelsize, c='black', marker='x')
        if station_name and stid is not None:
            for idx in np.where(zero_obs)[0]:
                axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

    if not nonzero.any():
        return None

    sizes = labelsize * (10 ** spratio[nonzero])
    scatter = axes.scatter(x[nonzero], y[nonzero], s=sizes, c=spratio_residual[nonzero], marker='o',
                           cmap=cmap, norm=_build_relative_residual_norm(spratio_residual[nonzero]),
                           edgecolors='black')

    if station_name and stid is not None:
        for idx in np.where(nonzero)[0]:
            axes.annotate(stid[idx], xy=(x[idx], y[idx]), xytext=(0, -8),
                      textcoords="offset points",
                      ha='center', va='top', fontsize=8, color='black')

    return scatter

def plot_mech(savepath, filename, filetype, moment_tensor,
              pol=None, pamp=None, spratio=None, takeoff=None, azimuth=None,
              best_dc=False, additional_info=False, PTaxis=True, skipzero=True,
              stid=None, station_name=False, **kwargs):
    '''
    Creates a plot of the mechanism.
    **kwargs{color_t, color_p, edgecolor, projection, linewidth, alpha, arcces, beachball_type, view}
    '''
    mt = pmt.MomentTensor.from_values(moment_tensor.flatten())

    # setup figure with aspect=1.0/1.0, ranges=[-1.1, 1.1]
    fig = plt.figure(figsize=(6., 6.))  # size in inch
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)    
    if additional_info:
        # beachball axes
        axes1 = fig.add_subplot(1, 2, 2, aspect=1.0)
        axes1.set_axis_off()
        axes1.set_xlim(-1.1, 1.1)
        axes1.set_ylim(-1.1, 1.1)
        # additional information axes
        axes2 = fig.add_subplot(1, 2, 1, aspect=1.0)
        axes2.set_axis_off()
        axes2.set_xlim(-1.1, 0.8)
        axes2.set_ylim(-1.1, 1.1)
    else:
        # only create beachball axes
        axes1 = fig.add_subplot(1, 1, 1, aspect=1.0)
        axes1.set_axis_off()
        axes1.set_xlim(-1.1, 1.1)
        axes1.set_ylim(-1.1, 1.1)

    # pyrocke beachball fucntion
    beachball.plot_beachball_mpl(mt, axes1, position=(0., 0.), size=2, size_units='data', **kwargs)
    
    if pol is not None:
        if takeoff is None or azimuth is None:
            raise ValueError("Error: Missing information about azimuth and takeoff angles.")
        else:
            _add_pol(axes1, azimuth, takeoff, pol, skipzero, stid, station_name,
                     projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
    
    if pamp is not None:
        if takeoff is None or azimuth is None:
            raise ValueError("Error: Missing information about azimuth and takeoff angles.")
        else:
            _add_pamp(axes1, azimuth, takeoff, pamp, skipzero, stid, station_name,
                      projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
    
    if spratio is not None:
        if takeoff is None or azimuth is None:
            raise ValueError("Error: Missing information about azimuth and takeoff angles.")
        else:
            _add_spratio(axes1, azimuth, takeoff, spratio, skipzero, stid, station_name,
                         projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
    
    # show nodal lines of best double-couple 
    if best_dc:
        _add_bestDC(axes1, moment_tensor, siz=1,
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black', 
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)

    # show additional information (nodal plane angles ...)
    if additional_info: 
        _add_text(axes2, moment_tensor)

    # show P and T axis
    if PTaxis:
        _add_PTaxis(axes1, moment_tensor, kwargs['projection'] if 'projection' in kwargs else 'lambert')

    # save figure
    fig.savefig(savepath / f"{filename}.{filetype}", dpi=300, bbox_inches='tight', facecolor='white', transparent=False)

def plot_mech_mpl(savepath, filename, filetype, mts, best_mt, additional_info=False, **kwargs):
    '''
    '''
    best_mt__ = pmt.MomentTensor.from_values(best_mt.flatten())
    mts__ = []
    for i in range(0,len(mts)):
        mts__.append(pmt.MomentTensor.from_values(mts[i,:]))

    # setup figure with aspect=1.0/1.0, ranges=[-1.1, 1.1]
    fig = plt.figure(figsize=(6., 6.))  # size in inch
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)    

    if additional_info:
        # beachball axes
        axes1 = fig.add_subplot(1, 2, 2, aspect=1.0)
        axes1.set_axis_off()
        axes1.set_xlim(-1.1, 1.1)
        axes1.set_ylim(-1.1, 1.1)
        # additional information axes
        axes2 = fig.add_subplot(1, 2, 1, aspect=1.0)
        axes2.set_axis_off()
        axes2.set_xlim(-1.1, 0.8)
        axes2.set_ylim(-1.1, 1.1)
    else:
        # only create beachball axes
        axes1 = fig.add_subplot(1, 1, 1, aspect=1.0)
        axes1.set_axis_off()
        axes1.set_xlim(-1.1, 1.1)
        axes1.set_ylim(-1.1, 1.1)

    beachball.plot_fuzzy_beachball_mpl_pixmap(mts__, axes1, best_mt__, position=(0., 0.), size=2, size_units='data', grid_resolution=1000, **kwargs)
    fig.savefig(savepath / f"{filename}.{filetype}", dpi=300, bbox_inches='tight', facecolor='white', transparent=False)

def plot_mech_data(savepath, filename, filetype, 
              moment_tensor, best_dc=False, PTaxis=True,
              takeoff=None, azimuth=None, pol=None, pamp=None, spratio=None, skipzero=True, 
              stid=None, station_name=False, **kwargs):
    '''
    Creates a plot of the mechanism with the P-polarity, amplitude, and S/P ratio measurements.
    **kwargs{color_t, color_p, edgecolor, projection, linewidth, alpha, arcces, beachball_type, view}
    '''
    if takeoff is None or azimuth is None:
        raise ValueError("Error: Missing information about azimuth and takeoff angles.")
    
    mt = pmt.MomentTensor.from_values(moment_tensor.flatten())

    fig, ax = plt.subplots(1, 3, figsize=(12., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    for i in range(0, len(ax)):
        ax[i].set_xlim(-1.1, 1.1)
        ax[i].set_ylim(-1.1, 1.1)
        ax[i].set_aspect(1.0)
        ax[i].set_axis_off()

    if pol is None:
        ax[0].text(-0.6, 0, "No P-polarity", {'size': 20})
    else:
        # pyrocke beachball fucntion
        beachball.plot_beachball_mpl(mt, ax[0], position=(0., 0.), size=2, size_units='data', **kwargs)
        _add_pol(ax[0], azimuth, takeoff, pol, skipzero, stid, station_name,
                 projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
        # show nodal lines of best double-couple 
        if best_dc:
            _add_bestDC(ax[0], moment_tensor, 
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black', 
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)
        # show P and T axis
        if PTaxis: 
            _add_PTaxis(ax[0], moment_tensor, projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
    
    if pamp is None:
        ax[1].text(-0.6, 0, "No P-amplitude", {'size': 20})
    else:
        # pyrocke beachball fucntion
        beachball.plot_beachball_mpl(mt, ax[1], position=(0., 0.), size=2, size_units='data', **kwargs)
        _add_pamp(ax[1], azimuth, takeoff, pamp, skipzero, stid, station_name=False,
                  projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
        # show nodal lines of best double-couple 
        if best_dc:
            _add_bestDC(ax[1], moment_tensor, 
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black', 
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)
        # show P and T axis
        if PTaxis: 
            _add_PTaxis(ax[1], moment_tensor, projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')    

    if spratio is None:
        ax[2].text(-0.6, 0, "No S/P ratio", {'size': 20})
    else:
        # pyrocke beachball fucntion
        beachball.plot_beachball_mpl(mt, ax[2], position=(0., 0.), size=2, size_units='data', **kwargs)
        _add_spratio(ax[2], azimuth, takeoff, spratio, skipzero, stid, station_name=False,
                     projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')
        # show nodal lines of best double-couple 
        if best_dc:
            _add_bestDC(ax[2], moment_tensor, 
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black', 
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)
        # show P and T axis
        if PTaxis: 
            _add_PTaxis(ax[2], moment_tensor, projection=kwargs['projection'] if 'projection' in kwargs else 'lambert')

    # save figure
    fig.savefig(savepath / f"{filename}.{filetype}", dpi=300, bbox_inches='tight', facecolor='white', transparent=False)

def plot_mech_residual_data(savepath, filename, filetype,
              moment_tensor, best_dc=False, PTaxis=True,
              takeoff=None, azimuth=None, pol=None, pamp=None, spratio=None,
              pamp_residual=None, spratio_residual=None, skipzero=True,
              stid=None, station_name=False, **kwargs):
    '''
    Creates a plot of the mechanism with polarity observations and relative residuals
    for P-wave amplitude and S/P ratio.
    **kwargs{color_t, color_p, edgecolor, projection, linewidth, alpha, arcces, beachball_type, view}
    '''
    if takeoff is None or azimuth is None:
        raise ValueError("Error: Missing information about azimuth and takeoff angles.")

    projection = kwargs['projection'] if 'projection' in kwargs else 'lambert'
    mt = pmt.MomentTensor.from_values(moment_tensor.flatten())

    fig, ax = plt.subplots(1, 3, figsize=(12., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0.08, top=1.)
    for i in range(0, len(ax)):
        ax[i].set_xlim(-1.1, 1.1)
        ax[i].set_ylim(-1.1, 1.1)
        ax[i].set_aspect(1.0)
        ax[i].set_axis_off()

    if pol is None:
        ax[0].text(-0.6, 0, "No P-polarity", {'size': 20})
    else:
        beachball.plot_beachball_mpl(mt, ax[0], position=(0., 0.), size=2, size_units='data', **kwargs)
        _add_pol(ax[0], azimuth, takeoff, pol, skipzero, stid, station_name, projection=projection)
        if best_dc:
            _add_bestDC(ax[0], moment_tensor,
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black',
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)
        if PTaxis:
            _add_PTaxis(ax[0], moment_tensor, projection=projection)

    amp_scatter = None
    if pamp is None or pamp_residual is None:
        ax[1].text(-0.6, 0, "No P-amplitude", {'size': 20})
    else:
        beachball.plot_beachball_mpl(mt, ax[1], position=(0., 0.), size=2, size_units='data', **kwargs)
        amp_scatter = _add_pamp_residual(ax[1], azimuth, takeoff, pamp, pamp_residual, skipzero,
                                         stid, False, projection=projection)
        if best_dc:
            _add_bestDC(ax[1], moment_tensor,
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black',
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)
        if PTaxis:
            _add_PTaxis(ax[1], moment_tensor, projection=projection)

    spr_scatter = None
    if spratio is None or spratio_residual is None:
        ax[2].text(-0.6, 0, "No S/P ratio", {'size': 20})
    else:
        beachball.plot_beachball_mpl(mt, ax[2], position=(0., 0.), size=2, size_units='data', **kwargs)
        spr_scatter = _add_spratio_residual(ax[2], azimuth, takeoff, spratio, spratio_residual, skipzero,
                                            stid, False, projection=projection)
        if best_dc:
            _add_bestDC(ax[2], moment_tensor,
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black',
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)
        if PTaxis:
            _add_PTaxis(ax[2], moment_tensor, projection=projection)

    if amp_scatter is not None:
        cbar = fig.colorbar(amp_scatter, ax=ax[1], orientation='horizontal',
                            fraction=0.05, pad=0.03)
        cbar.set_label('|log10(A_obs)|-|log10(A_syn)|')

    if spr_scatter is not None:
        cbar = fig.colorbar(spr_scatter, ax=ax[2], orientation='horizontal',
                            fraction=0.05, pad=0.03)
        cbar.set_label('(S/P)_obs-(S/P)_syn')

    fig.savefig(savepath / f"{filename}.{filetype}", dpi=300, bbox_inches='tight',
                facecolor='white', transparent=False)
    
def _add_bestDC(axes, moment_tensor, siz=1, n=181, linecolor='black', linewidth=2):
    '''
    from moment tensor (NED) to nodal lines coordinate under 'lambert projection'
    '''
    np1, np2, _, _, _ = mom2other(moment_tensor)
    s1, d1, s2, d2 = np1[0], np1[1], np2[0], np2[1]
    if np.isclose(d1, 90.0, atol=1e-6): d1 += 0.1
    if np.isclose(d2, 90.0, atol=1e-6): d2 += 0.1
    
    rak = np.linspace(0, -np.pi, n)
    str, dip = np.deg2rad(s1), np.deg2rad(d1)
    cosih = -np.sin(dip)*np.sin(rak)
    ih = np.arccos(cosih)
    cosdet = np.sqrt(1-cosih**2)
    fai = np.arccos(np.cos(rak)/cosdet)
    str1 = str+fai
    xs1 = siz * np.sqrt(2) * np.sin(ih/2) * np.sin(str1)
    ys1 = siz * np.sqrt(2) * np.sin(ih/2) * np.cos(str1)

    str2, dip2 = np.deg2rad(s2), np.deg2rad(d2)
    cosih = -np.sin(dip2)*np.sin(rak)
    ih = np.arccos(cosih)
    cosdet = np.sqrt(1-cosih**2)
    fai = np.arccos(np.cos(rak)/cosdet)
    str21 = str2+fai
    xs2 = siz * np.sqrt(2) * np.sin(ih/2) * np.sin(str21)
    ys2 = siz * np.sqrt(2) * np.sin(ih/2) * np.cos(str21)

    axes.plot(xs1, ys1, color=linecolor, linewidth=linewidth)
    axes.plot(xs2, ys2, color=linecolor, linewidth=linewidth)    

def _add_text(axes, moment_tensor, fontsize=18):

    np1, np2, Ptrpl, Ttrpl, Btrpl = mom2other(moment_tensor)
    moment = mt_to_moment(moment_tensor)
    magnitude = mt_to_magnitude(moment_tensor)
    gamma, delta  = mt_to_gamma_delta(moment_tensor)     
      
    text1 = [[-1,-1,-1,-1,-1,-1,-1], 
             [0.75,0.55,0.1,-0.1,-0.3,-0.55,-0.75],
            ['NP1:','NP2:','T:','P:','B:','M0:','Mw:']]
    for i in range(0, 7):
        axes.text(text1[0][i], text1[1][i], text1[2][i], {'size': fontsize})
    text2 = [[-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5], 
            [0.95,0.75,0.55,0.3,0.1,-0.1,-0.3,-0.55,-0.75],
            ['strike',round(np1[0],1),round(np2[0],1),'Azm',
             round(Ttrpl[0],1),round(Ptrpl[0],1),round(Btrpl[0],1),
             '{:.2e} Nm'.format(moment),round(magnitude,1)]]
    for i in range(0,9):
        axes.text(text2[0][i], text2[1][i], text2[2][i], {'size': fontsize})
    text3 = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1],
            [0.95,0.75,0.55,0.3,0.1,-0.1,-0.3],
            ['rake',round(np1[2],1),round(np2[2],1),'Plg',
             round(Ttrpl[1],1),round(Ptrpl[1],1),round(Btrpl[1],1)]]
    for i in range(0,7):
        axes.text(text3[0][i], text3[1][i], text3[2][i], {'size': fontsize})
    text4 = [[0.7,0.7,0.7],
             [0.95,0.75,0.55],
            ['dip',round(np1[1],1),round(np2[1],1)]]
    for i in range(0,3):
        axes.text(text4[0][i], text4[1][i], text4[2][i], {'size': fontsize})
    axes.text(-1, -0.95, '{}:   {:.1f}  {}:   {:.1f}'.format(chr(947),gamma,chr(948),delta), {'size': fontsize})

def _add_PTaxis(axes, moment_tensor, projection='lambert', fontsize=22):

    _, _, Ptrpl, Ttrpl, _ = mom2other(moment_tensor)
    xP, yP = \
        beachball.project(beachball.numpy_rtp2xyz(np.array([[1.0, np.deg2rad(90-Ptrpl[1]), np.deg2rad(90.-Ptrpl[0])]])), 
                        projection).T
    xT, yT = \
        beachball.project(beachball.numpy_rtp2xyz(np.array([[1.0, np.deg2rad(90-Ttrpl[1]), np.deg2rad(90.-Ttrpl[0])]])), 
                        projection).T
    axes.text(xP[0], yP[0], 'P', {'size': fontsize})
    axes.text(xT[0], yT[0], 'T', {'size': fontsize})

def plot_results_beachball(savepath, args, event, weight_list, mt_best):

    # ------------------------------------- beachball ---------------------------------------- #
    plot_mech(savepath=savepath, filename=event.evid, filetype=args.figtype, moment_tensor=mt_best, 
              best_dc=True, additional_info=False, PTaxis=True, 
              beachball_type='full', edgecolor='black', color_t='#f1939c')

    azimuth = event.ray['azi'].to_numpy()
    takeoff = event.ray['tko'].to_numpy()    
    pol     = None if weight_list[0]==0 or len(event.pol) == 0 else event.pol_raw
    amp     = None if weight_list[1]==0 or len(event.amp) == 0 else event.amp_raw
    spr     = None if weight_list[2]==0 or len(event.spr) == 0 else event.spr_raw
    amp_plot = None if amp is None else amp * event.ray['dist'].to_numpy()

    plot_mech_data(savepath=savepath, filename=event.evid+'_data', filetype=args.figtype, moment_tensor=mt_best, 
                   best_dc=True, PTaxis=True, takeoff=takeoff, azimuth=azimuth,
                   pol=pol, pamp=amp_plot, spratio=spr, skipzero=args.skipzero,
                   stid=event.stations['stid'].to_list(), station_name=args.plot_station_names,
                   beachball_type='full', edgecolor='black', color_t='#f1939c')

    amp_residual = None
    spr_residual = None
    if amp is not None or spr is not None:
        syn_amp, syn_spr = calc_synthetic(mts=mt_best, supplement=event.ray_array)
        syn_amp = np.asarray(syn_amp, dtype=float).reshape(-1)
        syn_spr = np.asarray(syn_spr, dtype=float).reshape(-1)

        if amp is not None:
            if syn_amp.size != amp.size:
                raise ValueError("Error: syn_pamp must have the same length as pamp.")

            amp_residual = np.full(amp.shape, np.nan, dtype=float)
            valid_amp = (
                np.isfinite(amp) & np.isfinite(syn_amp) &
                (amp != 0.0) & (syn_amp != 0.0)
            )
            amp_residual[valid_amp] = (
                np.log10(np.abs(amp[valid_amp])) -
                np.log10(np.abs(syn_amp[valid_amp]))
            )

        if spr is not None:
            if syn_spr.size != spr.size:
                raise ValueError("Error: syn_spratio must have the same length as spratio.")

            spr_residual = np.full(spr.shape, np.nan, dtype=float)
            valid_spr = np.isfinite(spr) & np.isfinite(syn_spr)
            spr_residual[valid_spr] = spr[valid_spr] - syn_spr[valid_spr]

    plot_mech_residual_data(savepath=savepath, filename=event.evid+'_data_residual',
                            filetype=args.figtype, moment_tensor=mt_best,
                            best_dc=True, PTaxis=True, takeoff=takeoff, azimuth=azimuth,
                            pol=pol, pamp=amp_plot, spratio=spr,
                            pamp_residual=amp_residual,
                            spratio_residual=spr_residual, skipzero=args.skipzero,
                            stid=event.stations['stid'].to_list(),
                            station_name=True,
                            beachball_type='full', edgecolor='black', color_t="#a19e9e")
