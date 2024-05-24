import os
import numpy as np
import warnings
import matplotlib
from matplotlib import pyplot as plt
from pyrocko.plot import beachball
from pyrocko import moment_tensor as pmt
from moment_tensor import mom2other, mt_to_moment, mt_to_magnitude, mt_to_delta_gamma

warnings.filterwarnings('ignore') # ignore scatter warning about x marker do not have edgecolor
matplotlib.use('Agg') # do not show figures, only save

def plot_mech(savepath: str, filename: str, filetype: str, 
              moment_tensor: np.ndarray, 
              best_dc: bool=False, additional_info: bool=False, PTaxis: bool=True,
              **kwargs):
    '''
    Creates a plot of the mechanism.

    **kwargs{color_t,
             color_p,
             edgecolor,
             projection,
             linewidth,
             alpha,
             arcces,
             beachball_type,
             view}
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
        # beachball axes
        axes1 = fig.add_subplot(1, 1, 1, aspect=1.0)
        axes1.set_axis_off()
        axes1.set_xlim(-1.1, 1.1)
        axes1.set_ylim(-1.1, 1.1)

    # pyrocke beachball fucntion
    beachball.plot_beachball_mpl(
        mt, 
        axes1,
        position = (0., 0.), 
        size = 2,
        size_units='data',
        **kwargs)
    
    # show nodal lines of best double-couple 
    if best_dc:
        _add_bestDC(axes1, moment_tensor, 
                    linecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black', 
                    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 2)

    # show additional information (nodal plane angles ...)
    if additional_info:
        _add_text(axes2, moment_tensor)

    # show P and T axis
    if PTaxis: 
        projection = kwargs['projection'] if 'projection' in kwargs else 'lambert'
        _add_PTaxis(axes1, moment_tensor, projection)
        
    # save figure
    fig.savefig(
        os.path.join(savepath, filename + '.' + filetype), 
        dpi=600, 
        bbox_inches='tight', 
        facecolor='white', 
        transparent=False)

def plot_mech_data(savepath: str, filename: str, filetype: str, 
              moment_tensor: np.ndarray, best_dc: bool=False, PTaxis: bool=True,
              takeoff: np.ndarray=None, azimuth: np.ndarray=None,
              pol: np.ndarray=None, pamp: np.ndarray=None, spratio: np.ndarray=None,
              **kwargs):
    '''
    Creates a plot of the mechanism with the P-polarity, amplitude, and S/P ratio measurements.

    **kwargs{color_t,
             color_p,
             edgecolor,
             projection,
             linewidth,
             alpha,
             arcces,
             beachball_type,
             view}
    '''
    if takeoff is None or azimuth is None:
        raise ImportError("Error: Missing information about azimuth and takeoff angles.")
    
    mt = pmt.MomentTensor.from_values(moment_tensor.flatten())
    # setup figure with aspect=1.0/1.0, ranges=[-1.1, 1.1]
    fig, ax = plt.subplots(1, 3, figsize=(12., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    for i in range(0, len(ax)):
        ax[i].set_xlim(-1.1, 1.1)
        ax[i].set_ylim(-1.1, 1.1)
        ax[i].set_aspect(1.0)
        ax[i].set_axis_off()

    if pol is None:
        ax[0].text(-0.6, 0, "None P-polarity", {'size': 20})
    else:
        # pyrocke beachball fucntion
        beachball.plot_beachball_mpl(
            mt, ax[0],
            position = (0., 0.), 
            size = 2,
            size_units='data',
            **kwargs)
        _add_pol(ax[0], azimuth, takeoff, pol, 
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
        ax[1].text(-0.6, 0, "None P-amplitude", {'size': 20})
    else:
        # pyrocke beachball fucntion
        beachball.plot_beachball_mpl(
            mt, ax[1],
            position = (0., 0.), 
            size = 2,
            size_units='data',
            **kwargs)
        _add_pamp(ax[1], azimuth, takeoff, pamp, 
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
        ax[2].text(-0.6, 0, "None S/P ratio", {'size': 20})
    else:
        # pyrocke beachball fucntion
        beachball.plot_beachball_mpl(
            mt, ax[2],
            position = (0., 0.), 
            size = 2,
            size_units='data',
            **kwargs)
        _add_spratio(ax[2], azimuth, takeoff, spratio, 
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
    fig.savefig(
        os.path.join(savepath, filename + '.' + filetype), 
        dpi=600, 
        bbox_inches='tight', 
        facecolor='white', 
        transparent=False)
    
def _add_bestDC(axes, moment_tensor: np.ndarray, siz: float=1, n: int=181, linecolor: str='black', linewidth: float=2):
    '''
    from moment tensor (NED) to nodal lines coordinate under 'lambert projection'
    '''
    mt = pmt.MomentTensor.from_values(moment_tensor.flatten())
    (s1, d1, r1), (s2, d2, r2) = mt.both_strike_dip_rake()
    if (d1-90.)<1e-6: d1 += 0.1
    if (d2-90.)<1e-6: d2 += 0.1
    
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

    axes.plot(xs1, ys1, color = linecolor, linewidth=linewidth)
    axes.plot(xs2, ys2, color = linecolor, linewidth=linewidth)    

def _add_text(axes, moment_tensor, fontsize: int=18):

    np1, np2, Ptrpl, Ttrpl, Btrpl = mom2other(moment_tensor)
    moment = mt_to_moment(moment_tensor)
    magnitude = mt_to_magnitude(moment_tensor)
    delta, gamma = mt_to_delta_gamma(moment_tensor)     
      
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

def _add_PTaxis(axes, moment_tensor: np.ndarray, projection: str='lambert', fontsize: int=18):
    _, _, Ptrpl, Ttrpl, _ = mom2other(moment_tensor)
    xP, yP = \
        beachball.project(beachball.numpy_rtp2xyz(np.array([[1.0, np.deg2rad(90-Ptrpl[1]), np.deg2rad(90.-Ptrpl[0])]])), 
                        projection).T
    xT, yT = \
        beachball.project(beachball.numpy_rtp2xyz(np.array([[1.0, np.deg2rad(90-Ttrpl[1]), np.deg2rad(90.-Ttrpl[0])]])), 
                        projection).T
    axes.text(xP, yP, 'P', {'size': fontsize})
    axes.text(xT, yT, 'T', {'size': fontsize})

def _add_pol(axes, azimuth, takeoff, pol, projection: str='lambert', labelsize: int=50):
        
    for i in range(0,len(pol)):
        rtp = np.array([[ 1.0 if takeoff[i] <= 90. else -1., np.deg2rad(takeoff[i]), np.deg2rad(90.-azimuth[i])]])
        # to 3D coordinates (x, y, z)
        point = beachball.numpy_rtp2xyz(rtp)
        # project to 2D with same projection as used in beachball
        x, y = beachball.project(point, projection=projection).T
        if pol[i] == 0.:
            marker, color = 'x', 'black'
        elif pol[i] > 0.:
            marker, color = 'o', 'black'
        else:
            marker, color = 'o', 'white'                 
        axes.scatter(
            x, y,
            s = labelsize,
            c = color,
            marker = marker,
            edgecolors = 'black')

def _add_pamp(axes, azimuth, takeoff, pamp, projection: str='lambert', labelsize: int=100):

    pamp = pamp/np.max(pamp)
    for i in range(0,len(np.abs(pamp))):
        rtp = np.array([[1.0 if takeoff[i] <= 90. else -1., np.deg2rad(takeoff[i]), np.deg2rad(90.-azimuth[i])]])
        # to 3D coordinates (x, y, z)
        point = beachball.numpy_rtp2xyz(rtp)
        # project to 2D with same projection as used in beachball
        x, y = beachball.project(point, projection=projection).T
        if pamp[i] == 0.:
            color, marker, markersize = 'black', 'x', labelsize
        elif pamp[i] > 0.:
            color, marker, markersize = 'black', 'o', np.abs(labelsize*pamp[i])
        else:
            color, marker, markersize = 'white', 'o', np.abs(labelsize*pamp[i])

        axes.scatter(x, y, s=markersize, c=color, marker=marker, edgecolors='black')

def _add_spratio(axes, azimuth, takeoff, spratio, projection: str='lambert', labelsize: int=50):
        
    for i in range(0,len(spratio)):
        rtp = np.array([[1.0 if takeoff[i] <= 90. else -1., np.deg2rad(takeoff[i]), np.deg2rad(90.-azimuth[i])]])
        # to 3D coordinates (x, y, z)
        point = beachball.numpy_rtp2xyz(rtp)
        # project to 2D with same projection as used in beachball
        x, y = beachball.project(point, projection=projection).T
        if spratio[i] == 0:
            marker, markersize = 'x', labelsize
        else:
            marker, markersize = 'o', labelsize*(10**spratio[i])
        axes.scatter(x, y, s=markersize, marker=marker, color='none', edgecolors='black')