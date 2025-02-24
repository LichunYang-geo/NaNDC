import os
import pygmt
import numpy as np
import xarray as xr
import scipy.interpolate as ipl

def plot_misfit_lune_pygmt(savepath, filename, 
                           lune_value, lune_limits=None, lune_marker=None,
                           colormap = "no_green"):
    
    '''
    plot misfit value and contours on eigenvalue lune by pygmt
    :param savepath: figure savepath
    :type savepath: str
    :param filename: figure savename
    :type filename: str
    :param lune_value: misfit values plot on lune, N*3 [lon,lat,values] 2-D array
    :type lune_value: np.ndarray
    :param lune_limits: misfit area for visualization
    :type lune_limits: np.ndarray or list [lower, upper]
    :param lune_marker: marker's longitude and latitude on eigenvalue lune
    :type lune_marker: np.ndarray [gamma, delta]
    '''

    if not os.path.exists(savepath): os.makedirs(savepath)
    
    if lune_limits is not None:
        ind = np.where((lune_value[:,2]<=lune_limits[1]) & (lune_value[:,2]>=lune_limits[0]))
        lune_value = lune_value[ind]

    lune_value = np.unique(lune_value, axis=0)
    minval, maxval = np.min(lune_value[:,-1]), np.max(lune_value[:,-1])
    cpt_step=((maxval-minval)/100)
    contours = [minval+2*cpt_step, minval+5*cpt_step, minval+10*cpt_step]

    grid_x, grid_y = np.mgrid[-30:30:60j, -90:90:180j]
    grid_z0 = ipl.griddata(points=lune_value[:,[0,1]], values=lune_value[:,2], xi=(grid_x, grid_y), method='linear')
    grid = xr.DataArray(data=grid_z0.T, dims=("lon","lat"), coords={"lon":grid_y[0,:], "lat":grid_x[:,0]})

    pygmt.config(
        MAP_FRAME_TYPE = "plain",
        MAP_FRAME_PEN  = "0.5p,black",
        MAP_GRID_PEN   = "0.3p,gray",
        FONT_ANNOT     = "10p, Helvetica", 
        FONT_LABEL     = "10p, Helvetica")

    fig = pygmt.Figure()

    fig.basemap(region=[-30, 30, -90, 90], projection="H0/5c", frame="f0g10")

    # misfit values
    fig.grdimage(grid=grid, cmap=colormap, nan_transparent=True)

    fig.plot(x=[30, -30, 30, -30], y=[54.7356, 35.2644, -35.2644, -54.7356], pen="1.5p,gray")

    fig.grdcontour(grid=grid, levels=contours, pen="2p,white")
    if lune_marker is not None:
        fig.plot(x=lune_marker[0], y=lune_marker[1], style="+15p", pen="3.5p,yellow")

    fig.text(text="ISO",  x=0.,   y=90.,  offset="0/17p",     no_clip=True)
    fig.text(text="ISO",  x=0.,   y=-90., offset="0/-17p",    no_clip=True)
    fig.text(text="CLVD", x=-30., y=0.,   offset="-20p/-15p", no_clip=True)
    fig.text(text="CLVD", x=30.,  y=0.,   offset="20p/-15p",  no_clip=True)

    fig.text(text="(1,1,1)",    x=0.,   y=90.,      offset="0/7p",   no_clip=True)
    fig.text(text="(-1,-1,-1)", x=0.,   y=-90.,     offset="0/-7p",  no_clip=True)
    fig.text(text="(2,-1,-1)",  x=-30., y=0.,       offset="-20p/0", no_clip=True)
    fig.text(text="(1,1,-2)",   x=30.,  y=0.,       offset="20p/0",  no_clip=True)
    fig.text(text="(1,1,0)",    x=30.,  y=54.7356,  offset="20p/0",  no_clip=True)
    fig.text(text="(1,0,0)",    x=-30., y=35.2644,  offset="-20p/0", no_clip=True)
    fig.text(text="(0,0,-1)",   x=30.,  y=-35.2644, offset="20p/0",  no_clip=True)
    fig.text(text="(0,-1,-1)",  x=-30., y=-54.7356, offset="-20p/0", no_clip=True)

    with pygmt.config(FONT_ANNOT = "24p,Helvetica",
                      FONT_LABEL = "24p,Helvetica", 
                      MAP_ANNOT_OFFSET = "7p",  
                      MAP_LABEL_OFFSET = "15p",):
        fig.colorbar(frame="x+lmisfit", position="jBR+w4c/0.3c+o-1.5c/-0.6c+ebf+ma+v")

    fig.savefig(os.path.join(savepath,filename), dpi=300)