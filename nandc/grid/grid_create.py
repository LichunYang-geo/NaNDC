import numpy as np
import xarray as xr
from moment_tensor import magnitude_to_rho, para_to_delta, para_to_gamma, para_to_v, para_to_w

def FullMomentTensorGridSegment(magnitudes: list=None, 
        npts_v: int=21, npts_w: int=45, npts_kappa: int=73, npts_sigma: int=37, npts_h: int=21, 
        tightness: float=0.9, uniformity: float=0.9) -> xr.DataArray:
    
    '''
    Creat search grid ARRAY in full-moment-tensor space
    :param magnitudes: Mw
    :type magnitudes: list or np.ndarray
    :param npts_v: number in v regular
    :type npts_v: int
    :param npts_w: number in v regular
    :type npts_w: int
    :param npts_kappa: number in v regular
    :type npts_kappa: int
    :param npts_sigma: number in v regular
    :type npts_sigma: int
    :param npts_h: number in v regular
    :type npts_h: int
    :param tightness: Value in range `[0,1)` that controls how close the extremal points lie to the 
        boundary of the `v, w` rectangle
    :type tightness: float
    :param uniformity: Value in range `[0,1]` that controls the spacing between points
    :type uniformity: float
    :return grid: moment tensor
    :rtype: xr.DataArray   
    '''

    rho = list(map(magnitude_to_rho, asarray(magnitudes))) if magnitudes is not None else [1.]
    v, w = _semiregular(npts_v, npts_w, tightness, uniformity)
    kappa = np.linspace(0, 360, npts_kappa)[:-1]
    sigma = np.linspace(-90, 90, npts_sigma)
    h = np.linspace(0, 1, npts_h)[1:]

    grid = xr.DataArray(dims=('rho','v','w','kappa','sigma','h'), 
                        coords={'rho':rho, 'v':v, 'w':w, 'kappa':kappa, 'sigma':sigma, 'h':h})
    
    return grid

def _semiregular(npts_v: int, npts_w: int, tightness: float, uniformity: float):

    ## refer from mtuq https://uafgeotools.github.io/mtuq ##
    '''
    Returns coordinate vectors along the `v, w` axes
    - For `uniformity=0`, the spacing will be regular in `Tape2012 
      <https://uafgeotools.github.io/mtuq/references.html>`_ parameters 
      `delta`, `gamma`, which is  good for avoiding distortion near the upper
      and lower edges.
    - For `uniformity=1`, the spacing will be regular in `Tape2015
      <https://uafgeotools.github.io/mtuq/references.html>`_ parameters `v, w`, 
      which means that points are uniformly-spaced in terms of moment tensor 
      Frobenius norms.
    - For intermediate values, the spacing will be `semiregular` in the sense of
      a linear interpolation between the above cases.

    :param npts_v: number in v regular
    :type npts_v: int
    :param npts_w: number in v regular
    :type npts_w: int
    :param tightness: Value in range `[0,1)` that controls how close the extremal points lie to the boundary of the `v, w` rectangle
    :type tightness: float
    :param uniformity: Value in range `[0,1]` that controls the spacing between points
    :type uniformity: float
    '''

    assert 0. <= tightness < 1.,\
        Exception("Allowable range: 0. < tightness < 1.")

    assert 0. <= uniformity <= 1.,\
        Exception("Allowable range: 0. <= uniformity <= 1.")

    v1 = (tightness * closed_interval(-1./3., 1./3., npts_v) +
          (1.-tightness) * open_interval(-1./3., 1./3., npts_v))

    w1 = (tightness * closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w) +
          (1.-tightness) * open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w))

    gamma2 = (tightness * closed_interval(-30., 30., npts_v) +
              (1.-tightness) * open_interval(-30, 30., npts_v))

    delta2 = (tightness * closed_interval(-90., 90., npts_w) +
              (1.-tightness) * open_interval(-90, 90., npts_w))

    delta = para_to_delta(w1)*(1.-uniformity) + delta2*uniformity
    gamma = para_to_gamma(v1)*(1.-uniformity) + gamma2*uniformity

    return para_to_v(gamma), para_to_w(delta)

def open_interval(x1, x2, N):
    """ 
    Covers the open interval (x1, x2) with N regularly-spaced points
    """

    # NOTE: np.linspace(x1, x2, N)[1:-1] would be slightly simpler
    # but not as readily used by matplotlib.pyplot.pcolor

    return np.linspace(x1, x2, 2*N+1)[1:-1:2]
    # return np.linspace(x1, x2, N)[1:-1]

def closed_interval(x1, x2, N):
    """ 
    Covers the closed interval [x1, x2] with N regularly-spaced points
    """
    return np.linspace(x1, x2, N)

def asarray(x):

    return np.array(x, dtype=np.float64, ndmin=1, copy=False)