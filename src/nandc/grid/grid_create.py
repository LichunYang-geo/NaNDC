import numpy as np
import xarray as xr

from nandc_mt import magnitude_to_rho, v_to_gamma, w_to_delta, gamma_to_v, delta_to_w

def DoubleCoupleGridSegment(magnitudes, npts_kappa, npts_sigma, npts_h):
    '''
    Creat search grid ARRAY in double-couple space
    :param magnitudes: Mw
    :type magnitudes: list or np.ndarray
    :param npts_kappa: number in strike regular
    :type npts_kappa: int
    :param npts_sigma: number in rake regular
    :type npts_sigma: int
    :param npts_h: number in dip regular
    :type npts_h: int
    :return grid: moment tensor
    :rtype: xr.DataArray   
    '''
    rho = list(map(magnitude_to_rho, np.array(magnitudes, dtype=np.float64, ndmin=1))) if magnitudes is not None else [1.]
    kappa = np.linspace(0, 360, npts_kappa)[:-1]
    sigma = np.linspace(-90, 90, npts_sigma)
    h = np.linspace(0, 1, npts_h)[1:]

    grid = xr.DataArray(dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'), 
                        coords={'rho':rho, 'v':[0.], 'w':[0.], 'kappa':kappa, 'sigma':sigma, 'h':h})
    return grid

def FullMomentTensorGridSegment(magnitudes, npts_v, npts_w, npts_kappa, npts_sigma, npts_h, tightness, uniformity):
    '''
    Creat search grid ARRAY in full-moment-tensor space
    :param magnitudes: Mw
    :type magnitudes: list or np.ndarray
    :param npts_v: number in v regular
    :type npts_v: int
    :param npts_w: number in w regular
    :type npts_w: int
    :param npts_kappa: number in strike regular
    :type npts_kappa: int
    :param npts_sigma: number in rake regular
    :type npts_sigma: int
    :param npts_h: number in dip regular
    :type npts_h: int
    :param tightness: Value in range `[0,1)` that controls how close the extremal points lie to the 
        boundary of the `v, w` rectangle
    :type tightness: float
    :param uniformity: Value in range `[0,1]` that controls the spacing between points
    :type uniformity: float
    :return grid: moment tensor
    :rtype: xr.DataArray   
    '''
    rho = list(map(magnitude_to_rho, np.array(magnitudes, dtype=np.float64, ndmin=1))) if magnitudes is not None else [1.]
    v, w = _semiregular_(npts_v, npts_w, tightness, uniformity)
    kappa = np.linspace(0, 360, npts_kappa)[:-1]
    sigma = np.linspace(-90, 90, npts_sigma)
    h = np.linspace(0, 1, npts_h)[1:]

    grid = xr.DataArray(dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'), 
                        coords={'rho':rho, 'v':v, 'w':w, 'kappa':kappa, 'sigma':sigma, 'h':h})
    return grid

def ConstrainedFullMomentTensorModelTable(tt_accepted_dc, npts_v, npts_w, tightness, uniformity):
    tt_accepted_dc = np.asarray(tt_accepted_dc, dtype=np.float64)

    if tt_accepted_dc.ndim != 2 or tt_accepted_dc.shape[1] < 6:
        raise ValueError("`tt_accepted_dc` must be a 2-D array with at least 6 columns")

    if tt_accepted_dc.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float64)

    v, w = _semiregular_(npts_v, npts_w, tightness, uniformity)
    vv, ww = np.meshgrid(v, w, indexing="ij")
    vw_pairs = np.column_stack([vv.ravel(), ww.ravel()])

    blocks = []
    for row in tt_accepted_dc:
        rho, _, _, kappa, sigma, h = row[:6]
        block = np.column_stack([
            np.full(vw_pairs.shape[0], rho, dtype=np.float64),
            vw_pairs[:, 0],
            vw_pairs[:, 1],
            np.full(vw_pairs.shape[0], kappa, dtype=np.float64),
            np.full(vw_pairs.shape[0], sigma, dtype=np.float64),
            np.full(vw_pairs.shape[0], h, dtype=np.float64),
        ])
        blocks.append(block)

    return np.vstack(blocks)

def _semiregular_(npts_v, npts_w, tightness, uniformity):
    '''
    #-- refer from mtuq https://uafgeotools.github.io/mtuq --#
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

    if not (0. <= tightness < 1.):
        raise ValueError("Allowable range: 0. <= tightness < 1.")

    if not (0. <= uniformity <= 1.):
        raise ValueError("Allowable range: 0. <= uniformity <= 1.")

    v1 = (tightness * _closed_interval_(-1./3., 1./3., npts_v) +
          (1.-tightness) * _open_interval_(-1./3., 1./3., npts_v))
    w1 = (tightness * _closed_interval_(-3./8.*np.pi, 3./8.*np.pi, npts_w) +
          (1.-tightness) * _open_interval_(-3./8.*np.pi, 3./8.*np.pi, npts_w))

    gamma2 = (tightness * _closed_interval_(-30., 30., npts_v) +
              (1.-tightness) * _open_interval_(-30, 30., npts_v))
    delta2 = (tightness * _closed_interval_(-90., 90., npts_w) +
              (1.-tightness) * _open_interval_(-90, 90., npts_w))

    delta = w_to_delta(w1) * (1.-uniformity) + delta2*uniformity
    gamma = v_to_gamma(v1) * (1.-uniformity) + gamma2*uniformity
    return gamma_to_v(gamma), delta_to_w(delta)

def _open_interval_(x1, x2, N):
    '''
    Covers the open interval (x1, x2) with N regularly-spaced points
    '''

    return np.linspace(x1, x2, 2*N+1)[1:-1:2]

def _closed_interval_(x1, x2, N):
    '''
    Covers the closed interval [x1, x2] with N regularly-spaced points
    '''
    
    return np.linspace(x1, x2, N)
