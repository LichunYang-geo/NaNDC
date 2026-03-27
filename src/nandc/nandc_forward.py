import numpy as np
from pandas import DataFrame

from obspy.geodetics import locations2degrees, gps2dist_azimuth

def calc_ray_paras(evlo, evla, evdp, stinfo, vmodel_TauPy, vmodel_DataFrame, alpha_user, beta_user, rho_user):

    ray_para = np.empty((len(stinfo), 7), dtype=np.float64)
    for i in range(0,len(stinfo)):
        stlo = stinfo.loc[i]['stlo']
        stla = stinfo.loc[i]['stla']
        stal = stinfo.loc[i]['stal'] # km

        dist_deg = locations2degrees(evla, evlo, stla, stlo)
        azi = gps2dist_azimuth(evla, evlo, stla, stlo)[1]
        ray_para[i, 0] = azi

        try:
            arrivals = vmodel_TauPy.get_ray_paths(source_depth_in_km=evdp, distance_in_degree=dist_deg, receiver_depth_in_km=0, phase_list=["p", "P"])

            tko  = arrivals[0].takeoff_angle
            aoi  = arrivals[0].incident_angle
            path = arrivals[0].path
            r = 6371.0 - path["depth"]
            theta = path["dist"]

            ds = np.sqrt(r[:-1]**2 + r[1:]**2 - 2 * r[:-1] * r[1:] * np.cos(theta[1:] - theta[:-1]))
            ray_length_km = np.sum(ds)

            ray_para[i, 1] = tko # degree [0-180]
            ray_para[i, 2] = aoi # degree not used in nandc
            ray_para[i, 3] = ray_length_km * 1000.0 # meters
        except (IndexError, AttributeError, KeyError, ValueError) as e:
            print(
                f" Warning: cannot calculate ray parameters for station "
                f"{stinfo.loc[i]['stid']}: {e}. Set to zero. "
            )
            ray_para[i, 1:4] = -12345

    depths = vmodel_DataFrame['depth'].values
    vps    = vmodel_DataFrame['vp'].values
    vss    = vmodel_DataFrame['vs'].values

    # 判断是否正好命中某个深度
    idx_exact = np.where(np.isclose(depths, evdp))[0]

    if len(idx_exact) > 0:
        i = idx_exact[0]
        vp_model  = vps[i]
        vs_model  = vss[i]
    else:
        if evdp < depths.min() or evdp > depths.max():
            raise ValueError(f'evdp={evdp} 超出速度模型深度范围 [{depths.min()}, {depths.max()}] km')

        i2 = np.searchsorted(depths, evdp)
        i1 = i2 - 1

        z1, z2 = depths[i1], depths[i2]
        w = (evdp - z1) / (z2 - z1)

        vp_model  = vps[i1]  + w * (vps[i2]  - vps[i1])
        vs_model  = vss[i1]  + w * (vss[i2]  - vss[i1])

    if alpha_user != 0:
        ray_para[:,4] = alpha_user * 1000 # m/s
    else:
        ray_para[:,4] = vp_model * 1000

    if beta_user != 0:
        ray_para[:,5] = beta_user * 1000
    else:
        ray_para[:,5] = vs_model * 1000
        
    if rho_user != 0:
        ray_para[:,6] = rho_user * 1000
    else:
        ray_para[:,6] = vmodel_DataFrame.loc[0, 'rho'] * 1000 # kg/m^3

    return  DataFrame(data=ray_para, columns=['azi', 'tko', 'aoi', 'dist', 'alpha', 'beta', 'rho'])

def calc_synthetic(mts, supplement):
    '''
    Calculate First Arrival P-wave Amplitude and S/P ratio
    :param mts: moment tensor (source parameter) [Mxx Myy Mzz Mxy Mxz Myz]
    :type mts: np.ndarray shape (mtn, 6)
    :param supplement: azimuth, takeoff angle, ray distance, average p-wave velocity, average s-wave velocity, average density
    :type supplement: np.ndarray (stn*7) or (stn*7)
    :return pamp: first arrival p-wave amplitude
    :rtype: np.ndarray (mtn*stn) or (1*stn)
    :return spraio: first arrival S/P amplitude ratio
    :rtype: np.ndarray (mtn*stn) or (1*stn)
    '''

    # f0 = 15
    # Qp = 300
    # Qs = 150

    if type(mts) is not np.ndarray:
        mts = np.array(mts)

    stn = np.size(supplement, 0)

    azi = supplement[:,0]
    tko = supplement[:,1]
    aoi = supplement[:,2]
    dist  = supplement[:,3]
    alpha = supplement[:,4]
    beta  = supplement[:,5]
    rho   = supplement[:,6]

    theta = np.deg2rad(tko)
    fai   = np.deg2rad(azi)

    St  = np.sin(theta)
    S2t = np.sin(2*theta)
    C2t = np.cos(2*theta)
    Ct  = np.cos(theta)
    Sf  = np.sin(fai)
    S2f = np.sin(2*fai)
    C2f = np.cos(2*fai)
    Cf  = np.cos(fai)

    Rp  = np.dot(mts, np.array([St*St*Cf*Cf, St*St*Sf*Sf, Ct*Ct, St*St*S2f, S2t*Cf, S2t*Sf])) # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    Rsv = np.dot(mts, np.array([0.5*S2t*Cf*Cf, 0.5*S2t*Sf*Sf, -0.5*S2t, 0.5*S2t*S2f, C2t*Cf, C2t*Sf]))
    Rsh = np.dot(mts, np.array([-0.5*St*S2f, 0.5*St*S2f, np.zeros(stn), St*C2f, -Ct*Sf, Ct*Cf]))
    
    # atten_p = np.exp(-np.pi * f0 * dist / (Qp * alpha))
    # atten_s = np.exp(-np.pi * f0 * dist / (Qs * beta))

    Pamp    = Rp/(4*np.pi*rho*(alpha**3)*dist)  # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    SVamp   = Rsv/(4*np.pi*rho*(beta**3)*dist)  # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    SHamp   = Rsh/(4*np.pi*rho*(beta**3)*dist)  # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    SPratio = np.log10(np.sqrt(SVamp**2+SHamp**2)/np.abs(Pamp))

    return Pamp, SPratio # mtn*stn