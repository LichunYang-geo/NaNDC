import numpy as np
from pandas import DataFrame
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, degrees2kilometers, gps2dist_azimuth

def calc_ray_paras(evlo, evla, evdp, stinfo, vmodel):
    """
    Computes location and ray informations
    :param evlo: event longitude
    :type evlo: float
    :param evla: event latitude
    :type evla: float
    :param evdp: event depth
    :type evdp: float
    :param stinfo: stations' location
    :type stinfo: pd.DataFrame
    :param vmodel: velocity model .nd
    :type vmodel: str
    :return : ray information
    :rtype: pd.DataFrame
    """
    ray_para = np.zeros((len(stinfo),7))

    # velocity in source layer and density in first layer (near station)
    velocity = []
    with open(vmodel,'r') as g:
        for line in g.readlines():
            if len(line.split()) != 1:
                layer, alpha, beta, rho, _, _ = line.split()
                velocity.append([float(layer),float(alpha),float(beta),float(rho)])
    velocity = DataFrame(data=velocity,columns=['layer','alpha','beta','rho'])
    rho   = velocity.loc[0]['rho']
    alpha = velocity.iloc[(velocity['layer']-evdp).abs().argsort()[0]]['alpha']
    beta  = velocity.iloc[(velocity['layer']-evdp).abs().argsort()[0]]['beta']
    ray_para[:,4] = alpha * 1000 # m/s
    ray_para[:,5] = beta  * 1000 # m/s
    ray_para[:,6] = rho   * 1000 # kg/m^3
    # calculate ray parameters
    model = TauPyModel(model='./'+vmodel.split('/')[-1].split('.')[0]+'.npz')

    for i in range(0,len(stinfo)):
        stlo = stinfo.loc[i]['stlo']
        stla = stinfo.loc[i]['stla']
        azi = gps2dist_azimuth(evla, evlo, stla, stlo)[1]  # azi
        deg_distance = locations2degrees(stla, stlo, evla, evlo)
        # p phase arrival time, tko, aoi, and pdist
        p_path = model.get_ray_paths(source_depth_in_km = evdp, distance_in_degree = deg_distance, phase_list = ['p', 'P'])
        tko = p_path[0].takeoff_angle
        aoi = p_path[0].incident_angle
        path = p_path[0].path
        # interp ?? mark
        ddis = degrees2kilometers(np.diff(np.rad2deg(path['dist'])))
        ddep = np.diff(path['depth'])
        dist = np.sum(np.sqrt(ddis**2+ddep**2))  
        ray_para[i, 0] = azi
        ray_para[i, 1] = tko
        ray_para[i, 2] = aoi
        ray_para[i, 3] = dist * 1000 # meters
    return  DataFrame(data=ray_para, columns=['azi','tko','aoi','dist','alpha','beta','rho'])

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
    if type(mts) is not np.ndarray:
        mts = np.array(mts)

    stn = np.size(supplement, 0)

    azi = supplement[:,0]
    tko = supplement[:,1]
    aoi = supplement[:,2]
    dist =  supplement[:,3]
    alpha = supplement[:,4]
    beta =  supplement[:,5]
    rho =   supplement[:,6]

    theta = np.deg2rad(tko)
    fai =   np.deg2rad(azi)

    St =  np.sin(theta)
    S2t = np.sin(2*theta)
    C2t = np.cos(2*theta)
    Ct =  np.cos(theta)
    Sf =  np.sin(fai)
    S2f = np.sin(2*fai)
    C2f = np.cos(2*fai)
    Cf =  np.cos(fai)

    Rp =  np.dot(mts, np.array([St*St*Cf*Cf, St*St*Sf*Sf, Ct*Ct, St*St*S2f, S2t*Cf, S2t*Sf]))# (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    Rsv = np.dot(mts, np.array([0.5*S2t*Cf*Cf, 0.5*S2t*Sf*Sf, -0.5*S2t, 0.5*S2t*S2f, C2t*Cf, C2t*Sf]))
    Rsh = np.dot(mts, np.array([-0.5*St*S2f, 0.5*St*S2f, np.zeros(stn), St*C2f, -Ct*Sf, Ct*Cf]))
    
    pamp =   Rp/(4*np.pi*rho*(alpha**3)*dist) # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    svamp =  Rsv/(4*np.pi*rho*(beta**3)*dist) # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    shamp =  Rsh/(4*np.pi*rho*(beta**3)*dist) # (mtn*6) * (6*stn) mtn is the moment tensor number  return size mtn*stn
    spratio = np.log10(np.sqrt(svamp**2+shamp**2)/np.abs(pamp))
    pol = np.sign(pamp)
    return pol, pamp, spratio # mtn*stn