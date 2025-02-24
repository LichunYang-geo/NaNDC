import numpy as np
  
def get_full_MT_array(mt):
    '''
    Convert moment tensor to 6*6 full matrix
    :param mt: moment tensor [mxx, myy, mzz, mxy, mxz, myz] with size of (6,) or (6,1) or (1,6)
    :type mt: np.ndarray or list
    :return full_MT: moment tensor matrix array with size of 6*6
    :rtype full_mt: np.ndarray
    '''
    if type(mt) == list: mt = np.array(mt)

    if len(mt.shape) == 1:
        mt_array = np.array(([[mt[0],mt[3],mt[4]],
                              [mt[3],mt[1],mt[5]],
                              [mt[4],mt[5],mt[2]]]))
    else:
        mt = mt.flatten()
        mt_array = np.array(([[mt[0],mt[3],mt[4]],
                              [mt[3],mt[1],mt[5]],
                              [mt[4],mt[5],mt[2]]]))
        
    return mt_array

def mt_to_gcmt(mt):
    '''
    Convert moment tensor format from NED to USE
    '''
    if mt.ndim == 1:
        gcmt = np.empty((1,6))
        gcmt[0,0] = mt[2]
        gcmt[0,1] = mt[0]
        gcmt[0,2] = mt[1]
        gcmt[0,3] = mt[4]
        gcmt[0,4] = -mt[5]
        gcmt[0,5] = -mt[3]
    elif mt.shape[0] == 6:
        gcmt = np.empty((mt.shape[1], 6))
        for i in range(0, mt.shape[1]):
            gcmt[i,0] = mt[2,i]
            gcmt[i,1] = mt[0,i]
            gcmt[i,2] = mt[1,i]
            gcmt[i,3] = mt[4,i]
            gcmt[i,4] = -mt[5,i]
            gcmt[i,5] = -mt[3,i]
    elif mt.shape[1] == 6:
        gcmt = np.empty((mt.shape[0], 6))
        for i in range(0, mt.shape[0]):
            gcmt[i,0] = mt[i,2]
            gcmt[i,1] = mt[i,0]
            gcmt[i,2] = mt[i,1]
            gcmt[i,3] = mt[i,4]
            gcmt[i,4] = -mt[i,5]
            gcmt[i,5] = -mt[i,3]
    else:
        raise ValueError('Input moment tensor has unacceptable shape.')
    return gcmt

def mt_to_moment(mt):
    '''
    Calculates seismic moment (`M_0`)
    :param mt: moment tensor [mxx, myy, mzz, mxy, mxz, myz] with size of (6,) or (6,1) or (1,6)
    :type mt: np.ndarray or list
    :return moment: seismic moment M0
    :rtype moment: float
    '''
    mt_array = get_full_MT_array(mt)
    moment = (np.tensordot(mt_array, mt_array)/2.)**0.5

    return moment

def mt_to_magnitude(mt):
    '''
    Calculates moment magnitude (Mw)
    `Mw = 2/3 * (log10(M0)-9.1)` Kanamori, 1978
    :param mt: moment tensor [mxx, myy, mzz, mxy, mxz, myz] with size of (6,) or (6,1) or (1,6)
    :type mt: np.ndarray or list
    :return magnitude: moment magnitude Mw
    :rtype magnitude: float
    '''
    moment = mt_to_moment(mt)
    magnitude = 2./3.*(np.log10(moment) - 9.1)

    return magnitude

def mt_to_gamma_delta(mt):
    '''
    Function to find delta and gamma given moment tensor.
    :param mt: moment tensor [mxx, myy, mzz, mxy, mxz, myz] with size of (6,) or (6,1) or (1,6)
    :type mt: np.ndarray or list
    :return gamma: lune longitude
    :rtype gamma: float
    :return delta: lune latitude
    :rtype delta: float
    '''
    mt_array = get_full_MT_array(mt)

    # Find the eigenvalues for the MT solution and sort into descending order:
    w, v = np.linalg.eigh(mt_array) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
    mt_array_eigvals_sorted = np.sort(w)[::-1] # Sort eigenvalues into descending order

    # Calculate gamma and delta (lat and lon) from the eigenvalues:
    lambda1 = mt_array_eigvals_sorted[0]
    lambda2 = mt_array_eigvals_sorted[1]
    lambda3 = mt_array_eigvals_sorted[2]
    gamma = np.arctan(((-1*lambda1) + (2*lambda2) - lambda3)/((3**0.5)*(lambda1 - lambda3))) # eq. 20a (Tape and Tape 2012)
    beta = np.arccos((lambda1+lambda2+lambda3)/((3**0.5)*((lambda1**2 + lambda2**2 + lambda3**2)**0.5))) # eq. 20b (Tape and Tape 2012)
    delta = (np.pi/2.) - beta # eq. 23 (Tape and Tape 2012)
    delta = np.rad2deg(delta)
    gamma = np.rad2deg(gamma)

    return gamma, delta

def magnitude_to_moment(magnitude):
    '''
    Converts from moment magnitude to scalar moment
    :param magnitude: moment magnitude Mw
    :type manitude: float
    :return moment: seismic moment
    :rtype: float
    '''
    moment = 10.**(1.5*float(magnitude) + 9.1)

    return moment

def magnitude_to_rho(magnitude):
    '''
    Converts from moment magnitude to Tape2012 magnitude parameter
    '''
    moment = magnitude_to_moment(magnitude)
    rho = moment * np.sqrt(2.)
                                 
    return rho

def v_w_to_gamma_delta(v, w):
    '''
    Converts from Tape2015 parameters to lune coordinates
    '''
    return v_to_gamma(v), w_to_delta(w)

def v_to_gamma(v):
    '''
    Converts from Tape2015 parameters to lune coordinates
    '''
    gamma = (1./3.)*np.arcsin(3.*v)
    gamma = np.rad2deg(gamma)

    return gamma

def w_to_delta(w):
    '''
    Converts from Tape2015 parameters to lune coordinates
    '''
    beta0 = np.linspace(0, np.pi, 300)
    u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)
    beta = np.interp(3.*np.pi/8. - w, u0, beta0)
    delta = np.rad2deg(np.pi/2. - beta)

    return delta

def gamma_delta_to_v_w(gamma, delta):
    '''
    Converts from lune coordinates to Tape2015 parameters
    '''
    return gamma_to_v(gamma), delta_to_w(delta)

def gamma_to_v(gamma):
    '''
    Converts from lune coordinates to Tape2015 parameters
    '''
    v = (1./3.)*np.sin(3.*np.deg2rad(gamma))

    return v

def delta_to_w(delta):
    '''
    Converts from lune coordinates to Tape2015 parameters
    '''
    beta = np.deg2rad(90. - delta)
    u = (0.75*beta - 0.5*np.sin(2.*beta) + 0.0625*np.sin(4.*beta))
    w = 3.*np.pi/8. - u
    
    return w

def tt_to_mij(rho, v, w, kappa, sigma, h, mt_type="NED"):
    '''
    Converts from tt parameters (Carl Tape and Walter Tape, 2012) to moment tensor parameters (north-east-down convention {Aki Richards})
    :param rho: sqrt(2)*M0
    :type rho: np.ndarray N·m shape (N,) or (N,1)
    :param v: lune longitude in range `[-1/3,1/3]`
    :type v: np.bdarray shape (N,) or (N,1)
    :param w: lune latitude in range `[-3*pi/8,3*pi/8]`
    :type w: np.ndarray shape (N,) or (N,1)
    :param kappa: strike in range `[0,360]`
    :type kappa: np.ndarray shape (N,) or (N,1)
    :param sigma: rake in range `[-90,90]`
    :type sigma: np.ndarray shape (N,) or (N,1)
    :param h: cos(theta) dip in range `(0,1]`
    :type h: np.ndarray shape (N,) or (N,1)
    :param mt_type: moment tensor convention, should be one of 'NED', 'USE', default 'NED'
    :type mt_type: str
    :return mt: moment tensors N*6, NED convention (default) [mxx, myy, mzz, mxy, mxz, myz] 
    :rtype: np.ndarray shape (N*6) or (1,6)
    '''
    if mt_type not in ["NED", "USE"]:
        raise TypeError("moment tensor convention should be one of 'NED' or 'USE'")
    
    kR3 = np.sqrt(3.)
    k2R6 = 2.*np.sqrt(6.)
    k2R3 = 2.*np.sqrt(3.)
    k4R6 = 4.*np.sqrt(6.)
    k8R6 = 8.*np.sqrt(6.)

    m0 = rho/np.sqrt(2.)

    gamma, delta = v_w_to_gamma_delta(v, w)

    gamma = np.deg2rad(gamma)
    beta = np.deg2rad(90. - delta)
    kappa = np.deg2rad(kappa)
    sigma = np.deg2rad(sigma)
    theta = np.arccos(h)

    Cb  = np.cos(beta)
    Cg  = np.cos(gamma)
    Cs  = np.cos(sigma)
    Ct  = np.cos(theta)
    Ck  = np.cos(kappa)
    C2k = np.cos(2.0*kappa)
    C2s = np.cos(2.0*sigma)
    C2t = np.cos(2.0*theta)

    Sb  = np.sin(beta)
    Sg  = np.sin(gamma)
    Ss  = np.sin(sigma)
    St  = np.sin(theta)
    Sk  = np.sin(kappa)
    S2k = np.sin(2.0*kappa)
    S2s = np.sin(2.0*sigma)
    S2t = np.sin(2.0*theta)

    mt0 = m0* (1./24.) * \
        (k8R6*Cb + Sb*(-24.*Cg*(Cs*St*S2k + S2t*Sk*Sk*Ss) + kR3*Sg * \
        ((1. + 3.*C2k)*(1. - 3.*C2s) + 12.*C2t*Cs*Cs*Sk*Sk - 12.*Ct*S2k*S2s))) #Mxx

    mt1 = m0* (1./6.) * \
        (k2R6*Cb + Sb*(kR3*Ct*Ct*Ck*Ck*(1. + 3.*C2s)*Sg - k2R3*Ck*Ck*Sg*St*St +
        kR3*(1. - 3.*C2s)*Sg*Sk*Sk + 6.*Cg*Cs*St*S2k +
        3.*Ct*(-4.*Cg*Ck*Ck*St*Ss + kR3*Sg*S2k*S2s))) #Myy

    mt2 = m0 * (1./12.) * \
        (k4R6*Cb + Sb*(kR3*Sg*(-1. - 3.*C2t + 6.*C2s*St*St) + 12.*Cg*S2t*Ss))  #Mzz

    mt3 = m0* (1./8.)*Sb*(4.*Cg*(2.*C2k*Cs*St + S2t*S2k*Ss) +
        kR3*Sg*((1. - 2.*C2t*Cs*Cs - 3.*C2s)*S2k + 4.*Ct*C2k*S2s)) #Mxy

    mt4 = m0* (-1./2.)*Sb*(k2R3*Cs*Sg*St*(Ct*Cs*Sk - Ck*Ss) +
        2.*Cg*(Ct*Ck*Cs + C2t*Sk*Ss))  #Mxz

    mt5 = m0* (1./2.)*Sb*(Ck*(kR3*Cs*Cs*Sg*S2t + 2.*Cg*C2t*Ss) +
        Sk*(-2.*Cg*Ct*Cs + kR3*Sg*St*S2s))  #Myz
    
    if mt_type == "NED":
        if type(mt0) is np.ndarray:
            return np.column_stack([mt0, mt1, mt2, mt3, mt4, mt5])
        else:
            return np.array([[mt0, mt1, mt2, mt3, mt4, mt5]])
    elif mt_type == "USE":
        if type(mt0) is np.ndarray:
            return np.column_stack([mt2, mt0, mt1, mt4, -mt5, -mt3])
        else:
            return np.array([[mt2, mt0, mt1, mt4, -mt5, -mt3]])

def standard_decomposition(mt, vector=True):
    '''
    Decomposition of the Moment tensor to isotropic (ISO), 
    compensated linear vector dipole (CLVD), and double-couple components (DC), according to e.g. Jost and Herrmann 1989.
    Percentage Calculation for the ISO, CLVD and DC components, according to e.g. Vaveycuk 2001.
    :param mt:  moment tensor [mxx, myy, mzz, mxy, mxz, myz] with size of (6,) or (6,1) or (1,6)
    :type mt: np.ndarray or list
    :return 
    :mt_iso, mt_clvd, mt_dc: 
    :p_iso, p_clvd, p_dc:
    :eps:
    '''
    mt_array = get_full_MT_array(mt)
    w, v = np.linalg.eig(mt_array) # w:eigenvalue of M v:eigenvector of M
    max_value = w[np.argsort(np.abs(w))[2]] # `M_|max|` maxinum absolute value of eigencalue of moment tensor
    vol = np.sum(w)/3  # isotropic
    m_star = w - vol # eigenvalue of deviatoric
    sorted_index = np.argsort(np.abs(m_star)) # min, med, max index
    max_value_star, min_value_star = m_star[sorted_index[2]], m_star[sorted_index[0]]
    max_a, med_a, min_a = np.array([v[:,sorted_index[2]]]).T, np.array([v[:,sorted_index[1]]]).T, np.array([v[:,sorted_index[0]]]).T
    
    # moment tensor decomposition according to Jost and Herrmann 1989
    mt_iso  = vol*np.eye(3) # eq (37)
    mt_clvd = max_value_star*(-min_value_star/max_value_star)*(2*max_a*max_a.T-med_a*med_a.T-min_a*min_a.T) # eq (37)
    mt_dc   = max_value_star*(1-2*(-min_value_star/max_value_star))*(max_a*max_a.T-med_a*med_a.T) # eq (37)
    
    # percentage of each components according to Vaveycuk 2001
    eps = -min_value_star/np.abs(max_value_star) # eq (7) 
    p_iso  = 100*(vol/np.abs(max_value)) # percentage of ISO components eq (8a)
    p_clvd = 2*eps*(100-np.abs(p_iso))   # percentage of CLVD components eq (8b)
    p_dc   = 100 - np.abs(p_iso) - np.abs(p_clvd) # percentage of DC components eq (8c)

    if vector:
        mt_iso = np.array([mt_iso[0,0],mt_iso[1,1],mt_iso[2,2],mt_iso[0,1],mt_iso[0,2],mt_iso[1,2]])
        mt_clvd = np.array([mt_clvd[0,0],mt_clvd[1,1],mt_clvd[2,2],mt_clvd[0,1],mt_clvd[0,2],mt_clvd[1,2]])
        mt_dc = np.array([mt_dc[0,0],mt_dc[1,1],mt_dc[2,2],mt_dc[0,1],mt_dc[0,2],mt_dc[1,2]])

    return mt_iso, mt_clvd, mt_dc, p_iso, p_clvd, p_dc, eps

def mom2other(mt):
    '''
    Convert moment tensor to nodal planes (strike, dip, rake) and trend (plunge) angles of P-, T- and B-axis
    :param mt: moment tensor [mxx, myy, mzz, mxy, mxz, myz] with size of (6,) or (6,1) or (1,6)
    :type mt: np.ndarray or list
    :return np1 np2: nodal planes
    :return Ptrpl, Ttrpl, Btrpl: 
    '''
    mt_array = get_full_MT_array(mt)

    [w, v] = np.linalg.eig(mt_array)
    sorted_index = np.argsort(w)
    Ptrpl = _v2trpl_(v[:, sorted_index[0]])
    Ttrpl = _v2trpl_(v[:, sorted_index[2]])
    str1, dip1, rake1, str2, dip2, rake2, Btrpl, PTangle = _pt2ds_(Ptrpl,Ttrpl)
    np1 = np.array([str1, dip1, rake1])
    np2 = np.array([str2, dip2, rake2])
    return np1, np2, Ptrpl, Ttrpl, Btrpl

def _v2trpl_(xyz):

    trpl = np.empty(2)

    for j in range(0,3):
        if (abs(xyz[j])) <= 0.0001:
            xyz[j] = 0.
        if abs(abs(xyz[j])-1.) < 0.0001:
            xyz[j] = xyz[j]/abs(xyz[j])

    if abs(xyz[2]) == 1.:
        if xyz[2] < 0:
            trpl[0] = 180.
        else:
            trpl[0] = 0.
        trpl[1] = 90.
        return trpl
    
    if abs(xyz[0]) < 0.0001:
        if xyz[1] > 0.:
            trpl[0] = 90.
        elif xyz[1] < 0:
            trpl[0] = 270.
        else:
            trpl[0] = 0.
    else:
        trpl[0] = np.rad2deg((np.arctan2(xyz[1],xyz[0])))

    hypotxy = np.sqrt(xyz[0]**2+xyz[1]**2)
    trpl[1] = np.rad2deg((np.arctan2(xyz[2],hypotxy)))

    if trpl[1] < 0.:
        trpl[1] = -trpl[1]
        trpl[0] = trpl[0] - 180.

    if trpl[0] < 0.:
        trpl[0] = trpl[0] + 360.

    return trpl

def _an2dsr_wan_(A,N):

    if N[2] == -1.:
        str = np.arctan2(A[1],A[0])
        dip = 0.
    else:
        str = np.arctan2(-N[0],N[1])
        if N[2] == 0.:
            dip = 0.5*np.pi
        elif abs(np.sin(str)) >= 0.1:
            dip = np.arctan2(-N[0]/np.sin(str),-N[2])
        else:
            dip = np.arctan2(N[1]/np.cos(str),-N[2])

    a1 = A[0]*np.cos(str) + A[1]*np.sin(str)

    if abs(a1) < 0.0001:
        a1 = 0.

    if A[2] != 0.:

        if dip != 0.:
            rake = np.arctan2(-A[2]/np.sin(dip),a1)
        else:
            rake = np.arctan2(-1000000.*A[2],a1)

    else:
        if a1 >1.:
            a1 = 1.
        if a1 < -1.:
            a1 = -1.
        rake = np.arccos(a1)

    if dip < 0.:
        dip = dip + np.pi
        rake = np.pi - rake
        if rake > np.pi:
            rake = rake - 2*np.pi

    if dip > 0.5*np.pi:
        dip = np.pi-dip
        str = str + np.pi
        rake = -rake
        if str >= 2*np.pi:
            str = str - 2*np.pi

    if str <0.:
        str = str + 2*np.pi

    str = np.rad2deg(str)
    dip = np.rad2deg(dip)
    rake = np.rad2deg(rake)

    return str, dip, rake

def _pt2ds_(Ptrpl,Ttrpl):

    Ptrend = np.deg2rad(Ptrpl[0])
    Pplunge = np.deg2rad(Ptrpl[1])
    Ttrend = np.deg2rad(Ttrpl[0])
    Tplunge = np.deg2rad(Ttrpl[1])
    Spt = np.sin(Ptrend)
    Cpt = np.cos(Ptrend)
    Spp = np.sin(Pplunge)
    Cpp = np.cos(Pplunge)
    Stt = np.sin(Ttrend)
    Ctt = np.cos(Ttrend)
    Stp = np.sin(Tplunge)
    Ctp = np.cos(Tplunge)
    P = np.array([Cpt*Cpp, Spt*Cpp, Spp])
    T = np.array([Ctt*Ctp, Stt*Ctp, Stp])
    PTangle = np.rad2deg(np.arccos(np.dot(P,T)))

    if abs(PTangle-90.) > 10.:
        print('Two nodal plane are not perpendicular!')

    B = np.cross(T,P)
    A = np.sqrt(2) * (T+P)
    N = np.sqrt(2) * (T-P)
    Btrpl = _v2trpl_(B)
    str1,dip1,rake1 = _an2dsr_wan_(A,N)
    str2,dip2,rake2 = _an2dsr_wan_(N,A)

    return str1,dip1,rake1,str2,dip2,rake2,Btrpl,PTangle