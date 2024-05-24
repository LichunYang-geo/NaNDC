import numpy as np
  
def get_full_MT_array(mt: np.ndarray) -> np.ndarray:
    '''
    convert 6 moment tensor to 6*6 full matrix
    :param mt: moment tensor [Mxx,Myy,Mzz,Mxy,Mxz,Myz] (6,) or (6,1) or (1,6)
    :type mt: np.ndarray
    :return full_MT: moment tensor matrix 6*6
    :rtype full_mt: np.ndarray
    '''
    if len(mt.shape) == 1:
        full_MT = np.array( ([[mt[0],mt[3],mt[4]],
                              [mt[3],mt[1],mt[5]],
                              [mt[4],mt[5],mt[2]]]) )
    else:
        mt = mt.flatten()
        full_MT = np.array( ([[mt[0],mt[3],mt[4]],
                              [mt[3],mt[1],mt[5]],
                              [mt[4],mt[5],mt[2]]]) )
    return full_MT

def mt_to_delta_gamma(mt: np.ndarray) -> float:
    '''
    Function to find delta and gamma given 6 moment tensor.
    :param mt: moment tensor [Mxx,Myy,Mzz,Mxy,Mxz,Myz] (6,) or (6,1) or (1,6)
    :type mt: np.ndarray
    :return delta: 
    '''
    full_MT = get_full_MT_array(mt)

    # Find the eigenvalues for the MT solution and sort into descending order:
    w,v = np.linalg.eigh(full_MT) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
    full_MT_eigvals_sorted = np.sort(w)[::-1] # Sort eigenvalues into descending order

    # Calculate gamma and delta (lat and lon) from the eigenvalues:
    lambda1 = full_MT_eigvals_sorted[0]
    lambda2 = full_MT_eigvals_sorted[1]
    lambda3 = full_MT_eigvals_sorted[2]
    gamma = np.arctan(((-1*lambda1) + (2*lambda2) - lambda3)/((3**0.5)*(lambda1 - lambda3))) # eq. 20a (Tape and Tape 2012)
    beta = np.arccos((lambda1+lambda2+lambda3)/((3**0.5)*((lambda1**2 + lambda2**2 + lambda3**2)**0.5))) # eq. 20b (Tape and Tape 2012)
    delta = (np.pi/2.) - beta # eq. 23 (Tape and Tape 2012)
    delta = np.rad2deg(delta)
    gamma = np.rad2deg(gamma)
    return delta, gamma 

def mt_to_gcmt(mt: np.ndarray) -> np.ndarray:
    '''
    Convert moment tensor format from NED to USE
    '''
    mt_shape = mt.shape
    if len(mt_shape) == 1:
        gcmt = np.empty((1,6))
        gcmt[0,0] = mt[2]
        gcmt[0,1] = mt[0]
        gcmt[0,2] = mt[1]
        gcmt[0,3] = mt[4]
        gcmt[0,4] = -mt[5]
        gcmt[0,5] = -mt[3]
    elif mt_shape[0] == 6:
        gcmt = np.empty((mt_shape[1],6))
        for i in range(0,mt_shape[1]):
            gcmt[i,0] = mt[2,i]
            gcmt[i,1] = mt[0,i]
            gcmt[i,2] = mt[1,i]
            gcmt[i,3] = mt[4,i]
            gcmt[i,4] = -mt[5,i]
            gcmt[i,5] = -mt[3,i]
    elif mt_shape[1] == 6:
        gcmt = np.empty((mt_shape[0],6))
        for i in range(0,mt_shape[0]):
            gcmt[i,0] = mt[i,2]
            gcmt[i,1] = mt[i,0]
            gcmt[i,2] = mt[i,1]
            gcmt[i,3] = mt[i,4]
            gcmt[i,4] = -mt[i,5]
            gcmt[i,5] = -mt[i,3]
    else:
        raise ValueError('Input moment tensor has unacceptable shape.')
    return gcmt

def mt_to_moment(mt: np.ndarray) -> float:
    """ 
    Calculates seismic moment (`M_0`)
    """
    full_mt = get_full_MT_array(mt)
    moment = (np.tensordot(full_mt, full_mt)/2.)**0.5
    return moment

def mt_to_magnitude(mt: np.ndarray) -> float:
    """ 
    Calculates moment magnitude (`M_w`)
    """
    magnitude = 2./3.*(np.log10(mt_to_moment(mt)) - 9.1)  
    return magnitude

def magnitude_to_moment(magnitude):
    """ 
    Converts from moment magnitude to scalar moment
    """
    return 10.**(1.5*float(magnitude) + 9.1)

def magnitude_to_rho(magnitude):
    """ 
    Converts from moment magnitude to Tape2012 magnitude parameter
    """
    return magnitude_to_moment(magnitude)*np.sqrt(2.)

def para_to_delta_gamma(v, w):
    """ 
    Converts from Tape2015 parameters to lune coordinates
    """
    return para_to_delta(w), para_to_gamma(v)

def para_to_delta(w):
    """ 
    Converts from Tape2015 parameter w to lune latitude
    """
    beta0 = np.linspace(0, np.pi, 300)
    u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)
    beta = np.interp(3.*np.pi/8. - w, u0, beta0)
    delta = np.rad2deg(np.pi/2. - beta)
    return delta

def para_to_gamma(v):
    """ 
    Converts from Tape2015 parameter v to lune longitude
    """   
    gamma = (1./3.)*np.arcsin(3.*v)
    gamma = np.rad2deg(gamma)
    return gamma

def para_to_v_w(delta, gamma):
    """ 
    Converts from lune coordinates to Tape2015 parameters
    """
    return para_to_v(gamma), para_to_w(delta)

def para_to_v(gamma):
    """ 
    Converts from lune longitude to Tape2015 parameter v
    """
    v = (1./3.)*np.sin(3.*np.deg2rad(gamma))
    return v

def para_to_w(delta):
    """ 
    Converts from lune latitude to Tape2015 parameter w
    """
    beta = np.deg2rad(90. - delta)
    u = (0.75*beta - 0.5*np.sin(2.*beta) + 0.0625*np.sin(4.*beta))
    w = 3.*np.pi/8. - u
    return w

def para_to_mij(rho: np.ndarray, v: np.ndarray, w: np.ndarray, kappa: np.ndarray, sigma: np.ndarray, h: np.ndarray, mt_type: str="NED") -> np.ndarray:
    """ 
    Converts from lune parameters to moment tensor parameters (north-east-down convention {Aki Richards})
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
    :return mt: moment tensors N*6, NED convention (default) [Mxx,Myy,Mzz,Mxy,Mxz,Myz] 
    :rtype: np.ndarray shape (N*6) or (1,6)
    """

    if mt_type not in ["NED", "USE"]:
        raise TypeError("moment tensor convention should be one of 'NED' or 'USE'")
    
    kR3 = np.sqrt(3.)
    k2R6 = 2.*np.sqrt(6.)
    k2R3 = 2.*np.sqrt(3.)
    k4R6 = 4.*np.sqrt(6.)
    k8R6 = 8.*np.sqrt(6.)

    m0 = rho/np.sqrt(2.)

    delta, gamma = para_to_delta_gamma(v, w)

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
        
def decom(mt, method='DC-CLVD'):

    full_mt = get_full_MT_array(mt)
    w, v = np.linalg.eig(full_mt) # w:eigenvalue of M v:eigenvector of M
    a1 = np.array([v[:,0]]).T
    a2 = np.array([v[:,1]]).T
    a3 = np.array([v[:,2]]).T
    vol = np.sum(w)/3  # isotropic
    max_value = w[np.argsort(np.abs(w))[2]]
    m_iso = vol * np.eye(3) 
    m_star = w - vol # eigenvalue of deviatoric
    ind = np.argsort(np.abs(m_star))
    max_value_star = m_star[ind[2]]
    min_value_star = m_star[ind[0]]
    max_a = np.array([v[:,ind[2]]]).T
    med_a = np.array([v[:,ind[1]]]).T
    min_a = np.array([v[:,ind[0]]]).T
    # ISO & DEV
    ISO = np.dot(np.dot(v,m_iso),v.T)  # M_ISO
    DEV = full_mt - ISO   # M_DEV
    if method == 'VD':
        # Vector Dipoles
        VD1 = m_star[0]*a1*a1.T
        VD2 = m_star[1]*a2*a2.T
        VD3 = m_star[2]*a3*a3.T
        return ISO, VD1, VD2, VD3
    elif method == 'DC':
        # Double Couples
        DC1 = (1/3)*(w[0]-w[1])*(a1*a1.T-a2*a2.T)
        DC2 = (1/3)*(w[1]-w[2])*(a2*a2.T-a3*a3.T)
        DC3 = (1/3)*(w[2]-w[0])*(a3*a3.T-a1*a1.T)
        return ISO, DC1, DC2, DC3
    elif method == 'CLVD':
        # CLVD
        CLVD1 = (1/3)*w[0]*(2*a1*a1.T-a2*a2.T-a3*a3.T)
        CLVD2 = (1/3)*w[1]*(2*a2*a2.T-a1*a1.T-a3*a3.T)
        CLVD3 = (1/3)*w[2]*(2*a3*a3.T-a1*a1.T-a2*a2.T)
        return ISO, CLVD1,CLVD2,CLVD3
    elif method == 'MAJ_MIN_DC':
        # Major and Minor Couple
        DC_MAJ = max_value_star*(max_a*max_a.T - med_a*med_a.T)
        DC_MIN = min_value_star*(min_a*min_a.T - med_a*med_a.T)
        return ISO, DC_MAJ, DC_MIN
    elif method == 'DC-CLVD':
        # Double Couple - CLVD Vavrycuk 2001
        F = -min_value_star/max_value_star
        DC = max_value_star*(1-2*F)*(max_a*max_a.T-med_a*med_a.T)
        CLVD = max_value_star*F*(2*max_a*max_a.T-med_a*med_a.T-min_a*min_a.T)
        eps = -min_value_star/np.abs(max_value_star)
        piso = (1/3)*100*(vol/np.abs(max_value))
        pclvd = 2*eps*(100-np.abs(piso))
        pdc = 100 - np.abs(piso) - np.abs(pclvd)
        p = np.array([pdc,piso,pclvd])
        mt_iso = np.array([ISO[0,0],ISO[1,1],ISO[2,2],ISO[0,1],ISO[0,2],ISO[1,2]])
        mt_dc = np.array([DC[0,0],DC[1,1],DC[2,2],DC[0,1],DC[0,2],DC[1,2]])
        mt_clvd = np.array([CLVD[0,0],CLVD[1,1],CLVD[2,2],CLVD[0,1],CLVD[0,2],CLVD[1,2]])
        return p, mt_iso, mt_dc, mt_clvd
    else:
        print('Input Method Error! (VD,DC,CLVD,MAJ_MIN_DC,DC-CLVD)')

def dsrin(str, dip, rake):

    str = np.deg2rad(str)
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    Ss = np.sin(str)
    Cs = np.cos(str)
    Sd = np.sin(dip)
    Cd = np.cos(dip)
    Sr = np.sin(rake)
    Cr = np.cos(rake)
    A = np.array([Cr*Cs+Sr*Cd*Ss,Cr*Ss-Sr*Cd*Cs,-Sr*Sd])
    N = np.array([-Ss*Sd,Cs*Sd,-Cd])
    T = np.sqrt(2) * (A+N)
    P = np.sqrt(2) * (A-N)
    B = np.cross(P,T)
    Ptrpl = v2trpl(P)
    Ttrpl = v2trpl(T)
    Btrpl = v2trpl(B)
    str,dip,rake = an2dsr_wan(N,A)

    return str, dip, rake, Ptrpl, Ttrpl, Btrpl


def v2trpl(xyz):

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

def an2dsr_wan(A,N):

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

def pt2ds(Ptrpl,Ttrpl):

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
    Btrpl = v2trpl(B)
    str1,dip1,rake1 = an2dsr_wan(A,N)
    str2,dip2,rake2 = an2dsr_wan(N,A)

    return str1,dip1,rake1,str2,dip2,rake2,Btrpl,PTangle

def mom2other(argc):
    '''
    Input: moment tensor [mxx,myy,mzz,mxy,mxz,myz]
    Output:
        F1: first fault para: strike, dip, rake
        F2: second fault para
        Ptrpl: P axis trend, plunge, also
        Ttrpl
        Btrpl
    '''
    m = get_full_MT_array(argc)

    [W,V] = np.linalg.eig(m)
    [W,V] = np.linalg.eig(m)
    ID = np.argsort(W)
    Ptrpl = v2trpl(V[:,ID[0]])
    Ttrpl = v2trpl(V[:,ID[2]])
    str1,dip1,rake1,str2,dip2,rake2,Btrpl,PTangle = pt2ds(Ptrpl,Ttrpl)
    F1 = np.array([str1,dip1,rake1])
    F2 = np.array([str2,dip2,rake2])

    return F1, F2, Ptrpl, Ttrpl, Btrpl