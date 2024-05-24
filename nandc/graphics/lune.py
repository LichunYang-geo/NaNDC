import os
import shutil
import warnings
import subprocess
import numpy as np
from os.path import basename, exists, splitext
from mtfunc import mt_to_delta_gamma, mt_to_gcmt

def plot_misfit_lune(savepath: str, filename: str, 
                     lune_value: np.ndarray, lune_limits: np.ndarray = None,
                     lune_marker: np.ndarray= None, lune_mts: np.ndarray = None,
                     **kwargs):
    '''
    plot misfit value and moment tensor on eigenvalue lune
    :param savepath: figure savepath
    :type savepath: str
    :param filename: figure savename
    :type filename: str
    :param lune_value: misfit values plot on lune, N*3 [lon,lat,values] 2-D array
    :type lune_value: np.ndarray
    :param lune_limits: misfit area for visualization
    :type lune_limits: np.ndarray [lower, upper]
    :param lune_marker: marker's longitude and latitude on eigenvalue lune
    :type lune_marker: np.ndarray [gamma, delta]
    :param lune_mt: moment tensor plot on lune, N*6 [Mxx,Myy,Mzz,Mxy,Mxz,Myz]
    :type lune_mt: np.ndarray
    **kwargs:
        colormap='no_green' 
        flip_cpt=False
        colorbar_limits=None 
    '''

    if not exists(savepath):
        os.makedirs(savepath)
    
    if lune_limits is not None:
        ind = np.where((lune_value[:,2]<=lune_limits[1]) & (lune_value[:,2]>=lune_limits[0]))
        lune_value = lune_value[ind]

    lune_value = np.unique(lune_value, axis=0)

    _call(shell_script=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plot_misfit_lune'),
            filename=os.path.join(savepath,filename),
            lune_value=lune_value,
            lune_marker=lune_marker,
            lune_mts=_parse_lune_array(lune_mts), 
            **kwargs)
    
def _call(shell_script, filename, lune_value, lune_marker=None, lune_mts=None,
    colormap='no_green', flip_cpt=False, colorbar_limits=None):

    print('  calling GMT script: %s' % basename(shell_script))

    # parse filename and title
    parts = splitext(filename)
    filename, filetype = parts[0], parts[1].lstrip('.')
    if filetype in gmt_formats:
        filetype = filetype
    else:
        filetype = 'png'
        warnings.warn('Unrecognized extension: defaulting to png')

    # parse color palette
    cpt_name = colormap

    # parse colorbar limits
    try:
        minval, maxval = colorbar_limits
    except:
        minval, maxval = np.min(lune_value[:,-1]), np.max(lune_value[:,-1])
    cpt_step=((maxval-minval)/100)

    # write values to be plotted as ASCII table
    ascii_file_1 = _safename('tmp_'+filename+'_ascii1.txt')
    _savetxt(ascii_file_1, lune_value)

    # write supplementatal ASCII table, if given (beachballs)
    ascii_file_2 = _safename('tmp_'+filename+'_ascii2.txt')
    if lune_mts is not None:
        _savetxt(ascii_file_2, lune_mts)

    # write marker coordinates, if given
    marker_coords_file = _safename('tmp_'+filename+'_marker_coords.txt')
    if lune_marker is not None:
        _savetxt(marker_coords_file, *lune_marker)

    # contour line value
    contours = '{:.3f},{:.3f},{:.3f}'.format(minval+2*cpt_step, minval+5*cpt_step, minval+10*cpt_step)
    # call bash script
    if exists_gmt():
        subprocess.call("%s %s %s %s %s %f %f %s %s %d %s %s" %
           (shell_script,
            filename,
            filetype,
            ascii_file_1,
            ascii_file_2,
            minval,
            maxval,
            # workaround GMT scientific notation parsing
            _float_to_str(cpt_step),
            cpt_name,
            int(bool(flip_cpt)),
            marker_coords_file,
            contours,
            ),
            shell=True)
    else:
        gmt_not_found_warning(
            'values_ascii')
        
def _parse_lune_array(lune_array: np.ndarray) -> np.ndarray:
    '''
    convert moment tensor to gmt format
    :param lune_array: moment tensors [mxx, myy, mzz, mxy, mxz, myz]
    :type lune_array: np.ndarray [X Y depth mrr mtt mff mrt mrf mtf exp newX newY]
    :return gmt_array: gmt format for visualization 
    :rtype: np.ndarray
    '''
    if lune_array is None:
        return None

    N = lune_array.shape[0]
    gmt_array = np.empty((N, 12))

    for _i in range(N):

        mt = np.array([lune_array[_i,0],lune_array[_i,1],lune_array[_i,2],lune_array[_i,3],lune_array[_i,4],lune_array[_i,5]])

        list_str =  [('%.2e' % mt[i]) for i in range(np.size(mt))]
        list_int = [int(string.split('e')[1]) for string in list_str]

        exponent = np.max(list_int)
        scaled_mt = mt/10**(exponent)
        dummy_value = 0
        gmt_array[_i, 1], gmt_array[_i, 0] = mt_to_delta_gamma(mt)
        gmt_array[_i, 2] = dummy_value
        gmt_array[_i, 3:9] = mt_to_gcmt(scaled_mt)
        gmt_array[_i, 9] = 23
        gmt_array[_i, 10:] = 0

    return gmt_array

def _float_to_str(val):
    # workaround GMT scientific notation parsing (problem with GMT >=6.1)
    if str(val).endswith('e+00'):
        return str(val).replace('e+00', '')
    else:
        return str(val).replace('e+', 'e')

def _savetxt(filename, *args, fmt='%.6e'):
    np.savetxt(filename, np.column_stack(args), fmt=fmt)

def _safename(filename):
    # used for writing temporary files only
    return filename.replace('/', '__')

gmt_formats = ['pdf', 'ps', 'eps', 'bmp', 'jpg', 'png', 'PNG', 'ppm', 'tif']

def exists_gmt():
    return bool(shutil.which('gmt'))

def gmt_not_found_warning(filename):
    warnings.warn("""
        WARNING

        Generic Mapping Tools executables not found on system path.
        PostScript output has not been written. 

        Misfit values have been saved to:
            %s
        """ % filename)