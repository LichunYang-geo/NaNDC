import os
import gc
import argparse
import numpy as np
from util.base import DataIN, DataOT
from obspy.taup.taup_create import build_taup_model
from util.io import read_data, save_results, plot_results
from util.methods import get_best_fullspace
from moment_tensor import mt_to_magnitude
from grid.grid_create import FullMomentTensorGridSegment
from grid.grid_search import GridSearchOptimalParallel

def read_args():
    '''
    Read arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_list", required=True, help="Inputfile: events list for inversion")
    parser.add_argument("--data_dir", required=True, help="Inputfile: observations for each event")
    parser.add_argument("--save_dir", default="inversion_results", help="Outputfile directory")
    parser.add_argument("--vmodel", required=True, help="1-D Velocity model")
    parser.add_argument("--weight", default="1/1/1", type=str, help="weight in misfit function")
    parser.add_argument("--ncpu", default=12, type=int, help="CPU numbers for parallel")
    parser.add_argument("--interval", default=50000, type=int, help="Grid number for one thread")
    parser.add_argument("--rmag", default=0.5, type=float, help="Searching magnitude range `[mag-dmag, mag+dmag]` if refer magnitude provided")
    parser.add_argument("--dmag", default=0.1, type=float, help="Searching magnitude interval")
    parser.add_argument("--min_mag", default=0, type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    parser.add_argument("--max_mag", default=5, type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    parser.add_argument("--figtype", default='tif', type=str, help="Filetype for graphics")
    args = parser.parse_args()
    return args

def searching_magnitude(event: DataIN, weight_list: list, rmag:float, dmag:float, min_mag:float, max_mag:float, interval: int, ncpu: int) -> float:
    '''
    First-step Grid search for best-fitting magnitude (seismic moment M0)
    If reference magnitude (like local magnitude) are provided, best-fitting magnitude will be search in range `[mag-rmag, mag+rmag]` with inverval dmag.
    Otherwise, best-fitting magnitude will be search in range `[min_mag, max_mag]` with inverval dmag.
    :param event: 
    :type event: DataIN
    :param weight_list: 
    :type weight_list: list or np.ndarray
    :param weight_list: 
    :type weight_list: list or np.ndarray    
    '''
    if weight_list[1] == 0:
        print("\n ----  Step 1: Skipping magnitude searching (zero weight for P-wave amplitude) ----\n")
        magnitude = None
    else:
        if hasattr(event, 'evmag'):
            print("\n ----  Step 1: Searching Best-Fitting Magnitude between {} - {} ----\n".format(event.evmag-rmag, event.evmag+rmag))
            ## create grid searching space
            mag_list = np.arange(event.evmag-rmag, event.evmag+rmag, dmag)
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11)
            gs, _ = GridSearchOptimalParallel(rawdata=event, grid=grid, weight=[0, 1, 0], interval=interval, ncpu=ncpu)
            _, best_mt = get_best_fullspace(gs)
            magnitude = round(mt_to_magnitude(best_mt),2)
            print('\n Best fitting magnitude -- {} \n'.format(magnitude))
        else:
            # magnitude search step one #
            print("\n ----  Step 1-1: Searching Best-Fitting Magnitude between {} - {}  ----\n".format(min_mag, max_mag))
            mag_list = np.arange(min_mag, max_mag, 1)
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11)
            gs, _ = GridSearchOptimalParallel(rawdata=event, grid=grid, weight=[0,1,0], interval=interval, ncpu=ncpu)
            _, best_mt = get_best_fullspace(gs)
            magnitude = round(mt_to_magnitude(best_mt), 2)
            # magnitude search step two #
            print("\n ----  Step 1-2: Searching Best-Fitting Magnitude between {} - {}  ----\n".format(magnitude-1, magnitude+1))
            mag_list = np.arange(magnitude-1, magnitude+1, dmag)                                                                                                                                                                                  
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11)
            gs, _ = GridSearchOptimalParallel(rawdata=event, grid=grid, weight=[0,1,0], interval=interval, ncpu=ncpu)
            _, best_mt = get_best_fullspace(gs)
            magnitude = round(mt_to_magnitude(best_mt), 2)
            print('\n Best fitting magnitude -- {} \n'.format(magnitude))
    return magnitude

def Inversion(event:DataIN, weight_list:list, rmag:float, dmag:float, min_mag:float, max_mag:float, interval: int, ncpu: int) -> DataOT:
    '''
    Main Function For Source Mechanism Inversion 
    :param args: event_list, data_dir, save_dir, save_fname, vmodel, weight, save_nc
    '''
    magnitude = searching_magnitude(event, weight_list, rmag, dmag, min_mag, max_mag,  interval, ncpu)
    print("\n ----  Step 2: Searching Optimal Moment Tensor ----\n")
    grid = FullMomentTensorGridSegment(magnitude)
    gs, misfit = GridSearchOptimalParallel(event, grid, weight_list, interval, ncpu)
    invs = DataOT(gs, event, misfit, magnitude)
    return invs

if __name__ == "__main__":
    # parameters used in inverison
    args = read_args()
    print(args)
    # inversion using P phase polar, amplitude and S/P amplitude ratio
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("\n -----------  Inversion Beginning  -----------\n")
    print("\n -----------    Reading Dataset    -----------\n")
    # read event datas for inversion and calculate ray parameters
    build_taup_model(filename=args.vmodel, output_folder='./')
    events = read_data(event_list=args.event_list, data_dir=args.data_dir, vmodel=args.vmodel)
    os.unlink("{}".format('./'+args.vmodel.split('/')[-1].split('.')[0]+'.npz'))
    print("\n load {} events for inversion\n".format(len(events)))
    weight_list = [float(args.weight.split('/')[0]), float(args.weight.split('/')[1]), float(args.weight.split('/')[2])]
    item = 1
    for event in events:
        print("\n ----  {}/{} Processing Event ID {}  ----\n".format(item, len(events), event.evid))
        # extract parameters
        invs = Inversion(event, weight_list, args.rmag, args.dmag, args.min_mag, args.max_mag, args.interval, args.ncpu)
        print("\n ----  Step 3: Save and Plot  ----\n")
        # save results
        savepath = os.path.join(args.save_dir, event.evid)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_results(savepath, invs, event)
        # plot results
        plot_results(savepath, args.figtype, invs, event, weight_list)
        print("\n ----------- Event: ID {}; misfit {:>.3f} -----------\n".format(event.evid, invs.min_misfit[0]))
        del invs
        gc.collect()
        item += 1
        # Inversion finished