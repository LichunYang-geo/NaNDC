# NaNDC v0.0.1
# Main Function for NaNDC 
# Author: Lichun Yang
# 11/29/2024

import os
import gc
import time
import argparse
from obspy.taup.taup_create import build_taup_model
from util.base import DataOT
from util.io import read_data, save_results, plot_results
from magnitude import searching_magnitude
from grid.grid_create import FullMomentTensorGridSegment
from grid.grid_search import GridSearchOptimalParallel

def read_args():
    '''
    Read parameters used in inversion
    '''
    parser = argparse.ArgumentParser()
    # event list (catalog)
    parser.add_argument("--event_list", required=True, help="Inputfile: events list for inversion")
    # io path
    parser.add_argument("--data_dir", required=True, help="Inputfile: observations for each event")
    parser.add_argument("--save_dir", default="inversion_results", help="Outputfile directory")
    # velocity model
    parser.add_argument("--vmodel", required=True, help="1-D Velocity model")
    # weight
    parser.add_argument("--weight", default="1/1/1", type=str, help="weight in misfit function")
    #-- grid search parameters --#
    # parallel
    parser.add_argument("--ncpu", default=24, type=int, help="CPU numbers for parallel")
    parser.add_argument("--pcpu", default=50000, type=int, help="Grid number in one thread")
    # magnitude search
    parser.add_argument("--rmag",    default=0.5, type=float, help="Searching magnitude range `[mag-rmag, mag+rmag]` if refer magnitude provided")
    parser.add_argument("--dmag",    default=0.1, type=float, help="Searching magnitude interval")
    parser.add_argument("--min_mag", default=-2,  type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    parser.add_argument("--max_mag", default=6,   type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    # full moment tensor
    parser.add_argument("--npts_v",     default=21,  type=int,   help="Number of v")
    parser.add_argument("--npts_w",     default=45,  type=int,   help="Number of w")
    parser.add_argument("--npts_kappa", default=73,  type=int,   help="Number of kappa")
    parser.add_argument("--npts_sigma", default=37,  type=int,   help="Number of sigma")
    parser.add_argument("--npts_h",     default=21,  type=int,   help="Number of h")
    parser.add_argument("--tightness",  default=0.9, type=float, help="how close the extremal points lie to the boundary of the `v, w` rectangle")
    parser.add_argument("--uniformity", default=0.9, type=float, help="the spacing between points")
    # graphics
    parser.add_argument("--figtype",  default='png', type=str, help="Filetype for graphics")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    print("\n -----------  Inversion Beginning  -----------")
    # load parameters used in inverison
    args = read_args()
    weight_list = [float(args.weight.split('/')[0]), float(args.weight.split('/')[1]), float(args.weight.split('/')[2])]
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    # inversion using P phase polar, amplitude and S/P amplitude ratio
    # read event datas for inversion and calculate ray parameters
    print("\n -----------    Reading Dataset    -----------")
    build_taup_model(filename=args.vmodel, output_folder='./', verbose=False)
    events = read_data(event_list=args.event_list, data_dir=args.data_dir, vmodel=args.vmodel)
    os.unlink("{}".format('./'+args.vmodel.split('/')[-1].split('.')[0]+'.npz'))
    print("\n Read {} events\n".format(len(events)))
    
    #------ inversion loop ------#
    item = 1
    for event in events:
        start_time = time.time()
        print("\n --------  {}/{} Event ID {}  --------\n".format(item, len(events), event.evid))
        # create savepath
        savepath = os.path.join(args.save_dir, event.evid)
        if not os.path.exists(savepath): 
            os.makedirs(savepath)
        flog = open(os.path.join(savepath, event.evid+".log"), 'w')
        flog.write(f"weight_polarity: {weight_list[0]}\nweight_p-amplitude: {weight_list[1]}\nweight_sp-ratio: {weight_list[2]}\n")
        flog.write(f"Number of Station: {len(event.stations)}\nNumber of polarity: {len(event.pol)}\nNumber of P-amplitude: {len(event.pamp)}\nNumber of sp-ratio: {len(event.spratio)}\n")
        # step 1
        print("\n ----  Step 1: Searching Best Fitting magnitude ----\n")
        magnitude = searching_magnitude(event, weight_list, args.rmag, args.dmag, args.min_mag, args.max_mag, args.pcpu, args.ncpu)
        # step 2
        print("\n ----  Step 2: Searching Optimal Moment Tensor ----\n")
        grid = FullMomentTensorGridSegment(magnitude, args.npts_v, args.npts_w, args.npts_kappa, args.npts_sigma, args.npts_h, args.tightness, args.uniformity)
        gs, misfit = GridSearchOptimalParallel(event, grid, weight_list, args.pcpu, args.ncpu)
        inv_result = DataOT(gs, event, misfit, magnitude)
        # step 3
        print("\n ----  Step 3: Save and Plot Inversion Results ----\n")
        save_results(savepath, inv_result, event)
        # plot results
        plot_results(savepath, args.figtype, inv_result, event, weight_list)
        end_time = time.time()
        running_time = round(end_time-start_time, 1)
        flog.write(f"RuningTime: {running_time}s\n")
        print("\n ----------- Event: ID {}; Misfit: {:>.3e}; RuningTime: {:>.1f}s ----------- \n".format(event.evid, inv_result.min_misfit[0], running_time))
        flog.close()
        item += 1
        gc.collect()
        del grid, misfit, inv_result # memory collection
    #------ inversion loop ------#