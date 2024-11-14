# Main Function for moment tensor inversion 
# Writtern by Lichun Yang
# 08/26/2024

import os
import gc
import argparse
from obspy.taup.taup_create import build_taup_model
from util.base import DataOT
from util.io import read_data, save_results, plot_results
from magnitude import searching_magnitude
from grid.grid_create import FullMomentTensorGridSegment
from grid.grid_search import GridSearchOptimalParallel

def read_args():
    '''
    Read arguments used in full moment tensor inversion
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_list", required=True, help="Inputfile: events list for inversion")
    parser.add_argument("--data_dir", required=True, help="Inputfile: observations for each event")
    parser.add_argument("--save_dir", default="inversion_results", help="Outputfile directory")
    parser.add_argument("--vmodel", required=True, help="1-D Velocity model")
    parser.add_argument("--weight", default="1/1/1", type=str, help="weight in misfit function")
    parser.add_argument("--ncpu", default=20, type=int, help="CPU numbers for parallel")
    parser.add_argument("--pcpu", default=50000, type=int, help="Grid number in one thread")
    parser.add_argument("--rmag", default=0.5, type=float, help="Searching magnitude range `[mag-dmag, mag+dmag]` if refer magnitude provided")
    parser.add_argument("--dmag", default=0.1, type=float, help="Searching magnitude interval")
    parser.add_argument("--min_mag", default=0, type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    parser.add_argument("--max_mag", default=10, type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    parser.add_argument("--figtype", default='png', type=str, help="Filetype for graphics")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    print("\n -----------  Inversion Beginning  -----------\n")
    # parameters used in inverison
    args = read_args()
    print(args)
    # inversion using P phase polar, amplitude and S/P amplitude ratio
    # read event datas for inversion and calculate ray parameters
    print("\n -----------    Reading Dataset    -----------\n")
    build_taup_model(filename=args.vmodel, output_folder='./')
    events = read_data(event_list=args.event_list, data_dir=args.data_dir, vmodel=args.vmodel)
    os.unlink("{}".format('./'+args.vmodel.split('/')[-1].split('.')[0]+'.npz'))
    print("\n Read {} events\n".format(len(events)))
    weight_list = [float(args.weight.split('/')[0]), float(args.weight.split('/')[1]), float(args.weight.split('/')[2])]
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    item = 1
    for event in events:
        print("\n ----  {}/{} Event ID {}  ----\n".format(item, len(events), event.evid))
        # create save path
        savepath = os.path.join(args.save_dir, event.evid)
        if not os.path.exists(savepath): os.makedirs(savepath)

        print("\n ----  Step 1: Searching Best Fitting magnitude ----\n")
        magnitude = searching_magnitude(event, weight_list, args.rmag, args.dmag, args.min_mag, args.max_mag, args.pcpu, args.ncpu)
        print("\n ----  Step 2: Searching Optimal Moment Tensor ----\n")
        grid = FullMomentTensorGridSegment(magnitude)
        gs, misfit = GridSearchOptimalParallel(event, grid, weight_list, args.pcpu, args.ncpu)

        invs = DataOT(gs, event, misfit, magnitude)
        print("\n ----  Step 3: Save and Plot inversion results ----\n")
        save_results(savepath, invs, event)
        # plot results
        plot_results(savepath, args.figtype, invs, event, weight_list)
        print("\n ----------- Event: ID {}; misfit {:>.3f} -----------\n".format(event.evid, invs.min_misfit[0]))
        item += 1
        gc.collect()
        del invs
        # Inversion finished
