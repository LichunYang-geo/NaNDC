# ----------------------------------------- #
#     NaNDC V1.0.0                          #
#     Main Function for NaNDC               #
#     Author: Lichun Yang                   #
#     Email: lichun.yang.geo@gmail.com      #
#     Last update time: 03/21/2026          #
# ----------------------------------------- #
import sys
import argparse
import traceback

from pathlib import Path
from tempfile import TemporaryDirectory

from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

from util.nandc_io import read_data, read_nd_model
from nandc_inversion import nandc_inv_main
# ---------------------------------------------------------------------------------------------------------- #
def read_args():
    '''
    Read parameters used in inversion
    Users can modify default parameters in the code or use command line to set parameters
    '''
    parser = argparse.ArgumentParser()

    #-- IO parameters --#
    parser.add_argument("--event_list", required=True,     help="Inputfile: catalog (evid evlo evla evdp [mag])") # event list (catalog)
    parser.add_argument("--data_dir",   required=True,     help="Inputfile: observations") # data dir
    parser.add_argument("--save_dir",   default="results", help="Output directory")        # save dir

    #-- velocity model --#
    parser.add_argument("--vmodel", required=True, help="1-D Velocity model")
    parser.add_argument("--alpha",  default=0, type=float, help="Average P-wave velocity") # when zero, velocity will extract from velocity model
    parser.add_argument("--beta",   default=0, type=float, help="Average S-wave velocity")
    parser.add_argument("--rho",    default=0, type=float, help="Average density")

    #-- misfit contral parameters --#
    parser.add_argument("--weight_pol", default=1, type=float, help="p-polarity weight")
    parser.add_argument("--weight_amp", default=1, type=float, help="p-amplitude weight")
    parser.add_argument("--weight_spr", default=1, type=float, help="sp-ratio weight")

    #-- parameters in misfit function --#
    parser.add_argument("--norm_pol", default="L1", type=str, choices=["L1", "L2"], help="L1 or L2 norm for polarity misfit")
    parser.add_argument("--norm_amp", default="L1", type=str, choices=["L1", "L2"], help="L1 or L2 norm for P-wave amplitude misfit")
    parser.add_argument("--norm_spr", default="L1", type=str, choices=["L1", "L2"], help="L1 or L2 norm for S/P ratio misfit")
    parser.add_argument("--normalization", action="store_true", help="misfit normalization")
    
    #-- grid search parameters --#
    parser.add_argument("--inv_method", default="FMT", type=str, choices=["DC", "FMT", "DC2FMT"], help="Inversion methods")

    # parallel
    parser.add_argument("--n_jobs",     default=4,     type=int, help="Thread numbers for parallel")
    parser.add_argument("--chunk_size", default=30000, type=int, help="Grid number in one thread")
    
    # magnitude search
    parser.add_argument("--rmag",    default=0.5, type=float, help="Searching magnitude range `[mag-rmag, mag+rmag]` if refer magnitude provided")
    parser.add_argument("--dmag",    default=0.1, type=float, help="Searching magnitude interval")
    parser.add_argument("--min_mag", default=0,   type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    parser.add_argument("--max_mag", default=5,   type=float, help="Searching magnitude range `[min_mag, max_mag]` if refer magnitude not provided")
    # grid search
    parser.add_argument("--npts_v",     default=21,    type=int,   help="Number of v (lune longitude)")
    parser.add_argument("--npts_w",     default=45,    type=int,   help="Number of w (lune latitude)")
    parser.add_argument("--npts_kappa", default=73,    type=int,   help="Number of kappa (strike)")
    parser.add_argument("--npts_sigma", default=37,    type=int,   help="Number of sigma (rake)")
    parser.add_argument("--npts_h",     default=21,    type=int,   help="Number of h (dip)")
    parser.add_argument("--tightness",  default=0.8,   type=float, help="how close the extremal points lie to the boundary of the `v, w` rectangle")
    parser.add_argument("--uniformity", default=0.9,   type=float, help="the spacing between v,w points")
    #-----------------------------#

    #-- graphics paramters --#
    parser.add_argument("--figtype",  default='png', type=str, help="filetype for graphics")
    parser.add_argument("--skipzero", action="store_true", help="skip zero observation in plot")
    parser.add_argument("--plot_station_names", action="store_true", help="If True, station names will be included in the beachball plot")

    args = parser.parse_args()

    return args
# ---------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    print("=====================================================\n" \
          "     Nodal Array Non-Double-Couple (NaNDC) V1.0.0    \n" \
          "=====================================================\n")
    
    # load parameters
    args = read_args()
    
    # check weight parameters (cannot be all zero)
    if args.weight_pol == 0. and args.weight_amp == 0. and args.weight_spr == 0.:
        raise ValueError(" All weights are zero, please reset them! ")

    # read and construct velocity models for ray parameter calculation
    print(f" TauPy: loading velocity model from {args.vmodel} ")
    tmpdir = TemporaryDirectory()
    try:
        nd_path = Path(args.vmodel).resolve()
        build_taup_model(filename=str(nd_path), output_folder=tmpdir.name, verbose=False)
        npz_path = Path(tmpdir.name) / f"{nd_path.stem}.npz"

        vmodel_DataFrame = read_nd_model(nd_path)
        vmodel_TauPy = TauPyModel(model=str(npz_path))
    except Exception as e:
        print(f" Error: failed to load velocity model from {args.vmodel}. ")
        print(f" Error reason: {e} ")
        traceback.print_exc()
        tmpdir.cleanup()
        sys.exit(1)
    finally:
        tmpdir.cleanup()

    print(" Velocity model preview (first 5 rows):")
    print(vmodel_DataFrame.head())
    
    # read and construct event class
    print(" -----------    Reading Dataset    ----------- ")
    events = read_data(args.event_list, args.data_dir, vmodel_TauPy, vmodel_DataFrame, args.alpha, args.beta, args.rho) # read event datas for inversion and calculate ray parameters
    if len(events) == 0:
        raise RuntimeError(" No event loaded! ")
    print(f" Loaded {len(events)} events ! ")

    # display grid search space
    if args.inv_method == "DC":
        invspace = "Double Couple"
    elif args.inv_method == "FMT":
        invspace = "Full Moment Tensor"
    elif args.inv_method == "DC2FMT":
        invspace = "Double Couple to Full Moment Tensor"
    else:
        pass
    print(f" Inversion Mode: {invspace} ")

    # inversion beginning
    print(" -----------  Inversion Beginning  ----------- ")
    #------ inversion loop ------#
    failed_events = []
    for item, event in enumerate(events):
        print(f" --------  {item+1}/{len(events)} Event ID {event.evid}  -------- ")
        try:
            nandc_inv_main(args, event)
        except Exception as e:
            failed_events.append(event.evid)
            print(f"Error in Event {event.evid}: {e}")
            traceback.print_exc()   # 打印完整的错误堆栈
            continue

    print(" -----------  Inversion Finished  ----------- ")
    print(f" Successful events: {len(events) - len(failed_events)} / {len(events)} ")
    if failed_events:
        print(f" Failed events ({len(failed_events)}): {', '.join(failed_events)} ")
