import time
import traceback
import numpy as np
from pathlib import Path

from grid.grid_create import (
    FullMomentTensorGridSegment, 
    DoubleCoupleGridSegment, 
    ConstrainedFullMomentTensorModelTable,
)
from grid.grid_search import GridSearchOptimalParallel, GridSearchOptimalParallelFromTable

from nandc_mt import mt_to_magnitude, v_to_gamma, w_to_delta, mt_to_gamma_delta

from util.nandc_method import (
    get_best_fullspace, 
    get_best_lunespace, 
    get_accepted_fullspace, 
    get_best_from_result_table, 
    get_accepted_from_result_table, 
    get_best_lunespace_from_result_table,
    mt2omega,
)
from util.nandc_io import write_results, write_file_lune
from util.nandc_log import (
    build_config_log_section, 
    build_coverage_log_section,
    build_data_log_section, 
    build_event_log_section, 
    build_result_log_section, 
    build_runtime_log_section,
    build_quality_log_section,
)
from util.nandc_quality import evaluate_quality_from_grid, evaluate_quality_from_table

from graphics.beachball import plot_results_beachball
from graphics.lune import plot_misfit_lune_pygmt

def apply_effective_gap_stats(event, weight_list):
    gap_defs = (
        ("pol", weight_list[0], event.pol_max_azi_gap, event.pol_max_takeoff_gap),
        ("amp", weight_list[1], event.amp_max_azi_gap, event.amp_max_takeoff_gap),
        ("spr", weight_list[2], event.spr_max_azi_gap, event.spr_max_takeoff_gap),
    )

    effective_gap_stats = {}
    for name, weight, raw_azi_gap, raw_takeoff_gap in gap_defs:
        if weight == 0:
            eff_azi_gap = np.nan
            eff_takeoff_gap = np.nan
        else:
            eff_azi_gap = raw_azi_gap
            eff_takeoff_gap = raw_takeoff_gap

        setattr(event, f"{name}_effective_max_azi_gap", eff_azi_gap)
        setattr(event, f"{name}_effective_max_takeoff_gap", eff_takeoff_gap)
        effective_gap_stats[name] = {
            "max_azi_gap": eff_azi_gap,
            "max_takeoff_gap": eff_takeoff_gap,
        }

    event.effective_gap_stats = effective_gap_stats

def searching_magnitude(event, args):
    '''
    First-step Grid search for best-fitting magnitude (seismic moment M0)
    If reference magnitude (like local magnitude) are provided, best-fitting magnitude will be search in range `[mag-rmag, mag+rmag]` with inverval dmag.
    Otherwise, best-fitting magnitude will be search in range `[min_mag, max_mag]` with inverval 1 first and
    then search in range `[mag-1, mag+1]` with inverval dmag.
    '''

    norm_list = [args.norm_pol, args.norm_amp, args.norm_spr]
    if hasattr(event, 'evmag'):
        print(f" ----  Searching Best-Fitting Magnitude between {event.evmag-args.rmag:.1f} - {event.evmag+args.rmag:.1f} ---- ")

        ## create grid searching space
        mag_list = np.arange(event.evmag-args.rmag, event.evmag+args.rmag, args.dmag)
        if args.inv_method == "DC" or args.inv_method == "DC2FMT":
            grid = DoubleCoupleGridSegment(magnitudes=mag_list, npts_kappa=37, npts_sigma=19, npts_h=11)
        elif args.inv_method == "FMT":
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11, tightness=0.8, uniformity=0.8)
        else:
            raise ValueError(f" Unknown inverse method ")
        gs, _      = GridSearchOptimalParallel(event, grid, [0, 1, 0], norm_list, args.normalization, args.n_jobs, args.chunk_size)
        _, mt_best = get_best_fullspace(gs)
        
        ## --------------------------------------------- ##
        # mean magnitude
        magnitude_array = np.zeros((len(mt_best),))
        for i in range(len(mt_best)):
            magnitude_array[i] = round(mt_to_magnitude(mt_best[i,:]), 2)
        magnitude = round(np.mean(magnitude_array), 1) if not np.isnan(np.mean(magnitude_array)) else None
        print(f" Best fitting magnitude -- {magnitude} ")
        ## --------------------------------------------- ##
        
        return magnitude
    
    else:
        # magnitude search step one #
        print(f" ----  1-1 Searching Best-Fitting Magnitude between {args.min_mag:.1f} - {args.max_mag:.1f}  ---- ")
    
        mag_list = np.arange(args.min_mag, args.max_mag, 0.5)
        if args.inv_method == "DC" or args.inv_method == "DC2FMT":
            grid = DoubleCoupleGridSegment(magnitudes=mag_list, npts_kappa=37, npts_sigma=19, npts_h=11)
        elif args.inv_method == "FMT":
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11, tightness=0.8, uniformity=0.8)
        else:
            raise ValueError(f" Unknown inverse method ")
        gs, _      = GridSearchOptimalParallel(event, grid, [0, 1, 0], norm_list, args.normalization, args.n_jobs, args.chunk_size)
        _, mt_best = get_best_fullspace(gs)
        
        ## --------------------------------------------- ##
        # mean magnitude
        magnitude_array = np.zeros((len(mt_best),))
        for i in range(len(mt_best)):
            magnitude_array[i] = round(mt_to_magnitude(mt_best[i,:]), 2)
        magnitude = round(np.mean(magnitude_array), 1) if not np.isnan(np.mean(magnitude_array)) else None
        ## --------------------------------------------- ##
        
        # magnitude search step two #
        print(f" ----  1-2 Searching Best-Fitting Magnitude between {magnitude-args.rmag:.1f} - {magnitude+args.rmag:.1f}  ---- ")
        
        mag_list = np.arange(magnitude-args.rmag, magnitude+args.rmag, args.dmag)                                                                                                                                                                                  
        if args.inv_method == "DC" or args.inv_method == "DC2FMT":
            grid = DoubleCoupleGridSegment(magnitudes=mag_list, npts_kappa=37, npts_sigma=19, npts_h=11)
        elif args.inv_method == "FMT":
            grid = FullMomentTensorGridSegment(magnitudes=mag_list, npts_v=7, npts_w=19, npts_kappa=37, npts_sigma=19, npts_h=11, tightness=0.8, uniformity=0.8)
        else:
            raise ValueError(f" Unknown inverse method ")
        gs, _      = GridSearchOptimalParallel(event, grid, [0, 1, 0], norm_list, args.normalization, args.n_jobs, args.chunk_size)
        _, mt_best = get_best_fullspace(gs)

        ## --------------------------------------------- ##
        # mean magnitude
        magnitude_array = np.zeros((len(mt_best),))
        for i in range(len(mt_best)):
            magnitude_array[i] = round(mt_to_magnitude(mt_best[i,:]), 2)
        magnitude = round(np.mean(magnitude_array), 1) if not np.isnan(np.mean(magnitude_array)) else None
        print(f" Best fitting magnitude -- {magnitude} ")
        ## --------------------------------------------- ##

        return magnitude

def nandc_inv_main(args, event):

    start_time = time.time()

    # create savepath
    savepath = Path(args.save_dir) / event.evid
    savepath.mkdir(parents=True, exist_ok=True)
    
    # log file
    flog = (savepath / f"{event.evid}.log").open("w")
    flog.write(build_event_log_section(event))
    flog.write(build_data_log_section(event))
    
    # ---------------------------------------------------------------------------------------------------------- #

    #-- grid search begin --#
    # step 1 搜索最优震级
    if args.weight_amp == 0:
        print(" ----  Step 1: Skipping magnitude searching (zero weight for P-wave amplitude) ---- ")
        magnitude = None
        weight_list = [args.weight_pol, 0, args.weight_spr]
    elif len(event.amp) == 0:
        print(" ----  Step 1: Skipping magnitude searching (no P-wave amplitude) ---- ")
        magnitude = None
        weight_list = [args.weight_pol, 0, args.weight_spr]
    else:
        print(" ----  Step 1: Searching Best Fitting magnitude ---- ")
        try:
            magnitude = searching_magnitude(event, args)
            weight_list = [args.weight_pol, args.weight_amp, args.weight_spr]
        except Exception as e:
            print(f" Error in searching magnitude: {e} ")
            traceback.print_exc()
            flog.close()
            raise

    apply_effective_gap_stats(event, weight_list)
    flog.write(build_config_log_section(args, weight_list))
    flog.write(build_coverage_log_section(event))
    flog.write(build_coverage_log_section(event, effective=True))
         
    # ---------------------------------------------------------------------------------------------------------- #

    # step 2 搜索全局最优震源机制
    print(" ----  Step 2: Searching Optimal Moment Tensor ---- ")
    norm_list = [args.norm_pol, args.norm_amp, args.norm_spr]
    if args.inv_method == "FMT":
        # ------------------------------------------------------------------------------------------------- #
        #                                Pure Full moment tensor Inversion                                  #
        # ------------------------------------------------------------------------------------------------- #
        grid_fmt = FullMomentTensorGridSegment(magnitude, args.npts_v, args.npts_w, args.npts_kappa, args.npts_sigma, args.npts_h, args.tightness, args.uniformity)
       
        # ------------------------------------- 网格搜索 ---------------------------------------- #
        gs_fmt, misfit_fmt = GridSearchOptimalParallel(event, grid_fmt, weight_list, norm_list, args.normalization, args.n_jobs, args.chunk_size)

        # ------------------------------------- 分析反演结果 ---------------------------------------- #
        # 全空间中误差最小的解，数量1（理论上）
        tt_best_fmt, mt_best_fmt = get_best_fullspace(gs_fmt)
        min_misfit_fmt_index = np.argmin(misfit_fmt[:, 0])

        # 每一种震源类型(v,w)所对应的误差最小的解 for lune plot 数量v*w 只有fmt反演需要
        tts_lune, mts_lune = get_best_lunespace(gs_fmt)

        ##### 验证代码块 ##### 仅用于验证 get_best_fullspace 和 get_best_lunespace 的正确性，正式运行时可以注释掉
        # grid_array = grid_fmt.stack(z=grid_fmt.dims).coords.to_index().to_frame(index=False).to_numpy()

        # best_full = tt_best_fmt[0, :6].astype(float)
        # best_argmin = grid_array[min_misfit_fmt_index, :6].astype(float)

        # lune_min_idx = np.argmin(tts_lune[:, 6])
        # best_lune = tts_lune[lune_min_idx, :6].astype(float)

        # print("check full vs argmin:", np.allclose(best_full, best_argmin))
        # print("check misfit:", np.isclose(float(tt_best_fmt[0, 6]), float(misfit_fmt[min_misfit_fmt_index, 0])))
        # print("check lune vs global:", np.allclose(best_lune, best_full))
        # print("check lune misfit:", np.isclose(float(tts_lune[lune_min_idx, 6]), float(tt_best_fmt[0, 6])))

        quality = evaluate_quality_from_grid(
            event=event,
            weight_list=weight_list,
            best_misfit_row=misfit_fmt[min_misfit_fmt_index],
            grid=grid_fmt,
            misfit_all=misfit_fmt,
            consistency_n_jobs=args.n_jobs,
        )

        # ------------------------------------- 保存结果 ---------------------------------------- #
        flog.write(build_result_log_section("FMT", magnitude, misfit_fmt[min_misfit_fmt_index]))
        
        write_results(savepath, event, mt_best_fmt, magnitude, misfit_fmt[min_misfit_fmt_index, 0], weight_list)
        write_file_lune(savepath, event.evid+'_lune.txt', tts_lune) # tts_lune[:, 6]这一列是误差

        # ------------------------------------- 绘图 ---------------------------------------- #
        if not args.no_plot:
            # beachball
            plot_results_beachball(savepath, args, event, weight_list, mt_best_fmt)

            #  lune plot
            best_gamma, best_delta = mt_to_gamma_delta(mt_best_fmt)
            misfit_lune = np.column_stack([v_to_gamma(tts_lune[:, 1]), w_to_delta(tts_lune[:, 2]), tts_lune[:,6]])
            plot_misfit_lune_pygmt(savepath=savepath, filename=event.evid+'_lune', filetype=args.figtype, 
                                   lune_value=misfit_lune, lune_marker=[best_gamma, best_delta], colormap='no_green')
        
        flog.write(build_quality_log_section(quality))

    elif args.inv_method == "DC":
        # ------------------------------------------------------------------------------------------------- #
        #                                Double Couple Inversion                                            #
        # ------------------------------------------------------------------------------------------------- #
        grid_dc = DoubleCoupleGridSegment(magnitude, args.npts_kappa, args.npts_sigma, args.npts_h)
        
        # ------------------------------------- 网格搜索 ---------------------------------------- #
        gs_dc, misfit_dc = GridSearchOptimalParallel(event, grid_dc, weight_list, norm_list, args.normalization, args.n_jobs, args.chunk_size)

        # ------------------------------------- 分析反演结果 ------------------------------------- #
        # 全空间中误差最小的解，数量1（理论上）
        tt_best_dc, mt_best_dc = get_best_fullspace(gs_dc)
        min_misfit_dc_index = np.argmin(misfit_dc[:, 0])
        
        quality = evaluate_quality_from_grid(
            event=event,
            weight_list=weight_list,
            best_misfit_row=misfit_dc[min_misfit_dc_index],
            grid=grid_dc,
            misfit_all=misfit_dc,
            consistency_n_jobs=args.n_jobs,
        )
        # ------------------------------------- 保存结果 ---------------------------------------- #
        flog.write(build_result_log_section("DC", magnitude, misfit_dc[min_misfit_dc_index]))
        
        write_results(savepath, event, mt_best_dc, magnitude, misfit_dc[min_misfit_dc_index, 0], weight_list)

        # # ------------------------------------- 绘图 ---------------------------------------- #
        if not args.no_plot:
            # beachball
            plot_results_beachball(savepath, args, event, weight_list, mt_best_dc)

        flog.write(build_quality_log_section(quality))

    elif args.inv_method == "DC2FMT":
        # ------------------------------------------------------------------------------------------------- #
        #                    Double Couple to Full Moment Tensor Inversion                                  #
        # ------------------------------------------------------------------------------------------------- #
        grid_dc = DoubleCoupleGridSegment(magnitude, args.npts_kappa, args.npts_sigma, args.npts_h)
        
        # ------------------------------------- 网格搜索-1 ---------------------------------------- #
        gs_dc, misfit_dc = GridSearchOptimalParallel(event, grid_dc, weight_list, norm_list, args.normalization, args.n_jobs, args.chunk_size)
        min_misfit_dc_index = np.argmin(misfit_dc[:, 0])
        tt_accepted_dc, mt_accepted_dc = get_accepted_fullspace(gs_dc, tolerance=1.1*misfit_dc[min_misfit_dc_index, 0])

        if len(tt_accepted_dc) == 0:
            raise RuntimeError("No accepted DC solutions available for constrained FMT search.")
        # ------------------------------------- 网格搜索-2 ---------------------------------------- #
        model_table_fmt = ConstrainedFullMomentTensorModelTable(tt_accepted_dc, args.npts_v, args.npts_w, args.tightness, args.uniformity)

        if len(model_table_fmt) == 0:
            raise RuntimeError("Constrained FMT model table is empty.")

        result_table_fmt, misfit_fmt = GridSearchOptimalParallelFromTable(event, model_table_fmt, weight_list, norm_list, args.normalization, args.n_jobs, args.chunk_size)

        if len(result_table_fmt) == 0:
            raise RuntimeError("No constrained FMT results were produced.")

        tt_best_fmt, mt_best_fmt = get_best_from_result_table(result_table_fmt)
        min_misfit_fmt_index = np.argmin(misfit_fmt[:, 0])
        
        tt_accepted_fmt, mt_accepted_fmt = get_accepted_from_result_table(result_table_fmt, tolerance=1.1 * misfit_fmt[min_misfit_fmt_index, 0])
        
        tts_lune, mts_lune = get_best_lunespace_from_result_table(result_table_fmt)

        flog.write(build_result_log_section("DC2FMT", magnitude, misfit_fmt[min_misfit_fmt_index]))
        flog.write(f"dc2fmt.num_dc_accepted = {len(tt_accepted_dc)}\n")
        flog.write(f"dc2fmt.num_constrained_models = {len(model_table_fmt)}\n")

        quality = evaluate_quality_from_table(
            event=event,
            weight_list=weight_list,
            best_misfit_row=misfit_fmt[min_misfit_fmt_index],
            result_table=result_table_fmt,
            consistency_n_jobs=args.n_jobs,
        )
        
        write_results(savepath, event, mt_best_fmt, magnitude, misfit_fmt[min_misfit_fmt_index, 0], weight_list)
        write_file_lune(savepath, event.evid + "_lune.txt", tts_lune)

        if not args.no_plot:
            plot_results_beachball(savepath, args, event, weight_list, mt_best_fmt)

            best_gamma, best_delta = mt_to_gamma_delta(mt_best_fmt)
            misfit_lune = np.column_stack([v_to_gamma(tts_lune[:, 1]), w_to_delta(tts_lune[:, 2]), tts_lune[:, 6]])
            plot_misfit_lune_pygmt(savepath=savepath, filename=event.evid + "_lune", filetype=args.figtype,lune_value=misfit_lune, lune_marker=[best_gamma, best_delta], colormap="no_green")

        flog.write(build_quality_log_section(quality))

    # ---------------------------------------------------------------------------------------------------------- #
    running_time = round(time.time()-start_time, 1)
    flog.write(build_runtime_log_section(running_time))
    flog.close()
    print(f" --------  Event ID {event.evid}  RunTime: {running_time}s -------- \n")
    #------ inversion loop ------#
