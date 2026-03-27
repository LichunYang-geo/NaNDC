import numpy as np

def _format_log_value(value, precision=None):
    if value is None:
        return "nan"
    if isinstance(value, (bool, np.bool_)):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if precision is None:
            return str(float(value))
        return f"{float(value):.{precision}f}"
    return str(value)


def _build_log_section(title, fields):
    lines = [f"[{title}]"]
    for key, value, precision in fields:
        lines.append(f"{key} = {_format_log_value(value, precision)}")
    return "\n".join(lines) + "\n\n"


def build_event_log_section(event):
    return _build_log_section("event", [
        ("event.id", event.evid, None),
        ("event.longitude_deg", event.evlo, 6),
        ("event.latitude_deg", event.evla, 6),
        ("event.depth_km", event.evdp, 3),
        ("event.magnitude_input", getattr(event, "evmag", np.nan), 2)
    ])


def build_data_log_section(event):
    return _build_log_section("data", [
        ("data.num_sta", len(event.stations), None),
        ("data.num_pol", len(event.pol), None),
        ("data.num_amp", len(event.amp), None),
        ("data.num_spr", len(event.spr), None)
    ])


def build_coverage_log_section(event, effective=False):
    if effective:
        title = "coverage_effective"
        prefix = "coverage_effective"
        pol_max_azi_gap = event.pol_effective_max_azi_gap
        pol_max_takeoff_gap = event.pol_effective_max_takeoff_gap
        amp_max_azi_gap = event.amp_effective_max_azi_gap
        amp_max_takeoff_gap = event.amp_effective_max_takeoff_gap
        spr_max_azi_gap = event.spr_effective_max_azi_gap
        spr_max_takeoff_gap = event.spr_effective_max_takeoff_gap
    else:
        title = "coverage"
        prefix = "coverage"
        pol_max_azi_gap = event.pol_max_azi_gap
        pol_max_takeoff_gap = event.pol_max_takeoff_gap
        amp_max_azi_gap = event.amp_max_azi_gap
        amp_max_takeoff_gap = event.amp_max_takeoff_gap
        spr_max_azi_gap = event.spr_max_azi_gap
        spr_max_takeoff_gap = event.spr_max_takeoff_gap

    return _build_log_section(title, [
        (f"{prefix}.pol.max_azi_gap_deg", pol_max_azi_gap, 2),
        (f"{prefix}.pol.max_takeoff_gap_deg", pol_max_takeoff_gap, 2),
        (f"{prefix}.amp.max_azi_gap_deg", amp_max_azi_gap, 2),
        (f"{prefix}.amp.max_takeoff_gap_deg", amp_max_takeoff_gap, 2),
        (f"{prefix}.spr.max_azi_gap_deg", spr_max_azi_gap, 2),
        (f"{prefix}.spr.max_takeoff_gap_deg", spr_max_takeoff_gap, 2)
    ])


def build_config_log_section(args, weight_list=None):
    if weight_list is None:
        weight_pol, weight_amp, weight_spr = args.weight_pol, args.weight_amp, args.weight_spr
    else:
        weight_pol, weight_amp, weight_spr = weight_list

    return _build_log_section("config", [
        ("config.inv_method", args.inv_method, None),
        ("config.weight_pol", weight_pol, None),
        ("config.weight_amp", weight_amp, None),
        ("config.weight_spr", weight_spr, None),
        ("config.normalization", args.normalization, None),
        ("config.norm_pol", args.norm_pol, None),
        ("config.norm_amp", args.norm_amp, None),
        ("config.norm_spr", args.norm_spr, None),
        ("config.n_jobs", args.n_jobs, None),
        ("config.chunk_size", args.chunk_size, None),
    ])


def build_result_log_section(method, magnitude, misfit):
    return _build_log_section("result", [
        ("result.method", method, None),
        ("result.best_magnitude", magnitude, 2),
        ("result.misfit.total", misfit[0], 6),
        ("result.misfit.pol", misfit[1], 6),
        ("result.misfit.amp", misfit[2], 6),
        ("result.misfit.spr", misfit[3], 6)
    ])


def build_quality_log_section(quality):
    flags = ", ".join(quality["flags"]) if quality["flags"] else "none"
    return _build_log_section("quality", [
        ("quality.overall_grade", quality["overall_grade"], None),
        ("quality.coverage_grade", quality["coverage_grade"], None),
        ("quality.count_grade", quality["count_grade"], None),
        ("quality.consistency_grade", quality["consistency_grade"], None),
        ("quality.fit_grade", quality["fit_grade"], None),
        ("quality.sharpness_grade", quality["sharpness_grade"], None),
        ("quality.coverage.max_azi_gap_active", quality["coverage"]["max_azi_gap_active"], 2),
        ("quality.coverage.max_takeoff_gap_active", quality["coverage"]["max_takeoff_gap_active"], 2),
        ("quality.count.num_pol", quality["counts"]["num_pol"], None),
        ("quality.count.num_amp", quality["counts"]["num_amp"], None),
        ("quality.count.num_spr", quality["counts"]["num_spr"], None),
        ("quality.count.min_active_count", quality["counts"]["min_active_count"], None),
        ("quality.fit.misfit_mean_active", quality["fit"]["misfit_mean_active"], 6),
        ("quality.stability.sharpness.low_error_count", quality["stability"]["sharpness"]["low_error_count"], None),
        ("quality.stability.sharpness.misfit_at_rank", quality["stability"]["sharpness"]["misfit_at_rank"], 6),
        ("quality.stability.sharpness.relative_span", quality["stability"]["sharpness"]["relative_span"], 6),
        ("quality.stability.consistency.near_optimal_ratio", quality["stability"]["consistency"]["near_optimal_ratio"], 4),
        ("quality.stability.consistency.near_optimal_count", quality["stability"]["consistency"]["near_optimal_count"], None),
        ("quality.stability.consistency.max_pairwise_omega_deg", quality["stability"]["consistency"]["max_pairwise_omega_deg"], 2),
        ("quality.flags", flags, None),
    ])


def build_runtime_log_section(running_time):
    return _build_log_section("runtime", [
        ("runtime.seconds", running_time, 1)
    ])
