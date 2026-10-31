import numpy as np

from nandc_mt import tt_to_mij, get_full_MT_array


DEFAULT_QUALITY_CONFIG = {
    "coverage": {
        "A": {"max_azi_gap": 90.0, "max_takeoff_gap": 30.0},
        "B": {"max_azi_gap": 135.0, "max_takeoff_gap": 50.0},
    },
    "fit": {
        "A": {"max_total": 0.50},
        "B": {"max_total": 0.80},
    },
    "stability": {
        "near_optimal_ratio": 0.05,
        "misfit_rank_count": 200,
        "A": {"min_misfit_rank_ratio": 0.10, "max_omega_p95_deg": 25.0},
        "B": {"min_misfit_rank_ratio": 0.05, "max_omega_p95_deg": 40.0},
    },
}


def _as_misfit_row(best_misfit_row):
    """Return the first four misfit terms as [total, pol, amp, spr]."""
    row = np.asarray(best_misfit_row, dtype=np.float64).reshape(-1)
    if row.size < 4:
        raise ValueError("`best_misfit_row` must contain at least [total, pol, amp, spr]")
    return row[:4]


def _as_weight_list(weight_list):
    """Return inversion weights as [pol, amp, spr]."""
    weight = np.asarray(weight_list, dtype=np.float64).reshape(-1)
    if weight.size != 3:
        raise ValueError("`weight_list` must contain exactly three values")
    return weight


def _validate_nonnegative(name, value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"`{name}` must be a finite non-negative number")
    return value


def _validate_positive_int(name, value):
    """Return a positive integer after rejecting bools and non-integral values."""
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be a positive integer")

    value_float = float(value)
    if not np.isfinite(value_float) or value_float <= 0.0 or not value_float.is_integer():
        raise ValueError(f"`{name}` must be a positive integer")

    return int(value_float)


def _relative_misfit_increase(value, reference):
    """Return the relative increase from a reference misfit."""
    value = float(value)
    reference = float(reference)
    if reference > 0.0:
        return float((value - reference) / reference)
    if value == reference:
        return 0.0
    return np.inf


def ensure_effective_gap_stats(event, weight_list):
    """Ensure effective coverage gaps are available on the event object."""
    _as_weight_list(weight_list)

    required_attrs = (
        "pol_effective_max_azi_gap",
        "pol_effective_max_takeoff_gap",
        "amp_effective_max_azi_gap",
        "amp_effective_max_takeoff_gap",
        "spr_effective_max_azi_gap",
        "spr_effective_max_takeoff_gap",
    )
    missing_attrs = [name for name in required_attrs if not hasattr(event, name)]
    if missing_attrs:
        raise AttributeError(
            "Effective gap statistics are missing on `event`. "
            "Please construct them before calling `nandc_quality`. "
            f"Missing attributes: {', '.join(missing_attrs)}"
        )

    if hasattr(event, "effective_gap_stats"):
        return event.effective_gap_stats

    effective_gap_stats = {
        "pol": {
            "max_azi_gap": float(event.pol_effective_max_azi_gap),
            "max_takeoff_gap": float(event.pol_effective_max_takeoff_gap),
        },
        "amp": {
            "max_azi_gap": float(event.amp_effective_max_azi_gap),
            "max_takeoff_gap": float(event.amp_effective_max_takeoff_gap),
        },
        "spr": {
            "max_azi_gap": float(event.spr_effective_max_azi_gap),
            "max_takeoff_gap": float(event.spr_effective_max_takeoff_gap),
        },
    }
    event.effective_gap_stats = effective_gap_stats
    return effective_gap_stats


def build_result_table_from_grid(grid, misfit_all):
    """Convert a grid-search result into [model parameters, misfit terms]."""
    if grid is None:
        raise ValueError("`grid` must not be None")

    misfit_all = np.asarray(misfit_all, dtype=np.float64)
    if misfit_all.ndim != 2 or misfit_all.shape[1] < 4:
        raise ValueError("`misfit_all` must be a 2-D array with at least 4 columns")

    grid_array = np.ascontiguousarray(
        np.asarray(
            grid.stack(z=grid.dims).coords.to_index().to_frame(index=False).to_numpy(),
            dtype=np.float64,
        )
    )

    if grid_array.shape[0] != misfit_all.shape[0]:
        raise ValueError("`grid` and `misfit_all` do not contain the same number of models")

    if grid_array.shape[1] < 6:
        raise ValueError("Expanded grid must provide at least 6 model parameters")

    return np.column_stack([grid_array[:, :6], misfit_all[:, :4]])


def build_result_table_from_table(result_table):
    """Return the first ten columns of an existing result table."""
    result_table = np.asarray(result_table, dtype=np.float64)
    if result_table.ndim != 2 or result_table.shape[1] < 10:
        raise ValueError("`result_table` must be a 2-D array with at least 10 columns")
    return result_table[:, :10].copy()


def _build_result_table(grid=None, misfit_all=None, result_table=None):
    """Build the unified result table from either grid arrays or a table."""
    if result_table is not None:
        return build_result_table_from_table(result_table)
    if grid is not None and misfit_all is not None:
        return build_result_table_from_grid(grid, misfit_all)
    raise ValueError("Provide either (`grid`, `misfit_all`) or `result_table`")


def _active_component_info(event, weight_list):
    """Describe which data components are active in the current inversion."""
    weight = _as_weight_list(weight_list)
    component_defs = (
        ("pol", weight[0], len(event.pol)),
        ("amp", weight[1], len(event.amp)),
        ("spr", weight[2], len(event.spr)),
    )

    return [
        {
            "name": name,
            "weight": float(this_weight),
            "count": int(count),
            "active": bool(this_weight != 0 and count > 0),
        }
        for name, this_weight, count in component_defs
    ]


def compute_quality_counts(event, weight_list):
    """Collect observation counts as diagnostics; counts do not grade quality."""
    component_info = _active_component_info(event, weight_list)
    active_counts = [item["count"] for item in component_info if item["active"]]

    return {
        "num_sta": int(len(event.stations)),
        "num_pol": int(len(event.pol) if component_info[0]["active"] else 0),
        "num_amp": int(len(event.amp) if component_info[1]["active"] else 0),
        "num_spr": int(len(event.spr) if component_info[2]["active"] else 0),
        "num_effective": int(sum(active_counts)),
        "num_active_types": int(len(active_counts)),
        "min_active_count": int(min(active_counts)) if active_counts else 0,
        "active_counts": active_counts,
    }


def compute_quality_coverage(event, weight_list):
    """Compute geometry-only coverage metrics for active data components."""
    ensure_effective_gap_stats(event, weight_list)

    component_info = _active_component_info(event, weight_list)
    active_names = [item["name"] for item in component_info if item["active"]]

    coverage = {
        "pol_max_azi_gap": float(getattr(event, "pol_effective_max_azi_gap", np.nan)),
        "pol_max_takeoff_gap": float(getattr(event, "pol_effective_max_takeoff_gap", np.nan)),
        "amp_max_azi_gap": float(getattr(event, "amp_effective_max_azi_gap", np.nan)),
        "amp_max_takeoff_gap": float(getattr(event, "amp_effective_max_takeoff_gap", np.nan)),
        "spr_max_azi_gap": float(getattr(event, "spr_effective_max_azi_gap", np.nan)),
        "spr_max_takeoff_gap": float(getattr(event, "spr_effective_max_takeoff_gap", np.nan)),
    }

    if active_names:
        coverage["max_azi_gap_active"] = float(max(coverage[f"{name}_max_azi_gap"] for name in active_names))
        coverage["max_takeoff_gap_active"] = float(
            max(coverage[f"{name}_max_takeoff_gap"] for name in active_names)
        )
    else:
        coverage["max_azi_gap_active"] = np.nan
        coverage["max_takeoff_gap_active"] = np.nan

    return coverage


def compute_quality_fit(event, weight_list, best_misfit_row):
    """Compute fit diagnostics from the best total and component misfits."""
    best_row = _as_misfit_row(best_misfit_row)
    component_info = _active_component_info(event, weight_list)
    component_misfits = {
        "pol": float(best_row[1]),
        "amp": float(best_row[2]),
        "spr": float(best_row[3]),
    }

    # Keep the active-component mean as a diagnostic only; grading uses total misfit.
    active_terms = [
        component_misfits[item["name"]]
        for item in component_info
        if item["active"]
    ]

    return {
        "misfit_total": float(best_row[0]),
        "misfit_pol": component_misfits["pol"],
        "misfit_amp": component_misfits["amp"],
        "misfit_spr": component_misfits["spr"],
        "misfit_mean_active": float(np.mean(active_terms)) if active_terms else np.nan,
    }


def build_relative_misfit_subset(result_table, min_misfit, ratio):
    """Return models whose total misfit is within a relative tolerance."""
    table = np.asarray(result_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 10:
        raise ValueError("`result_table` must be a 2-D array with at least 10 columns")

    ratio = _validate_nonnegative("ratio", ratio)
    threshold = float(min_misfit) * (1.0 + ratio)

    if table.shape[0] == 0:
        subset = np.empty((0, 10), dtype=np.float64)
    else:
        subset = table[table[:, 6] <= threshold]
        subset = subset[np.argsort(subset[:, 6])]

    return {
        "subset_table": subset,
        "subset_size": int(subset.shape[0]),
        "threshold": threshold,
        "ratio": ratio,
    }


def _result_table_to_mt(result_table):
    """Convert result-table Tape parameters into moment tensor vectors."""
    table = np.asarray(result_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 6:
        raise ValueError("`result_table` must be a 2-D array with at least 6 columns")

    if table.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float64)

    return tt_to_mij(
        table[:, 0], table[:, 1], table[:, 2],
        table[:, 3], table[:, 4], table[:, 5],
    )


def _mts_to_unit_flat(mts):
    """Convert moment tensors to normalized flattened full matrices."""
    mts = np.asarray(mts, dtype=np.float64)
    if mts.ndim != 2 or mts.shape[1] != 6:
        raise ValueError("`mts` must be a 2-D array with shape (N, 6)")

    n_model = mts.shape[0]
    full_mts = np.stack([get_full_MT_array(mt) for mt in mts], axis=0)
    flat_mts = full_mts.reshape(n_model, -1)
    norms = np.linalg.norm(flat_mts, axis=1)
    valid_norm = norms > 0.0
    unit_flat = np.divide(
        flat_mts,
        norms[:, None],
        out=np.zeros_like(flat_mts),
        where=valid_norm[:, None],
    )
    return unit_flat, valid_norm


def _omega_to_reference_deg(mts, reference_mt):
    """Compute omega angles from each moment tensor to one reference tensor."""
    mts = np.asarray(mts, dtype=np.float64)
    reference_mt = np.asarray(reference_mt, dtype=np.float64).reshape(1, 6)
    if mts.ndim != 2 or mts.shape[1] != 6:
        raise ValueError("`mts` must be a 2-D array with shape (N, 6)")

    if mts.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    unit_flat, valid_norm = _mts_to_unit_flat(mts)
    reference_unit, reference_valid = _mts_to_unit_flat(reference_mt)

    if not reference_valid[0]:
        return np.full(mts.shape[0], np.nan, dtype=np.float64)

    # Invalid zero-norm MTs cannot define an angle, so mark them as nan.
    cosom = unit_flat @ reference_unit[0]
    cosom = np.clip(cosom, -1.0, 1.0)
    omega = np.rad2deg(np.arccos(cosom))
    omega[~valid_norm] = np.nan
    return omega


def compute_quality_stability(
    result_table,
    min_misfit,
    *,
    near_optimal_ratio=0.05,
    misfit_rank_count=200,
):
    """Measure stability from low-error sharpness and mechanism spread."""
    table = np.asarray(result_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 10:
        raise ValueError("`result_table` must be a 2-D array with at least 10 columns")

    near_optimal_ratio = _validate_nonnegative("near_optimal_ratio", near_optimal_ratio)
    misfit_rank_count = _validate_positive_int("misfit_rank_count", misfit_rank_count)
    near_optimal = build_relative_misfit_subset(table, min_misfit, near_optimal_ratio)
    candidate_count = int(table.shape[0])
    accepted_count = int(near_optimal["subset_size"])

    if candidate_count == 0:
        misfit_rank_index = 0
        misfit_min = np.nan
        misfit_at_rank = np.nan
        misfit_rank_ratio = np.nan
    else:
        # Use a fixed low-error rank so this metric is not diluted by huge grids.
        sorted_table = table[np.argsort(table[:, 6])]
        misfit_rank_index = min(misfit_rank_count, candidate_count)
        misfit_min = float(sorted_table[0, 6])
        misfit_at_rank = float(sorted_table[misfit_rank_index - 1, 6])
        misfit_rank_ratio = _relative_misfit_increase(misfit_at_rank, misfit_min)

    if candidate_count == 0 or accepted_count == 0:
        omega_p95_deg = np.nan
        omega_max_deg = np.nan
    else:
        # The accepted ensemble is judged by its angular spread around the best model.
        best_table = table[np.argsort(table[:, 6])][0:1]
        reference_mt = _result_table_to_mt(best_table)[0]
        accepted_mt = _result_table_to_mt(near_optimal["subset_table"])
        omega_to_best = _omega_to_reference_deg(accepted_mt, reference_mt)
        finite_omega = omega_to_best[np.isfinite(omega_to_best)]
        if finite_omega.size == 0:
            omega_p95_deg = np.nan
            omega_max_deg = np.nan
        else:
            omega_p95_deg = float(np.percentile(finite_omega, 95.0))
            omega_max_deg = float(np.max(finite_omega))

    return {
        "near_optimal_ratio": near_optimal_ratio,
        "accepted_threshold": float(near_optimal["threshold"]),
        "accepted_count": accepted_count,
        "candidate_count": candidate_count,
        "misfit_rank_count": int(misfit_rank_count),
        "misfit_rank_index": int(misfit_rank_index),
        "misfit_min": misfit_min,
        "misfit_at_rank": misfit_at_rank,
        "misfit_rank_ratio": misfit_rank_ratio,
        "omega_p95_deg": omega_p95_deg,
        "omega_max_deg": omega_max_deg,
    }


def grade_coverage(coverage):
    """Grade geometry coverage using active azimuth and takeoff gaps only."""
    max_azi_gap = float(coverage["max_azi_gap_active"])
    max_takeoff_gap = float(coverage["max_takeoff_gap_active"])

    if (
        max_azi_gap <= DEFAULT_QUALITY_CONFIG["coverage"]["A"]["max_azi_gap"]
        and max_takeoff_gap <= DEFAULT_QUALITY_CONFIG["coverage"]["A"]["max_takeoff_gap"]
    ):
        return "A"
    if (
        max_azi_gap <= DEFAULT_QUALITY_CONFIG["coverage"]["B"]["max_azi_gap"]
        and max_takeoff_gap <= DEFAULT_QUALITY_CONFIG["coverage"]["B"]["max_takeoff_gap"]
    ):
        return "B"
    return "C"


def grade_fit(fit):
    """Grade fit with the best total misfit only."""
    misfit_total = float(fit["misfit_total"])
    if misfit_total <= DEFAULT_QUALITY_CONFIG["fit"]["A"]["max_total"]:
        return "A"
    if misfit_total <= DEFAULT_QUALITY_CONFIG["fit"]["B"]["max_total"]:
        return "B"
    return "C"


def grade_misfit_sharpness(stability):
    """Grade whether low-error models rise quickly away from the best model."""
    misfit_rank_ratio = float(stability["misfit_rank_ratio"])
    if misfit_rank_ratio >= DEFAULT_QUALITY_CONFIG["stability"]["A"]["min_misfit_rank_ratio"]:
        return "A"
    if misfit_rank_ratio >= DEFAULT_QUALITY_CONFIG["stability"]["B"]["min_misfit_rank_ratio"]:
        return "B"
    return "C"


def grade_mechanism_consistency(stability):
    """Grade whether accepted mechanisms cluster around the best model."""
    omega_p95_deg = float(stability["omega_p95_deg"])
    if omega_p95_deg <= DEFAULT_QUALITY_CONFIG["stability"]["A"]["max_omega_p95_deg"]:
        return "A"
    if omega_p95_deg <= DEFAULT_QUALITY_CONFIG["stability"]["B"]["max_omega_p95_deg"]:
        return "B"
    return "C"


def grade_stability(stability):
    """Combine misfit sharpness and mechanism consistency into stability."""
    if (
        stability["misfit_sharpness_grade"] == "A"
        and stability["mechanism_consistency_grade"] == "A"
    ):
        return "A"
    if (
        stability["misfit_sharpness_grade"] != "C"
        and stability["mechanism_consistency_grade"] != "C"
    ):
        return "B"
    return "C"


def grade_solution_quality(fit_grade, stability_grade, coverage_grade):
    """Combine the three independent quality axes into one solution grade."""
    if fit_grade == "A" and stability_grade == "A" and coverage_grade == "A":
        return "A"
    if fit_grade != "C" and stability_grade != "C" and coverage_grade != "C":
        return "B"
    return "C"


def build_quality_flags(coverage_grade, fit_grade, stability_grade):
    """Build concise flags for any axis that prevents a reliable solution."""
    flags = []
    if coverage_grade == "C":
        flags.append("low_coverage")
    if fit_grade == "C":
        flags.append("high_misfit")
    if stability_grade == "C":
        flags.append("low_stability")
    return flags


def evaluate_solution_quality(
    event,
    weight_list,
    best_misfit_row,
    *,
    grid=None,
    misfit_all=None,
    result_table=None,
    consistency_n_jobs=1,
):
    """
    Evaluate inversion quality for one event using the unified result-table format.

    Parameters
    ----------
    event : DataIN-like object
        Event object produced by the existing NaNDC data-reading workflow.
        This function assumes effective gap fields have already been attached
        to `event` by the inversion workflow.
    weight_list : sequence of 3 floats
        Effective weights actually used in inversion, ordered as
        [polarity, amplitude, S/P ratio].
    best_misfit_row : array-like
        Best-fitting misfit row with at least four values:
        [total, pol, amp, spr].
    grid, misfit_all : optional
        Use these when the inversion result comes from a full grid search.
        Internally they will be converted into the unified result-table format.
    result_table : optional
        Use this when a result table already exists, such as in DC2FMT.

    Returns
    -------
    dict
        Nested dictionary containing the solution grade, sub-grades, raw
        metrics, and stability diagnostics.
    """
    config = DEFAULT_QUALITY_CONFIG
    stability_config = config["stability"]
    # Kept for API compatibility with existing inversion callers.
    _ = consistency_n_jobs

    best_misfit = _as_misfit_row(best_misfit_row)
    all_result_table = _build_result_table(grid=grid, misfit_all=misfit_all, result_table=result_table)

    # Counts remain useful diagnostics, but they are not part of solution quality.
    counts = compute_quality_counts(event, weight_list)
    coverage = compute_quality_coverage(event, weight_list)
    fit = compute_quality_fit(event, weight_list, best_misfit)
    stability = compute_quality_stability(
        all_result_table,
        fit["misfit_total"],
        near_optimal_ratio=stability_config["near_optimal_ratio"],
        misfit_rank_count=stability_config["misfit_rank_count"],
    )
    stability["misfit_sharpness_grade"] = grade_misfit_sharpness(stability)
    stability["mechanism_consistency_grade"] = grade_mechanism_consistency(stability)

    coverage_grade = grade_coverage(coverage)
    fit_grade = grade_fit(fit)
    stability_grade = grade_stability(stability)
    solution_quality = grade_solution_quality(
        fit_grade,
        stability_grade,
        coverage_grade,
    )

    flags = build_quality_flags(
        coverage_grade,
        fit_grade,
        stability_grade,
    )

    return {
        "solution_quality": solution_quality,
        "overall_grade": solution_quality,
        "coverage_grade": coverage_grade,
        "fit_grade": fit_grade,
        "stability_grade": stability_grade,
        "coverage": coverage,
        "counts": counts,
        "fit": fit,
        "stability": stability,
        "flags": flags,
    }


def evaluate_quality_from_grid(event, weight_list, best_misfit_row, grid, misfit_all, consistency_n_jobs=1):
    """
    Convenience wrapper for DC/FMT-style workflows.

    Parameters
    ----------
    grid : xarray.DataArray
        Search grid used during inversion.
    misfit_all : ndarray, shape (N, 4)
        Misfit table returned by grid search, ordered as
        [total, pol, amp, spr].
    """
    return evaluate_solution_quality(
        event,
        weight_list,
        best_misfit_row,
        grid=grid,
        misfit_all=misfit_all,
        consistency_n_jobs=consistency_n_jobs,
    )


def evaluate_quality_from_table(event, weight_list, best_misfit_row, result_table, consistency_n_jobs=1):
    """
    Convenience wrapper for workflows that already produce a result table.

    Parameters
    ----------
    result_table : ndarray, shape (N, >=10)
        Table ordered as
        [rho, v, w, kappa, sigma, h, total, pol, amp, spr, ...].
        Extra trailing columns are ignored by the quality module.
    """
    return evaluate_solution_quality(
        event,
        weight_list,
        best_misfit_row,
        result_table=result_table,
        consistency_n_jobs=consistency_n_jobs,
    )
