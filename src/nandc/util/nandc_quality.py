import numpy as np

from nandc_mt import tt_to_mij, get_full_MT_array


DEFAULT_QUALITY_CONFIG = {
    "coverage": {
        "A": {"max_azi_gap": 90.0, "max_takeoff_gap": 30.0},
        "B": {"max_azi_gap": 135.0, "max_takeoff_gap": 50.0},
    },
    "count": {
        "A": {"min_active_each": 15},
        "B": {"min_active_each": 10},
    },
    "fit": {
        "A": {"max_mean_active": 0.20},
        "B": {"max_mean_active": 0.35},
    },
    "sharpness": {
        "low_error_count": 200,
        "A": {"min_relative_span": 0.10},
        "B": {"min_relative_span": 0.05},
    },
    "consistency": {
        "near_optimal_ratio": 0.05,
        "A": {"max_angle_deg": 25.0},
        "B": {"max_angle_deg": 40.0},
    },
}


def _as_misfit_row(best_misfit_row):
    row = np.asarray(best_misfit_row, dtype=np.float64).reshape(-1)
    if row.size < 4:
        raise ValueError("`best_misfit_row` must contain at least [total, pol, amp, spr]")
    return row[:4]


def _as_weight_list(weight_list):
    weight = np.asarray(weight_list, dtype=np.float64).reshape(-1)
    if weight.size != 3:
        raise ValueError("`weight_list` must contain exactly three values")
    return weight


def _validate_fraction(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"`{name}` must be within (0, 1]")
    return value


def _validate_nonnegative(name, value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"`{name}` must be a finite non-negative number")
    return value


def _validate_positive_int(name, value):
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be a positive integer")

    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a positive integer") from exc

    if not np.isfinite(value_float) or value_float <= 0.0 or not value_float.is_integer():
        raise ValueError(f"`{name}` must be a positive integer")

    return int(value_float)


def _relative_span(value, reference):
    value = float(value)
    reference = float(reference)
    if reference > 0.0:
        return float((value - reference) / reference)
    if value == reference:
        return 0.0
    return np.inf


def ensure_effective_gap_stats(event, weight_list):
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
    result_table = np.asarray(result_table, dtype=np.float64)
    if result_table.ndim != 2 or result_table.shape[1] < 10:
        raise ValueError("`result_table` must be a 2-D array with at least 10 columns")
    return result_table[:, :10].copy()


def _build_result_table(grid=None, misfit_all=None, result_table=None):
    if result_table is not None:
        return build_result_table_from_table(result_table)
    if grid is not None and misfit_all is not None:
        return build_result_table_from_grid(grid, misfit_all)
    raise ValueError("Provide either (`grid`, `misfit_all`) or `result_table`")


def _active_component_info(event, weight_list):
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
    best_row = _as_misfit_row(best_misfit_row)
    component_info = _active_component_info(event, weight_list)
    component_misfits = {
        "pol": float(best_row[1]),
        "amp": float(best_row[2]),
        "spr": float(best_row[3]),
    }

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
    table = np.asarray(result_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 6:
        raise ValueError("`result_table` must be a 2-D array with at least 6 columns")

    if table.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float64)

    return tt_to_mij(
        table[:, 0], table[:, 1], table[:, 2],
        table[:, 3], table[:, 4], table[:, 5],
    )


def _pairwise_omega_matrix(mts):
    mts = np.asarray(mts, dtype=np.float64)
    if mts.ndim != 2 or mts.shape[1] != 6:
        raise ValueError("`mts` must be a 2-D array with shape (N, 6)")

    n_model = mts.shape[0]
    if n_model == 0:
        return np.empty((0, 0), dtype=np.float64)
    if n_model == 1:
        return np.zeros((1, 1), dtype=np.float64)

    full_mts = np.stack([get_full_MT_array(mt) for mt in mts], axis=0)
    norms = np.linalg.norm(full_mts.reshape(n_model, -1), axis=1)
    dot = np.tensordot(full_mts, full_mts, axes=([1, 2], [1, 2]))

    denom = norms[:, None] * norms[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosom = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)

    cosom = np.clip(cosom, -1.0, 1.0)
    omega = np.rad2deg(np.arccos(cosom))
    np.fill_diagonal(omega, 0.0)
    return omega


def evaluate_sharpness(
    result_table,
    *,
    low_error_count=100,
    min_relative_span_a=0.10,
    min_relative_span_b=0.05,
):
    table = np.asarray(result_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 10:
        raise ValueError("`result_table` must be a 2-D array with at least 10 columns")

    low_error_count = _validate_positive_int("low_error_count", low_error_count)
    min_relative_span_a = _validate_nonnegative("min_relative_span_a", min_relative_span_a)
    min_relative_span_b = _validate_nonnegative("min_relative_span_b", min_relative_span_b)
    if min_relative_span_a < min_relative_span_b:
        raise ValueError("`min_relative_span_a` must be greater than or equal to `min_relative_span_b`")

    if table.shape[0] == 0:
        return {
            "low_error_count": low_error_count,
            "rank_index": 0,
            "candidate_count": 0,
            "misfit_min": np.nan,
            "misfit_at_rank": np.nan,
            "relative_span": np.nan,
            "sharpness_grade": "C",
        }

    sorted_table = table[np.argsort(table[:, 6])]
    candidate_count = sorted_table.shape[0]
    rank_count = min(low_error_count, candidate_count)
    rank_index = rank_count - 1
    misfit_min = float(sorted_table[0, 6])
    misfit_at_rank = float(sorted_table[rank_index, 6])
    relative_span = _relative_span(misfit_at_rank, misfit_min)

    if relative_span >= min_relative_span_a:
        grade = "A"
    elif relative_span >= min_relative_span_b:
        grade = "B"
    else:
        grade = "C"

    return {
        "low_error_count": int(low_error_count),
        "rank_index": int(rank_index + 1),
        "candidate_count": int(candidate_count),
        "misfit_min": misfit_min,
        "misfit_at_rank": misfit_at_rank,
        "relative_span": float(relative_span),
        "sharpness_grade": grade,
    }


def evaluate_consistency(
    result_table,
    min_misfit,
    *,
    near_optimal_ratio=0.05,
    max_angle_deg_a=20.0,
    max_angle_deg_b=30.0,
):
    near_optimal_ratio = _validate_nonnegative("near_optimal_ratio", near_optimal_ratio)
    max_angle_deg_a = _validate_nonnegative("max_angle_deg_a", max_angle_deg_a)
    max_angle_deg_b = _validate_nonnegative("max_angle_deg_b", max_angle_deg_b)
    if max_angle_deg_a > max_angle_deg_b:
        raise ValueError("`max_angle_deg_a` must be less than or equal to `max_angle_deg_b`")

    near_optimal = build_relative_misfit_subset(result_table, min_misfit, near_optimal_ratio)
    subset_table = near_optimal["subset_table"]
    subset_mt = _result_table_to_mt(subset_table)

    if subset_mt.shape[0] == 0:
        max_pairwise_omega_deg = np.nan
        grade = "C"
    else:
        omega_pairwise = _pairwise_omega_matrix(subset_mt)
        max_pairwise_omega_deg = float(np.nanmax(omega_pairwise))
        if max_pairwise_omega_deg <= max_angle_deg_a:
            grade = "A"
        elif max_pairwise_omega_deg <= max_angle_deg_b:
            grade = "B"
        else:
            grade = "C"

    return {
        "near_optimal_ratio": near_optimal_ratio,
        "near_optimal_threshold": float(near_optimal["threshold"]),
        "near_optimal_count": int(near_optimal["subset_size"]),
        "max_pairwise_omega_deg": max_pairwise_omega_deg,
        "consistency_grade": grade,
    }


def grade_coverage(coverage):
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


def grade_count(counts):
    min_active_count = int(counts["min_active_count"])

    if min_active_count >= DEFAULT_QUALITY_CONFIG["count"]["A"]["min_active_each"]:
        return "A"
    if min_active_count >= DEFAULT_QUALITY_CONFIG["count"]["B"]["min_active_each"]:
        return "B"
    return "C"


def grade_fit(fit):
    misfit_mean_active = float(fit["misfit_mean_active"])
    if misfit_mean_active <= DEFAULT_QUALITY_CONFIG["fit"]["A"]["max_mean_active"]:
        return "A"
    if misfit_mean_active <= DEFAULT_QUALITY_CONFIG["fit"]["B"]["max_mean_active"]:
        return "B"
    return "C"


def grade_overall(coverage_grade, count_grade, fit_grade, sharpness_grade, consistency_grade):
    if (
        coverage_grade == "A"
        and sharpness_grade == "A"
        and consistency_grade == "A"
        and count_grade != "C"
        and fit_grade != "C"
    ):
        return "A"
    if (
        coverage_grade != "C"
        and sharpness_grade != "C"
        and consistency_grade != "C"
    ):
        return "B"
    return "C"


def build_quality_flags(coverage_grade, count_grade, fit_grade, sharpness_grade, consistency_grade):
    flags = []
    if coverage_grade == "C":
        flags.append("low_coverage")
    if count_grade == "C":
        flags.append("low_observation_count")
    if fit_grade == "C":
        flags.append("high_misfit")
    if sharpness_grade == "C":
        flags.append("low_sharpness")
    if consistency_grade == "C":
        flags.append("low_consistency")
    return flags


def evaluate_solution_quality(
    event,
    weight_list,
    best_misfit_row,
    *,
    grid=None,
    misfit_all=None,
    result_table=None,
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
        Nested dictionary containing sub-grades, overall grade, raw metrics,
        stability diagnostics, and the internally constructed result table.
    """
    config = DEFAULT_QUALITY_CONFIG
    sharpness_config = config["sharpness"]
    consistency_config = config["consistency"]

    best_misfit = _as_misfit_row(best_misfit_row)
    all_result_table = _build_result_table(grid=grid, misfit_all=misfit_all, result_table=result_table)

    counts = compute_quality_counts(event, weight_list)
    coverage = compute_quality_coverage(event, weight_list)
    fit = compute_quality_fit(event, weight_list, best_misfit)
    sharpness = evaluate_sharpness(
        all_result_table,
        low_error_count=sharpness_config["low_error_count"],
        min_relative_span_a=sharpness_config["A"]["min_relative_span"],
        min_relative_span_b=sharpness_config["B"]["min_relative_span"],
    )
    consistency = evaluate_consistency(
        all_result_table,
        fit["misfit_total"],
        near_optimal_ratio=consistency_config["near_optimal_ratio"],
        max_angle_deg_a=consistency_config["A"]["max_angle_deg"],
        max_angle_deg_b=consistency_config["B"]["max_angle_deg"],
    )

    coverage_grade = grade_coverage(coverage)
    count_grade = grade_count(counts)
    fit_grade = grade_fit(fit)
    sharpness_grade = sharpness["sharpness_grade"]
    consistency_grade = consistency["consistency_grade"]
    overall_grade = grade_overall(
        coverage_grade,
        count_grade,
        fit_grade,
        sharpness_grade,
        consistency_grade,
    )

    flags = build_quality_flags(
        coverage_grade,
        count_grade,
        fit_grade,
        sharpness_grade,
        consistency_grade,
    )

    return {
        "overall_grade": overall_grade,
        "coverage_grade": coverage_grade,
        "count_grade": count_grade,
        "fit_grade": fit_grade,
        "sharpness_grade": sharpness_grade,
        "consistency_grade": consistency_grade,
        "coverage": coverage,
        "counts": counts,
        "fit": fit,
        "stability": {
            "sharpness": sharpness,
            "consistency": consistency,
        },
        "flags": flags,
    }


def evaluate_quality_from_grid(event, weight_list, best_misfit_row, grid, misfit_all):
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
    )


def evaluate_quality_from_table(event, weight_list, best_misfit_row, result_table):
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
    )
