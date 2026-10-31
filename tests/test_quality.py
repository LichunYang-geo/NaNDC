import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "nandc"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "nandc" / "util"))

from nandc_log import build_quality_log_section
from nandc_mt import get_full_MT_array, tt_to_mij
from nandc_quality import _omega_to_reference_deg, evaluate_quality_from_table


class DummyEvent:
    """Minimal event object exposing the fields used by quality evaluation."""

    def __init__(self):
        self.stations = ["A", "B", "C"]
        self.pol = np.ones(3)
        self.amp = np.ones(3)
        self.spr = np.ones(3)
        self.pol_effective_max_azi_gap = 80.0
        self.pol_effective_max_takeoff_gap = 20.0
        self.amp_effective_max_azi_gap = 85.0
        self.amp_effective_max_takeoff_gap = 25.0
        self.spr_effective_max_azi_gap = 70.0
        self.spr_effective_max_takeoff_gap = 15.0


def _compact_result_table():
    """Build a small table whose accepted models form a tight mechanism cluster."""
    n_model = 100
    result_table = np.zeros((n_model, 10), dtype=np.float64)
    result_table[:, 0] = 1.0
    result_table[:, 3] = np.linspace(0.0, 1.0, n_model)
    result_table[:, 5] = 0.5
    result_table[:, 6] = np.linspace(1.20, 3.0, n_model)
    result_table[0, 6] = 1.00
    result_table[1, 6] = 1.04
    result_table[:, 7] = 0.10
    result_table[:, 8] = 0.60
    result_table[:, 9] = 0.60
    return result_table


def _flat_low_error_result_table():
    """Build a table where rank sharpness is poor despite many candidates."""
    n_model = 10000
    result_table = np.zeros((n_model, 10), dtype=np.float64)
    result_table[:, 0] = 1.0
    result_table[:, 5] = 0.5
    result_table[:, 6] = 2.0
    result_table[:200, 6] = np.linspace(0.10, 0.104, 200)
    result_table[:, 7:10] = 0.10
    return result_table


def _legacy_pairwise_omega_matrix(mts):
    """Reproduce the original non-parallel pairwise omega implementation."""
    mts = np.asarray(mts, dtype=np.float64)
    full_mts = np.stack([get_full_MT_array(mt) for mt in mts], axis=0)
    norms = np.linalg.norm(full_mts.reshape(mts.shape[0], -1), axis=1)
    dot = np.tensordot(full_mts, full_mts, axes=([1, 2], [1, 2]))
    denom = norms[:, None] * norms[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosom = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)
    omega = np.rad2deg(np.arccos(np.clip(cosom, -1.0, 1.0)))
    np.fill_diagonal(omega, 0.0)
    return omega


def _legacy_blocked_max_pairwise_omega(mts, block_size=2):
    """Reproduce the original blocked max-pairwise omega calculation."""
    mts = np.asarray(mts, dtype=np.float64)
    full_mts = np.stack([get_full_MT_array(mt) for mt in mts], axis=0)
    flat_mts = full_mts.reshape(mts.shape[0], -1)
    norms = np.linalg.norm(flat_mts, axis=1)
    valid_norm = norms > 0.0
    unit_flat = np.divide(
        flat_mts,
        norms[:, None],
        out=np.zeros_like(flat_mts),
        where=valid_norm[:, None],
    )

    row_min_cos = []
    n_blocks = int(np.ceil(mts.shape[0] / block_size))
    for block_index in range(n_blocks):
        i0 = block_index * block_size
        i1 = min(i0 + block_size, mts.shape[0])
        this_row_min_cos = 1.0
        for j0 in range(i0, mts.shape[0], block_size):
            j1 = min(j0 + block_size, mts.shape[0])
            cosom = unit_flat[i0:i1] @ unit_flat[j0:j1].T
            valid_i = valid_norm[i0:i1]
            valid_j = valid_norm[j0:j1]
            if not (np.all(valid_i) and np.all(valid_j)):
                valid_pair = valid_i[:, None] & valid_j[None, :]
                np.putmask(cosom, ~valid_pair, 1.0)
            np.clip(cosom, -1.0, 1.0, out=cosom)
            if i0 == j0:
                np.fill_diagonal(cosom, 1.0)
            this_row_min_cos = min(this_row_min_cos, float(np.nanmin(cosom)))
        row_min_cos.append(this_row_min_cos)

    min_cos = float(np.nanmin(row_min_cos))
    return float(np.rad2deg(np.arccos(np.clip(min_cos, -1.0, 1.0))))


class QualityEvaluationTests(unittest.TestCase):
    """Behavior tests for the catalog-oriented solution quality score."""

    def test_solution_quality_uses_total_misfit_and_compact_stability(self):
        """High component misfits do not lower fit grade when total misfit is low."""
        result_table = _compact_result_table()
        result_table[0, 6] = 0.10
        result_table[1, 6] = 0.104
        result_table[:, 7:10] = 0.60
        best_misfit = result_table[0, 6:10]

        quality = evaluate_quality_from_table(
            DummyEvent(),
            weight_list=[10.0, 1.0, 1.0],
            best_misfit_row=best_misfit,
            result_table=result_table,
        )

        self.assertEqual(quality["solution_quality"], "A")
        self.assertEqual(quality["overall_grade"], "A")
        self.assertEqual(quality["fit_grade"], "A")
        self.assertEqual(quality["stability_grade"], "A")
        self.assertEqual(quality["coverage_grade"], "A")
        self.assertEqual(quality["fit"]["misfit_total"], 0.10)
        self.assertEqual(quality["stability"]["accepted_count"], 2)
        self.assertEqual(quality["stability"]["misfit_sharpness_grade"], "A")
        self.assertLessEqual(quality["stability"]["omega_p95_deg"], 25.0)

        log_text = build_quality_log_section(quality)
        self.assertIn("quality.solution_quality = A", log_text)
        self.assertIn("quality.stability.misfit_rank_ratio", log_text)
        self.assertNotIn("quality.count_grade", log_text)
        self.assertNotIn("quality.fit.misfit_weighted_active", log_text)

    def test_stability_uses_rank_sharpness_for_flat_low_error_models(self):
        """A flat low-error rank lowers stability even with many candidates."""
        result_table = _flat_low_error_result_table()
        best_misfit = result_table[0, 6:10]

        quality = evaluate_quality_from_table(
            DummyEvent(),
            weight_list=[1.0, 1.0, 1.0],
            best_misfit_row=best_misfit,
            result_table=result_table,
        )

        self.assertEqual(quality["fit_grade"], "A")
        self.assertEqual(quality["coverage_grade"], "A")
        self.assertEqual(quality["stability"]["accepted_count"], 200)
        self.assertAlmostEqual(quality["stability"]["misfit_rank_ratio"], 0.04)
        self.assertEqual(quality["stability"]["misfit_sharpness_grade"], "C")
        self.assertEqual(quality["stability_grade"], "C")
        self.assertEqual(quality["solution_quality"], "C")

        log_text = build_quality_log_section(quality)
        self.assertIn("quality.stability.misfit_sharpness_grade = C", log_text)
        self.assertIn("quality.stability.misfit_rank_ratio = 0.040000", log_text)

    def test_omega_to_reference_matches_legacy_pairwise_column(self):
        """Current omega-to-best uses the same angle formula as the old matrix code."""
        mts = tt_to_mij(
            np.ones(5),
            np.array([0.0, 0.05, -0.08, 0.12, -0.18]),
            np.array([0.0, 0.1, -0.12, 0.2, -0.25]),
            np.array([0.0, 20.0, 60.0, 120.0, 240.0]),
            np.array([0.0, 15.0, -30.0, 45.0, -60.0]),
            np.array([0.5, 0.7, 0.8, 0.6, 0.9]),
        )

        legacy_omega = _legacy_pairwise_omega_matrix(mts)
        omega_to_best = _omega_to_reference_deg(mts, mts[0])
        legacy_blocked_max = _legacy_blocked_max_pairwise_omega(mts, block_size=2)

        np.testing.assert_allclose(omega_to_best, legacy_omega[:, 0], atol=1e-12)
        self.assertAlmostEqual(legacy_blocked_max, float(np.nanmax(legacy_omega)), places=12)


if __name__ == "__main__":
    unittest.main()
