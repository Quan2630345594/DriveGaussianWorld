"""CPU-only trajectory geometry tests, without model or analysis execution."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("analyze_parkinge2e", Path(__file__).parents[1] / "scripts/analyze_parkinge2e.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class TrajectoryMetricsTests(unittest.TestCase):
    def test_same_polyline_different_sampling(self):
        p = [[0, 0, 0], [2, 0, 0], [2, 2, np.pi/2]]
        r = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, np.pi/4], [2, 2, np.pi/2]]
        m = analysis.compare_paths(p, r, 33)
        self.assertAlmostEqual(m["position_m"]["rmse"], 0)
        self.assertAlmostEqual(m["discrete_frechet_m"], 0)
        self.assertAlmostEqual(m["yaw_rad"]["rmse"], 0)
        self.assertIn("not_time", m["label"])

    def test_translation_is_not_fitted_away(self):
        p = np.array([[0, 0, 0], [2, 0, 0]])
        shifted = p + [0, 3, 0]
        m = analysis.compare_paths(shifted, p, 17)
        for key in ("rmse", "mean", "p95"):
            self.assertAlmostEqual(m["position_m"][key], 3)
        self.assertAlmostEqual(m["hausdorff_m"], 3)
        self.assertAlmostEqual(m["discrete_frechet_m"], 3)
        self.assertAlmostEqual(m["endpoint_position_error_m"], 3)

    def test_reversed_orientation_retained(self):
        p = np.array([[0, 0, 0], [5, 0, 0]])
        m = analysis.compare_paths(p[::-1], p, 21)
        self.assertAlmostEqual(m["hausdorff_m"], 0)
        self.assertAlmostEqual(m["discrete_frechet_m"], 5)
        self.assertAlmostEqual(m["endpoint_position_error_m"], 5)
        self.assertGreater(m["position_m"]["rmse"], 2)

    def test_yaw_unwrap_and_wrap(self):
        p = [[0, 0, np.deg2rad(179)], [2, 0, np.deg2rad(-179)]]
        r = [[0, 0, np.deg2rad(179)], [2, 0, np.deg2rad(181)]]
        self.assertAlmostEqual(analysis.compare_paths(p, r, 11)["yaw_rad"]["rmse"], 0)
        middle = analysis.resample_path(p, 3)[1, 2]
        self.assertAlmostEqual(abs(middle), np.pi)

    def test_duplicates_cusp_and_backtracking_preserved(self):
        p = [[0, 0, 0], [1, 0, 0], [1, 0, np.pi], [0, 0, np.pi]]
        q = analysis.resample_path(p, 5)
        np.testing.assert_allclose(q[:, 0], [0, .5, 1, .5, 0])
        self.assertAlmostEqual(q[2, 2], np.pi)
        self.assertTrue(np.isfinite(q).all())
        self.assertAlmostEqual(analysis.compare_paths(p, p, 9)["position_m"]["rmse"], 0)

    def test_stationary_and_single_vertex(self):
        p = [[2, 3, 0], [2, 3, np.pi/2]]
        q = analysis.resample_path(p, 3)
        np.testing.assert_allclose(q[:, 2], [0, np.pi/4, np.pi/2])
        m = analysis.compare_paths(p, [[2, 3, 0]], 3)
        self.assertEqual(m["position_m"]["rmse"], 0)
        self.assertAlmostEqual(m["endpoint_yaw_error_rad"], np.pi/2)

    def test_tracking_projects_to_segment_not_vertex(self):
        p = [[0, 0, 0], [10, 0, 0]]
        e = [[5, 2, 0], [8, 2, np.pi/2]]
        m = analysis.tracking_errors(e, p)
        self.assertAlmostEqual(m["position_m"]["rmse"], 2)
        self.assertAlmostEqual(m["endpoint_position_error_m"], np.sqrt(8))
        self.assertAlmostEqual(m["endpoint_yaw_error_rad"], np.pi/2)

    def test_tracking_degenerate_and_duplicate_segments(self):
        e = [[1, 1, 0]]
        for p in ([[0, 0, 0]], [[0, 0, 0], [0, 0, 0]]):
            self.assertAlmostEqual(analysis.tracking_errors(e, p)["position_m"]["mean"], np.sqrt(2))

    def test_missing_paths_not_fabricated(self):
        for p in (None, {}, {"path": []}):
            self.assertIsNone(analysis._case_path(p, "path"))

    def test_invalid_inputs(self):
        for path in ([], [[0, 1]], [[0, 1, np.nan]], [[0, 1, np.inf]]):
            with self.assertRaises(ValueError):
                analysis.compare_paths(path, [[0, 0, 0]])
        with self.assertRaises(ValueError):
            analysis.resample_path([[0, 0, 0]], 1)


if __name__ == "__main__":
    unittest.main()
