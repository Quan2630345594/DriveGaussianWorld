"""One frozen CPU Gaussian-state/view geometry contract, no fitting or rendering."""

from copy import deepcopy
import json
import resource
import sys
import time
import unittest

import numpy as np

from gaussian_park_world.world_model import gaussian_state_contract as contract
from gaussian_park_world.world_model.geometry import invert_transform, project_points
from midpoint_oracle import oracle_rollout as synthetic_bicycle_ORACLE
from gaussian_park_world.world_model.geometry import CAMERA_IDS, yaw_transform


PIXEL_ATOL = 1e-9
JACOBIAN_ATOL = 1e-6
COVARIANCE_ATOL = 1e-10
TRANSFORM_ATOL = 1e-12
FD_STEP_M = 1e-6
MAX_TEST_SECONDS = 10.
MAX_PEAK_RSS_BYTES = 256 * 1024**2


class GaussianContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.make_static_fixture()
        cls.metrics = {}
        cls.rejected_cases = []

    def identity_camera(self):
        return contract.current_camera_pose(np.eye(4), np.eye(4))

    def one_gaussian(self, center=(1., .5, 5.)):
        return dict(provenance=contract.PROVENANCE, ids=np.array([1], dtype=np.int64),
                    centers_world_m=np.array([center], dtype=np.float64),
                    covariances_world_m2=np.array([np.diag([.04, .09, .16])]),
                    opacities=np.array([.5]), colors_rgb=np.array([[.2, .4, .6]]))

    def project(self, gaussians, camera=None, K=None):
        f = self.fixture
        return contract.project_gaussians(gaussians, self.identity_camera() if camera is None else camera,
                                           f["K"] if K is None else K, f["image_size"])

    def reject(self, label, callback):
        with self.subTest(case=label):
            with self.assertRaises((ValueError, TypeError)):
                callback()
            self.rejected_cases.append(label)

    def assert_same(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if isinstance(actual[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                self.assertEqual(actual[key], expected[key])

    def test_analytic_projection_jacobian_and_screen_covariance(self):
        projected = self.project(self.one_gaussian())
        self.assertTrue(projected["valid_mask"][0])
        uv_error = float(np.abs(projected["means_uv_px"][0] - [80., 56.]).max())
        expected_J = np.array([[16., 0., -3.2], [0., 16., -1.6]])
        J_error = float(np.abs(projected["jacobians_uv_per_m"][0] - expected_J).max())
        expected_covariance = np.array([[11.8784, .8192], [.8192, 23.4496]])
        covariance_error = float(np.abs(projected["covariances_uv_px2"][0] - expected_covariance).max())
        self.assertLess(uv_error, PIXEL_ATOL)
        self.assertLess(J_error, TRANSFORM_ATOL)
        self.assertLess(covariance_error, COVARIANCE_ATOL)
        self.metrics.update(analytic_projection_max_px=uv_error, analytic_jacobian_max_px_per_m=J_error,
                            analytic_covariance_max_px2=covariance_error)

    def test_jacobian_finite_difference(self):
        points = np.array([[1., .5, 5.], [.2, -.3, 4.], [-.4, .1, 7.]])
        f = self.fixture
        J = contract.pinhole_jacobian(points, f["K"], f["image_size"])
        numerical = np.empty_like(J)
        for axis in range(3):
            plus, minus = points.copy(), points.copy()
            plus[:, axis] += FD_STEP_M
            minus[:, axis] -= FD_STEP_M
            uv_plus, _, valid_plus = project_points(plus, f["K"], f["image_size"])
            uv_minus, _, valid_minus = project_points(minus, f["K"], f["image_size"])
            self.assertTrue(valid_plus.all() and valid_minus.all())
            numerical[:, :, axis] = (uv_plus - uv_minus) / (2 * FD_STEP_M)
        error = float(np.abs(J - numerical).max())
        self.assertTrue(np.isfinite(error))
        self.assertLess(error, JACOBIAN_ATOL)
        self.metrics["finite_difference_jacobian_max_px_per_m"] = error

    def test_coordinate_composition_and_frame_gauge_consistency(self):
        f = self.fixture
        observation = contract.static_multiview_observation(f["gaussians"], f["T_world_from_vehicle"], f["rig"], f["K"], f["image_size"])
        errors, symmetry, minimum_eigenvalue = [], [], []
        for v, view in enumerate(CAMERA_IDS):
            direct_uv, _, direct_valid = project_points(f["optical_targets"][view], f["K"], f["image_size"])
            self.assertTrue(direct_valid.all())
            rows = slice(4 * v, 4 * v + 4)
            self.assertTrue(observation[view]["valid_mask"][rows].all())
            errors.append(np.abs(observation[view]["means_uv_px"][rows] - direct_uv).max())
            covariance = observation[view]["covariances_uv_px2"][observation[view]["valid_mask"]]
            symmetry.append(np.abs(covariance - covariance.transpose(0, 2, 1)).max())
            minimum_eigenvalue.append(np.linalg.eigvalsh(covariance).min())
        H = yaw_transform(.29, [-.5, .3, .2])
        transformed = contract.transform_world_subset(f["gaussians"], f["gaussians"]["ids"], H)
        changed = contract.static_multiview_observation(transformed, H @ f["T_world_from_vehicle"], f["rig"], f["K"], f["image_size"])
        covariance_errors = []
        for view in CAMERA_IDS:
            np.testing.assert_array_equal(observation[view]["valid_mask"], changed[view]["valid_mask"])
            mask = observation[view]["valid_mask"]
            errors.append(np.abs(observation[view]["means_uv_px"][mask] - changed[view]["means_uv_px"][mask]).max())
            covariance_errors.append(np.abs(observation[view]["covariances_uv_px2"][mask] - changed[view]["covariances_uv_px2"][mask]).max())
        self.assertLess(max(errors), PIXEL_ATOL)
        self.assertLess(max(covariance_errors), COVARIANCE_ATOL)
        self.assertLess(max(symmetry), TRANSFORM_ATOL)
        self.assertGreater(min(minimum_eigenvalue), 0.)
        self.metrics.update(coordinate_and_gauge_projection_max_px=float(max(errors)),
                            gauge_covariance_max_px2=float(max(covariance_errors)),
                            screen_covariance_symmetry_max_px2=float(max(symmetry)),
                            current_screen_covariance_min_eigenvalue_px2=float(min(minimum_eigenvalue)))

    def test_static_world_action_changes_view_not_world(self):
        f = self.fixture
        before = deepcopy(f["gaussians"])
        dt = np.array([.07, .13, .11, .17, .09, .16, .08, .14, .12, .1])
        candidates = {"forward": np.tile([1., 0.], (10, 1)), "left": np.tile([1., .3], (10, 1))}
        initial = contract.static_multiview_observation(f["gaussians"], f["T_world_from_vehicle"], f["rig"], f["K"], f["image_size"])
        final, mask_changes, cov_eigenvalues = {}, 0, []
        for name, actions in candidates.items():
            future_hypothetical = synthetic_bicycle_ORACLE(f["initial_state"], actions, dt, 2.7)
            for state in future_hypothetical:
                T_world_from_vehicle = yaw_transform(state[2], [state[0], state[1], 0.])
                observation = contract.static_multiview_observation(f["gaussians"], T_world_from_vehicle, f["rig"], f["K"], f["image_size"])
                for view in CAMERA_IDS:
                    mask = observation[view]["valid_mask"]
                    self.assertGreater(mask.sum(), 0)
                    mask_changes += int(np.count_nonzero(mask != initial[view]["valid_mask"]))
                    cov_eigenvalues.extend(np.linalg.eigvalsh(observation[view]["covariances_uv_px2"][mask]).reshape(-1))
                self.assert_same(f["gaussians"], before)
            final[name] = observation
        both = final["forward"]["front"]["valid_mask"] & final["left"]["front"]["valid_mask"]
        self.assertGreater(both.sum(), 0)
        difference = float(np.linalg.norm(final["forward"]["front"]["means_uv_px"][both]
                                         - final["left"]["front"]["means_uv_px"][both], axis=1).max())
        self.assertGreater(difference, 1e-3)
        self.assertGreater(mask_changes, 0)
        self.assertGreater(min(cov_eigenvalues), 0.)
        self.metrics.update(candidate_common_front_centers=int(both.sum()), candidate_view_max_difference_px=difference,
                            center_mask_changes_across_candidate_observations=mask_changes,
                            candidate_screen_covariance_min_eigenvalue_px2=float(min(cov_eigenvalues)),
                            static_center_max_change_m=float(np.abs(f["gaussians"]["centers_world_m"] - before["centers_world_m"]).max()),
                            static_covariance_max_change_m2=float(np.abs(f["gaussians"]["covariances_world_m2"] - before["covariances_world_m2"]).max()),
                            oracle_candidate_count=2, oracle_steps_each=10)

    def test_external_subset_motion_is_world_state_change(self):
        f = self.fixture
        before = deepcopy(f["gaussians"])
        selected = np.array([7000, 7002, 7009], dtype=np.int64)
        H = yaw_transform(.35, [.3, -.2, .15])
        moved = contract.transform_world_subset(f["gaussians"], selected, H)
        mask = np.isin(before["ids"], selected)
        expected_centers = before["centers_world_m"][mask] @ H[:3, :3].T + H[:3, 3]
        expected_covariance = np.stack([H[:3, :3] @ C @ H[:3, :3].T for C in before["covariances_world_m2"][mask]])
        center_error = float(np.abs(moved["centers_world_m"][mask] - expected_centers).max())
        covariance_error = float(np.abs(moved["covariances_world_m2"][mask] - expected_covariance).max())
        self.assertLess(center_error, TRANSFORM_ATOL)
        self.assertLess(covariance_error, TRANSFORM_ATOL)
        self.assertTrue((np.linalg.norm(moved["centers_world_m"][mask] - before["centers_world_m"][mask], axis=1) > 1e-3).all())
        self.assertTrue((np.linalg.norm(moved["covariances_world_m2"][mask] - before["covariances_world_m2"][mask], axis=(1, 2)) > 1e-6).all())
        for key in ("centers_world_m", "covariances_world_m2"):
            np.testing.assert_array_equal(moved[key][~mask], before[key][~mask])
        for key in ("ids", "opacities", "colors_rgb"):
            np.testing.assert_array_equal(moved[key], before[key])
        self.assert_same(f["gaussians"], before)
        self.metrics.update(external_motion_changed_count=int(mask.sum()), external_motion_center_max_m=center_error,
                            external_motion_covariance_max_m2=covariance_error,
                            unaffected_center_max_change_m=float(np.abs(moved["centers_world_m"][~mask] - before["centers_world_m"][~mask]).max()),
                            unaffected_covariance_max_change_m2=float(np.abs(moved["covariances_world_m2"][~mask] - before["covariances_world_m2"][~mask]).max()))

    def test_permutation_invariance_by_fixture_id(self):
        f = self.fixture
        camera = contract.current_camera_pose(f["T_world_from_vehicle"], f["rig"]["front"])
        original = self.project(f["gaussians"], camera)
        order = np.random.default_rng(contract.PERMUTATION_SEED).permutation(16)
        shuffled = {k: v[order].copy() if isinstance(v, np.ndarray) else v for k, v in f["gaussians"].items()}
        projected = self.project(shuffled, camera)
        back = np.argsort(projected["ids"])
        errors = []
        for key in original:
            np.testing.assert_allclose(original[key], projected[key][back], atol=TRANSFORM_ATOL, rtol=0, equal_nan=True)
            if original[key].dtype.kind == "f" and key in ("means_uv_px", "covariances_uv_px2"):
                finite = np.isfinite(original[key])
                errors.append(float(np.abs(original[key][finite] - projected[key][back][finite]).max()))
        self.assertLess(max(errors), TRANSFORM_ATOL)
        self.metrics["permutation_aligned_max_error"] = max(errors)

    def test_invalid_geometry_and_depth_are_nan_masked(self):
        cases = []
        for center in ((0., 0., -1.), (0., 0., 0.), (0., 0., np.nan), (np.inf, 0., 1.),
                       (4., 0., 5.), (0., 0., 1e-300)):
            cases.append((f"center_{center}", self.one_gaussian(center)))
        for label, covariance in (("non_spd", np.diag([1., -1., 1.])), ("singular", np.diag([1., 0., 1.])),
                                  ("asymmetric", np.array([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]])),
                                  ("nan_cov", np.full((3, 3), np.nan)), ("inf_cov", np.full((3, 3), np.inf))):
            bad = self.one_gaussian()
            bad["covariances_world_m2"][0] = covariance
            cases.append((label, bad))
        for label, bad in cases:
            with self.subTest(masked=label):
                projected = self.project(bad)
                self.assertFalse(projected["valid_mask"][0])
                for key in ("means_uv_px", "covariances_uv_px2", "jacobians_uv_per_m", "depth_z_m"):
                    self.assertTrue(np.isnan(projected[key]).all())
        left_edge = self.project(self.one_gaussian((-4., 0., 5.)))
        self.assertTrue(left_edge["valid_mask"][0])
        mixed = deepcopy(self.fixture["gaussians"])
        mixed["covariances_world_m2"][0] = np.diag([1., -1., 1.])
        camera = contract.current_camera_pose(self.fixture["T_world_from_vehicle"], self.fixture["rig"]["front"])
        mixed_result = self.project(mixed, camera)
        self.assertFalse(mixed_result["valid_mask"][0])
        self.assertTrue(mixed_result["valid_mask"][1:4].all())
        self.metrics["invalid_geometry_masked_cases"] = len(cases)

    def test_invalid_schema_attributes_camera_and_intrinsics_rejected(self):
        for field, value in (("opacities", np.array([-1.])), ("opacities", np.array([1.1])),
                             ("opacities", np.array([np.nan])), ("colors_rgb", np.array([[1.1, 0., 0.]])),
                             ("colors_rgb", np.array([[np.inf, 0., 0.]])),
                             ("centers_world_m", np.zeros((1, 4))), ("covariances_world_m2", np.zeros((1, 3))),
                             ("centers_world_m", np.zeros((1, 3), dtype=np.float32))):
            bad = self.one_gaussian()
            bad[field] = value
            self.reject(f"invalid_{field}_{len(self.rejected_cases)}", lambda: self.project(bad))
        bad = deepcopy(self.fixture["gaussians"])
        bad["ids"][1] = bad["ids"][0]
        self.reject("duplicate_ID", lambda: self.project(bad))
        reflection = np.diag([1., -1., 1., 1.])
        self.reject("reflection_as_rotation", lambda: contract.current_camera_pose(reflection, np.eye(4)))
        self.reject("malformed_transform", lambda: contract.current_camera_pose(np.eye(3), np.eye(4)))
        for value in (np.nan, np.inf):
            bad_T = np.eye(4)
            bad_T[0, 3] = value
            self.reject(f"nonfinite_camera_{value}", lambda: contract.current_camera_pose(bad_T, np.eye(4)))
        camera = contract.current_camera_pose(self.fixture["T_world_from_vehicle"], self.fixture["rig"]["front"])
        wrong = deepcopy(camera)
        wrong["T_world_from_optical"] = invert_transform(camera["T_world_from_optical"])
        self.reject("inverse_matrix_wrong_direction", lambda: self.project(self.one_gaussian(), wrong))
        wrong = deepcopy(camera)
        wrong["T_optical_from_world"] = wrong.pop("T_world_from_optical")
        self.reject("wrong_direction_name", lambda: self.project(self.one_gaussian(), wrong))
        for row, col, value in ((0, 0, 0.), (1, 1, -1.), (2, 2, 2.), (0, 1, .5), (0, 0, np.nan), (0, 0, np.inf)):
            K = self.fixture["K"].copy()
            K[row, col] = value
            self.reject(f"bad_K_{row}_{col}_{value}", lambda: self.project(self.one_gaussian(), K=K))
        for size in ([0, 96], [128., 96.]):
            self.reject(f"bad_image_size_{size}", lambda: contract.project_gaussians(self.one_gaussian(), self.identity_camera(), self.fixture["K"], size))
        self.reject("missing_object_ID", lambda: contract.transform_world_subset(self.fixture["gaussians"], np.array([9999], dtype=np.int64), np.eye(4)))

    def test_future_fields_and_trajectories_rejected(self):
        for field in ("future_target", "future_gaussians", "full_episode", "future_camera_poses"):
            bad = self.one_gaussian()
            bad[field] = np.zeros((10, 4, 4))
            self.reject(f"Gaussian_{field}", lambda: self.project(bad))
        camera = self.identity_camera()
        camera["future_camera_poses"] = np.zeros((10, 4, 4))
        self.reject("camera_future_poses", lambda: self.project(self.one_gaussian(), camera))
        f = self.fixture
        self.reject("entire_trajectory_as_current", lambda: contract.static_multiview_observation(
            f["gaussians"], np.tile(np.eye(4), (10, 1, 1)), f["rig"], f["K"], f["image_size"]))
        self.reject("future_keyword", lambda: contract.project_gaussians(
            f["gaussians"], self.identity_camera(), f["K"], f["image_size"], future_target=np.zeros((10, 3))))

    def test_inputs_immutable_and_opacity_is_not_visibility(self):
        f = self.fixture
        g, rig, vehicle, K, size = deepcopy(f["gaussians"]), deepcopy(f["rig"]), f["T_world_from_vehicle"].copy(), f["K"].copy(), f["image_size"].copy()
        before = deepcopy((g, rig, vehicle, K, size))
        observation = contract.static_multiview_observation(g, vehicle, rig, K, size)
        self.assert_same(g, before[0])
        self.assert_same(rig, before[1])
        for actual, expected in zip((vehicle, K, size), before[2:]):
            np.testing.assert_array_equal(actual, expected)
        observation["front"]["colors_rgb"][:] = 0.
        self.assert_same(g, before[0])
        transparent = self.one_gaussian()
        transparent["opacities"][:] = 0.
        self.assertTrue(self.project(transparent)["valid_mask"][0])
        self.assertFalse(any(name in sys.modules for name in ("torch", "scipy", "gsplat", "carla")))


if __name__ == "__main__":
    start = time.perf_counter()
    result = unittest.main(verbosity=2, exit=False).result
    elapsed = time.perf_counter() - start
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    budget_ok = elapsed < MAX_TEST_SECONDS and peak_rss_bytes < MAX_PEAK_RSS_BYTES
    status = "BLOCKED" if not result.wasSuccessful() else ("PASS" if budget_ok else "NOT_CONFIRMED")
    print("CONTRACT_RESULTS " + json.dumps(dict(status=status, seed=contract.SEED,
          permutation_seed=contract.PERMUTATION_SEED, gaussian_count=16, metrics=GaussianContractTests.metrics,
          rejected_cases=GaussianContractTests.rejected_cases, rejected_count=len(GaussianContractTests.rejected_cases),
          pixel_atol=PIXEL_ATOL, jacobian_atol=JACOBIAN_ATOL, covariance_atol=COVARIANCE_ATOL,
          transform_atol=TRANSFORM_ATOL, fd_step_m=FD_STEP_M, real_data_gate="BLOCKED_MISSING_EVIDENCE"), allow_nan=False))
    print("TEST_SUMMARY " + json.dumps(dict(status=status, tests=result.testsRun, failures=len(result.failures),
          errors=len(result.errors), skipped=len(result.skipped), elapsed_seconds=elapsed, peak_rss_bytes=peak_rss_bytes,
          max_test_seconds=MAX_TEST_SECONDS, max_peak_rss_bytes=MAX_PEAK_RSS_BYTES,
          python=sys.version.split()[0], numpy=np.__version__, cpu_only=True)))
    sys.exit(0 if result.wasSuccessful() and budget_ok else 1)
