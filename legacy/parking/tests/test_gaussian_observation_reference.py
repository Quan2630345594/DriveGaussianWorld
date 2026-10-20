"""One fixed CPU reference-render test run with independent analytic targets."""

import time
PROCESS_START = time.perf_counter()

from copy import deepcopy
import json
import math
import resource
import sys
import unittest
from unittest.mock import patch

import numpy as np

from gaussian_park_world.world_model import gaussian_observation_reference as reference
from gaussian_park_world.world_model import gaussian_state_contract as state_contract
from midpoint_oracle import oracle_rollout as synthetic_bicycle_ORACLE
from gaussian_park_world.world_model.geometry import CAMERA_IDS, yaw_transform


ANALYTIC_ATOL = 1e-10
PERMUTATION_ATOL = 1e-12
RANGE_ATOL = 1e-12
MOTION_RGB_MIN = 1e-6
PERMUTATION_SEED = 808002
MAX_WALL_SECONDS = 10.
MAX_RSS_BYTES = 256 * 1024**2


class ObservationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = state_contract.make_static_fixture()
        cls.metrics = {}
        cls.rejected = []

    def scene(self, depths=(2.,), ids=None, opacities=None, colors=None):
        n = len(depths)
        return dict(provenance=state_contract.PROVENANCE,
                    ids=np.arange(1, n + 1, dtype=np.int64) if ids is None else np.array(ids, dtype=np.int64),
                    centers_world_m=np.column_stack((np.zeros((n, 2)), depths)).astype(np.float64),
                    covariances_world_m2=np.array([np.diag([z*z, z*z, .04]) for z in depths]),
                    opacities=np.full(n, .4) if opacities is None else np.array(opacities, dtype=np.float64),
                    colors_rgb=np.tile([.2, .5, .8], (n, 1)) if colors is None else np.array(colors, dtype=np.float64))

    def camera(self):
        return state_contract.current_camera_pose(np.eye(4), np.eye(4))

    def render_pixel(self, scene, background=(0., 0., 0.)):
        return reference.render_reference_observation(scene, self.camera(), np.eye(3), [1, 1], background)

    def reject(self, label, callback):
        with self.subTest(case=label):
            with self.assertRaises((ValueError, TypeError)):
                callback()
            self.rejected.append(label)

    def same(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if isinstance(actual[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                self.assertEqual(actual[key], expected[key])

    def finite_ranges(self, result):
        for key in ("rgb", "alpha", "transmittance"):
            value = result[key]
            self.assertEqual(value.dtype, np.float64)
            self.assertTrue(np.isfinite(value).all())
            self.assertGreaterEqual(value.min(), -RANGE_ATOL)
            self.assertLessEqual(value.max(), 1 + RANGE_ATOL)
        np.testing.assert_allclose(result["alpha"] + result["transmittance"], 1., atol=RANGE_ATOL, rtol=0)

    def test_one_splat_center_background_and_zero_opacity(self):
        errors = []
        for background, expected in (([0., 0., 0.], [.08, .2, .32]),
                                     ([1., 1., 1.], [.68, .8, .92]),
                                     ([.1, .2, .3], [.14, .32, .5])):
            result = self.render_pixel(self.scene(), background)
            self.finite_ranges(result)
            errors.extend([float(abs(result["alpha"][0, 0] - .4)),
                           float(np.abs(result["rgb"][0, 0] - expected).max())])
        zero = self.render_pixel(self.scene(opacities=[0.]), [.2, .4, .6])
        np.testing.assert_array_equal(zero["rgb"][0, 0], [.2, .4, .6])
        np.testing.assert_array_equal(zero["alpha"], [[0.]])
        np.testing.assert_array_equal(zero["transmittance"], [[1.]])
        self.assertLess(max(errors), ANALYTIC_ATOL)
        self.metrics["one_splat_max_rgb_alpha_error"] = max(errors)

    def test_two_splats_front_to_back_and_reversed_depth(self):
        scene = self.scene((2., 4.), ids=[20, 10], opacities=[.25, .5], colors=[[1., 0., 0.], [0., 0., 1.]])
        first = self.render_pixel(scene, [.2, .4, .6])
        # Hand calculation: .25*red + .75*.5*blue + .375*background.
        expected = np.array([.325, .15, .6])
        error = max(float(np.abs(first["rgb"][0, 0] - expected).max()), float(abs(first["alpha"][0, 0] - .625)))
        np.testing.assert_array_equal(first["sorted_ids"], [20, 10])
        reversed_scene = self.scene((4., 2.), ids=[20, 10], opacities=[.25, .5], colors=[[1., 0., 0.], [0., 0., 1.]])
        reverse = self.render_pixel(reversed_scene, [.2, .4, .6])
        # .5*blue + .5*.25*red + .375*background.
        reverse_error = max(float(np.abs(reverse["rgb"][0, 0] - [.2, .15, .725]).max()),
                            float(abs(reverse["alpha"][0, 0] - .625)))
        np.testing.assert_array_equal(reverse["sorted_ids"], [10, 20])
        self.assertGreater(np.abs(first["rgb"] - reverse["rgb"]).max(), .1)
        self.assertLess(max(error, reverse_error), ANALYTIC_ATOL)
        self.metrics.update(two_splat_max_rgb_alpha_error=max(error, reverse_error),
                            reversed_depth_max_rgb_change=float(np.abs(first["rgb"] - reverse["rgb"]).max()))

    def test_opaque_front_splat_blocks_back_reference(self):
        scene = self.scene((2., 4.), opacities=[1., .7], colors=[[1., 0., 0.], [0., 0., 1.]])
        result = self.render_pixel(scene, [1., 1., 1.])
        np.testing.assert_array_equal(result["rgb"], [[[1., 0., 0.]]])
        np.testing.assert_array_equal(result["transmittance"], [[0.]])
        np.testing.assert_array_equal(result["alpha"], [[1.]])

    def test_kernel_at_integer_pixel_isotropic_and_correlated(self):
        K, size = np.array([[10., 0., 3.], [0., 10., 2.], [0., 0., 1.]]), [7, 5]
        errors = []
        cases = ((np.diag([.25, .25, .04]), .6 * math.exp(-2.5)),
                 (np.array([[.5, .125, 0.], [.125, .25, 0.], [0., 0., .04]]), .6 * math.exp(-2.)))
        for covariance, expected_alpha in cases:
            scene = self.scene((5.,), opacities=[.6], colors=[[1., 0., 0.]])
            scene["covariances_world_m2"][0] = covariance
            result = reference.render_reference_observation(scene, self.camera(), K, size, [.2, .4, .6])
            self.assertEqual(result["rgb"].shape, (5, 7, 3))
            self.assertEqual(result["alpha"].shape, (5, 7))
            expected_rgb = np.array([expected_alpha + (1 - expected_alpha)*.2,
                                     (1 - expected_alpha)*.4, (1 - expected_alpha)*.6])
            errors.extend([float(abs(result["alpha"][4, 4] - expected_alpha)),
                           float(np.abs(result["rgb"][4, 4] - expected_rgb).max())])
        self.assertLess(max(errors), ANALYTIC_ATOL)
        self.metrics["independent_off_center_kernel_max_error"] = max(errors)

    def test_tie_by_id_and_permutation_invariance(self):
        scene = self.scene((2., 2.), ids=[20, 10], opacities=[.25, .5], colors=[[1., 0., 0.], [0., 0., 1.]])
        result = self.render_pixel(scene, [.2, .4, .6])
        np.testing.assert_array_equal(result["sorted_ids"], [10, 20])
        np.testing.assert_allclose(result["rgb"][0, 0], [.2, .15, .725], atol=ANALYTIC_ATOL, rtol=0)
        reversed_input = {k: v[::-1].copy() if isinstance(v, np.ndarray) else v for k, v in scene.items()}
        reverse = self.render_pixel(reversed_input, [.2, .4, .6])
        tie_error = float(np.abs(result["rgb"] - reverse["rgb"]).max())
        f = self.fixture
        order = np.random.default_rng(PERMUTATION_SEED).permutation(16)
        shuffled = {k: v[order].copy() if isinstance(v, np.ndarray) else v for k, v in f["gaussians"].items()}
        camera = state_contract.current_camera_pose(f["T_world_from_vehicle"], f["rig"]["front"])
        original = reference.render_reference_observation(f["gaussians"], camera, f["K"], f["image_size"])
        changed = reference.render_reference_observation(shuffled, camera, f["K"], f["image_size"])
        error = max(tie_error, *(float(np.abs(original[k] - changed[k]).max()) for k in ("rgb", "alpha", "transmittance")))
        self.assertLess(error, PERMUTATION_ATOL)
        np.testing.assert_array_equal(original["sorted_ids"], changed["sorted_ids"])
        self.metrics["permutation_and_depth_tie_max_error"] = error

    def test_all_invalid_offscreen_centers_and_zero_opacity(self):
        background = [.2, .4, .6]
        result = self.render_pixel(self.scene((-1., -2.)), background)
        np.testing.assert_array_equal(result["rgb"][0, 0], background)
        self.assertFalse(result["projected_valid_mask"].any())
        self.assertFalse(result["renderable_mask"].any())
        self.assertEqual(len(result["sorted_ids"]), 0)
        self.assertEqual(result["alpha"].max(), 0.)
        nonfinite = self.scene((2., 4.))
        nonfinite["centers_world_m"][0, 0] = np.nan
        nonfinite["centers_world_m"][1, 0] = np.inf
        masked = self.render_pixel(nonfinite, background)
        self.assertFalse(masked["projected_valid_mask"].any())
        np.testing.assert_array_equal(masked["rgb"][0, 0], background)
        self.finite_ranges(masked)
        K = np.array([[10., 0., 3.], [0., 10., 2.], [0., 0., 1.]])
        offscreen = self.scene((5.,))
        offscreen["centers_world_m"][0, 0] = 2.  # u=7 at the excluded right boundary.
        result = reference.render_reference_observation(offscreen, self.camera(), K, [7, 5], background)
        np.testing.assert_array_equal(result["rgb"], np.tile(background, (5, 7, 1)))
        self.assertFalse(result["projected_valid_mask"].any())
        self.assertEqual(result["alpha"].max(), 0.)
        zero = self.render_pixel(self.scene((2., 4.), opacities=[0., 0.]), [1., 1., 1.])
        np.testing.assert_array_equal(zero["rgb"], np.ones((1, 1, 3)))
        self.assertTrue(zero["projected_valid_mask"].all())
        self.assertEqual(zero["alpha"].max(), 0.)
        self.finite_ranges(result)

    def test_covariance_screen_rejection_and_underflow_policy(self):
        skipped_count = 0
        for covariance in (np.diag([4e-14, 4e-14, .04]), np.diag([4e-8, 4e4, .04])):
            scene = self.scene()
            scene["covariances_world_m2"][0] = covariance
            result = self.render_pixel(scene, [.2, .4, .6])
            self.assertTrue(result["projected_valid_mask"][0])
            self.assertFalse(result["renderable_mask"][0])
            np.testing.assert_array_equal(result["skipped_screen_ids"], [1])
            np.testing.assert_array_equal(result["rgb"][0, 0], [.2, .4, .6])
            self.assertEqual(result["alpha"].max(), 0.)
            skipped_count += 1
        narrow = self.scene((5.,))
        narrow["covariances_world_m2"][0] = np.diag([2.5e-9, 2.5e-9, .04])
        K = np.array([[10., 0., 3.], [0., 10., 2.], [0., 0., 1.]])
        result = reference.render_reference_observation(narrow, self.camera(), K, [7, 5])
        self.assertTrue(result["renderable_mask"][0])
        self.assertAlmostEqual(result["alpha"][2, 3], .4)
        self.assertEqual(result["alpha"][2, 4], 0.)
        self.finite_ranges(result)
        self.metrics["explicit_screen_covariance_skips"] = skipped_count

    def test_four_views_camera_motion_and_static_world(self):
        f = self.fixture
        before = deepcopy(f["gaussians"])
        backgrounds = (.05, .1, .15)
        initial = {}
        for view in CAMERA_IDS:
            camera = state_contract.current_camera_pose(f["T_world_from_vehicle"], f["rig"][view])
            result = reference.render_reference_observation(f["gaussians"], camera, f["K"], f["image_size"], backgrounds)
            projection = state_contract.project_gaussians(f["gaussians"], camera, f["K"], f["image_size"])
            np.testing.assert_array_equal(result["projected_valid_mask"], projection["valid_mask"])
            expected_rows = np.flatnonzero(projection["valid_mask"])
            expected_rows = expected_rows[np.lexsort((projection["ids"][expected_rows], projection["depth_z_m"][expected_rows]))]
            np.testing.assert_array_equal(result["sorted_ids"], projection["ids"][expected_rows])
            self.assertEqual(result["rgb"].shape, (96, 128, 3))
            self.assertGreater(result["alpha"].max(), 1e-3)
            self.finite_ranges(result)
            initial[view] = result
        dt = np.array([.07, .13, .11, .17, .09, .16, .08, .14, .12, .1])
        differences, final = {}, {}
        for label, action in (("forward", [1., 0.]), ("left", [1., .3])):
            # Separate, explicitly synthetic ORACLE; renderer sees one pose only.
            hypothetical = synthetic_bicycle_ORACLE(f["initial_state"], np.tile(action, (10, 1)), dt, 2.7)[-1]
            vehicle = yaw_transform(hypothetical[2], [hypothetical[0], hypothetical[1], 0.])
            differences[label], final[label] = {}, {}
            for view in CAMERA_IDS:
                camera = state_contract.current_camera_pose(vehicle, f["rig"][view])
                result = reference.render_reference_observation(f["gaussians"], camera, f["K"], f["image_size"], backgrounds)
                self.finite_ranges(result)
                self.assertGreater(result["alpha"].max(), 1e-3)
                differences[label][view] = float(np.abs(result["rgb"] - initial[view]["rgb"]).max())
                final[label][view] = result
            self.same(f["gaussians"], before)
        candidate_difference = float(np.abs(final["forward"]["front"]["rgb"] - final["left"]["front"]["rgb"]).max())
        self.assertGreater(max(v for by_view in differences.values() for v in by_view.values()), MOTION_RGB_MIN)
        self.assertGreater(candidate_difference, MOTION_RGB_MIN)
        self.metrics.update(camera_motion_rgb_max_differences=differences, forward_left_front_rgb_max_difference=candidate_difference,
                            static_center_max_change_m=float(np.abs(f["gaussians"]["centers_world_m"] - before["centers_world_m"]).max()),
                            static_covariance_max_change_m2=float(np.abs(f["gaussians"]["covariances_world_m2"] - before["covariances_world_m2"]).max()))

    def test_external_object_motion_observation_change(self):
        f = self.fixture
        before = deepcopy(f["gaussians"])
        selected = np.array([7000, 7002, 7009], dtype=np.int64)
        moved = state_contract.transform_world_subset(f["gaussians"], selected, yaw_transform(.35, [.3, -.2, .15]))
        changes = {}
        for view in ("front", "left"):
            camera = state_contract.current_camera_pose(f["T_world_from_vehicle"], f["rig"][view])
            original = reference.render_reference_observation(f["gaussians"], camera, f["K"], f["image_size"])
            changed = reference.render_reference_observation(moved, camera, f["K"], f["image_size"])
            changes[view] = float(np.abs(original["rgb"] - changed["rgb"]).max())
            self.assertGreater(changes[view], MOTION_RGB_MIN)
        unaffected = ~np.isin(f["gaussians"]["ids"], selected)
        for key in ("centers_world_m", "covariances_world_m2"):
            np.testing.assert_array_equal(moved[key][unaffected], before[key][unaffected])
        self.same(f["gaussians"], before)
        self.metrics["external_object_motion_rgb_max_differences"] = changes

    def test_repeatability_no_mutation_and_projection_is_reused(self):
        f = self.fixture
        camera = state_contract.current_camera_pose(f["T_world_from_vehicle"], f["rig"]["front"])
        scene, K, size, background = deepcopy(f["gaussians"]), f["K"].copy(), f["image_size"].copy(), np.array([.2, .3, .4])
        snapshots = deepcopy((scene, camera, K, size, background))
        projection = state_contract.project_gaussians(scene, camera, K, size)
        snapshot_projection = deepcopy(projection)
        with patch.object(reference, "project_gaussians", return_value=projection) as reused:
            first = reference.render_reference_observation(scene, camera, K, size, background)
            self.assertEqual(reused.call_count, 1)
        self.same(projection, snapshot_projection)
        repeated = reference.render_reference_observation(scene, camera, K, size, background)
        self.same(first, repeated)
        self.same(scene, snapshots[0])
        self.same(camera, snapshots[1])
        for actual, expected in zip((K, size, background), snapshots[2:]):
            np.testing.assert_array_equal(actual, expected)
        first["projected_valid_mask"][:] = False
        self.same(projection, snapshot_projection)
        self.metrics["deterministic_repeat_max_rgb_error"] = float(np.abs(first["rgb"] - repeated["rgb"]).max())
        self.assertFalse(any(name in sys.modules for name in ("torch", "scipy", "gsplat", "carla")))

    def test_illegal_inputs_settings_and_future_fields_rejected(self):
        for background in ([np.nan, 0., 0.], [np.inf, 0., 0.], [-.1, 0., 0.], [1.1, 0., 0.], [0., 0.], [[0., 0., 0.]], ["red", "green", "blue"]):
            self.reject(f"background_{len(self.rejected)}", lambda: self.render_pixel(self.scene(), background))
        for size in ([129, 96], [128, 97], [0, 96], [128., 96.], [128, 96, 3]):
            self.reject(f"image_size_{size}", lambda: reference.render_reference_observation(self.scene(), self.camera(), np.eye(3), size))
        for kwargs in ({"max_condition": 0.}, {"max_condition": np.nan}, {"max_condition": True},
                       {"min_variance_px2": 0.}, {"min_variance_px2": np.inf}):
            self.reject(f"settings_{len(self.rejected)}", lambda: reference.render_reference_observation(self.scene(), self.camera(), np.eye(3), [1, 1], **kwargs))
        for field, value in (("opacities", np.array([np.nan])), ("colors_rgb", np.array([[-1., 0., 0.]])),
                             ("centers_world_m", np.zeros((1, 4))), ("future_target", np.zeros((10, 3))),
                             ("future_observation", np.zeros((10, 1, 1, 3))), ("full_episode", {}),
                             ("recorded_actions", np.zeros((10, 2)))):
            bad = self.scene()
            bad[field] = value
            self.reject(f"scene_{field}", lambda: self.render_pixel(bad))
        camera = self.camera()
        camera["future_camera_trajectory"] = np.tile(np.eye(4), (10, 1, 1))
        self.reject("camera_future_trajectory", lambda: reference.render_reference_observation(self.scene(), camera, np.eye(3), [1, 1]))
        camera = self.camera()
        camera["T_world_from_optical"] = np.tile(np.eye(4), (10, 1, 1))
        self.reject("trajectory_as_one_camera", lambda: reference.render_reference_observation(self.scene(), camera, np.eye(3), [1, 1]))
        self.reject("bad_K_shape", lambda: reference.render_reference_observation(self.scene(), self.camera(), np.eye(4), [1, 1]))
        self.reject("extra_future_keyword", lambda: reference.render_reference_observation(self.scene(), self.camera(), np.eye(3), [1, 1], future_target={}))


if __name__ == "__main__":
    start = time.perf_counter()
    result = unittest.main(verbosity=2, exit=False).result
    elapsed = time.perf_counter() - start
    wall = time.perf_counter() - PROCESS_START
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    budget_ok = wall <= MAX_WALL_SECONDS and rss < MAX_RSS_BYTES
    status = "BLOCKED" if not result.wasSuccessful() else ("PASS" if budget_ok else "NOT_CONFIRMED")
    print("REFERENCE_RESULTS " + json.dumps(dict(status=status, fixture_seed=state_contract.SEED,
          permutation_seed=PERMUTATION_SEED, metrics=ObservationReferenceTests.metrics,
          rejected_count=len(ObservationReferenceTests.rejected), rejected_cases=ObservationReferenceTests.rejected,
          analytic_atol=ANALYTIC_ATOL, permutation_atol=PERMUTATION_ATOL, range_atol=RANGE_ATOL,
          log_kernel_cutoff=reference.LOG_KERNEL_CUTOFF, max_condition=reference.MAX_CONDITION,
          min_variance_px2=reference.MIN_VARIANCE_PX2, real_data_gate="BLOCKED_MISSING_EVIDENCE"), allow_nan=False))
    print("TEST_SUMMARY " + json.dumps(dict(status=status, tests=result.testsRun, failures=len(result.failures),
          errors=len(result.errors), skipped=len(result.skipped), test_elapsed_seconds=elapsed, wall_seconds=wall,
          peak_rss_bytes=rss, max_wall_seconds=MAX_WALL_SECONDS, max_rss_bytes=MAX_RSS_BYTES,
          python=sys.version.split()[0], numpy=np.__version__, cpu_only=True)))
    sys.exit(0 if result.wasSuccessful() and budget_ok else 1)
