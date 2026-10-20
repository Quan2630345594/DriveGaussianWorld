"""Focused CPU-only tests for trusted fixture planning; no training/rendering."""

from copy import deepcopy
import unittest

import numpy as np

from gaussian_park_world.planning import (
    FailureReason, collision_free, fixture_map, integrate, plan_and_track,
    require_planning_geometry,
)


class OfflinePlanningTests(unittest.TestCase):
    def setUp(self):
        self.map = fixture_map("offline-test")
        self.start = np.array([0., 0., 0., 0.])

    def assert_replay(self, result, goal):
        self.assertEqual(result["status"], "SUCCESS", result["reason"])
        self.assertIsNone(result["reason"])
        states, actions, dt = (result[k] for k in ("states", "actions", "dt"))
        self.assertEqual(states.shape, (len(actions) + 1, 4))
        self.assertEqual(actions.shape, (len(dt), 2))
        self.assertTrue(np.all(dt > 0))
        self.assertEqual(result["metadata"]["simulator"], "SYNTHETIC_ZERO_DELAY")
        self.assertEqual(result["metadata"]["wheelbase"], 2.7)
        for i, (action, duration) in enumerate(zip(actions, dt)):
            np.testing.assert_allclose(integrate(states[i], action, duration), states[i + 1], atol=1e-12)
            self.assertEqual(states[i + 1, 3], action[0])
            for fraction in np.linspace(.05, 1., 20):
                self.assertTrue(collision_free(integrate(states[i], action, duration * fraction), self.map))
        self.assertTrue(np.all(np.abs(actions[:, 0]) <= 1.))
        self.assertTrue(np.all(np.abs(actions[:, 1]) <= .55))
        np.testing.assert_array_equal(actions[-1], [0., 0.])
        self.assertEqual(states[-1, 3], 0.)
        self.assertLessEqual(np.linalg.norm(states[-1, :2] - goal[:2]), .35)
        yaw_error = states[-1, 2] - goal[2]
        self.assertLessEqual(abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))), .12)

    def test_forward_reverse_and_action_replay(self):
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                goal = np.array([sign * 3., 0., 0., 0.])
                result = plan_and_track(self.map, self.start, goal)
                self.assert_replay(result, goal)
                self.assertTrue(np.any(sign * result["actions"][:, 0] > 0))
                self.assertEqual(result["states"][0, 3], 0.)
                self.assertEqual(result["states"][1, 3], sign)

    def test_forward_and_reverse_steering(self):
        for speed, steering in ((1., .55), (1., -.55), (-1., .55), (-1., -.55)):
            with self.subTest(speed=speed, steering=steering):
                goal = integrate(self.start, [speed, steering], 3.)
                goal[3] = 0.
                result = plan_and_track(self.map, self.start, goal)
                self.assert_replay(result, goal)
                self.assertTrue(np.any(np.abs(result["actions"][:, 1]) > .01))

    def test_exact_arc_and_speed_snap(self):
        state = integrate(self.start, [1., .4], 2.)
        radius = 2.7 / np.tan(.4)
        np.testing.assert_allclose(state, [radius * np.sin(2 / radius),
                                         radius * (1 - np.cos(2 / radius)), 2 / radius, 1.])
        np.testing.assert_allclose(integrate(state, [-1., .4], 2.)[:3], self.start[:3], atol=1e-12)
        np.testing.assert_array_equal(integrate(state, [0., 0.], .1)[:3], state[:3])

    def test_already_at_goal_still_stops(self):
        start = np.array([0., 0., 0., .5])
        result = plan_and_track(self.map, start, self.start)
        self.assert_replay(result, self.start)
        self.assertEqual(len(result["actions"]), 1)
        np.testing.assert_array_equal(result["states"][0], start)

    def test_geometry_gate_rejects_missing_evidence_and_real_unknown(self):
        scene = deepcopy(self.map)
        self.assertIs(require_planning_geometry(scene, self.map), self.map)
        for field, value in (("metric_scale_trusted", False), ("metric_scale_evidence", ""),
                             ("gauge_id", "other"), ("source_type", "REAL_UNKNOWN")):
            bad = deepcopy(scene)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_planning_geometry(bad, self.map)
        for field, value in (("ground", dict(trusted=True)), ("unknown_occupied", False),
                             ("source_type", "GAUSSIAN_ALPHA"), ("obstacles", [[0., 0., -1., 1.]])):
            bad = deepcopy(self.map)
            bad[field] = value
            self.assertFalse(collision_free(self.start, bad))
            self.assertEqual(plan_and_track(bad, self.start, [3., 0., 0.])["reason"],
                             FailureReason.UNTRUSTED_GEOMETRY)

    def test_full_footprint_unknown_and_bounds(self):
        self.assertFalse(collision_free([13., 0., 0., 0.], self.map))
        self.map["unknown_rectangles"] = [[3., -.2, 3.1, .2]]
        self.assertFalse(collision_free(self.start, self.map))
        self.map["unknown_rectangles"] = []
        self.map["alpha"] = np.ones(100)
        self.assertTrue(collision_free(self.start, self.map))

    def test_scene_manifest_gauge_and_evidence(self):
        scene = dict(scene_id="offline-test", source_classification="SYNTHETIC_PLANNING_FIXTURE",
                     planning_status="SYNTHETIC_FIXTURE_ONLY",
                     world_gauge=dict(id="synthetic:offline-test:0", metric_scale="verified",
                                      units="meters", axes="x_forward_y_left_z_up",
                                      confidence="synthetic_by_construction"))
        self.assertIs(require_planning_geometry(scene, self.map), self.map)
        for field, value in (("id", "other"), ("metric_scale", "unknown"),
                             ("confidence", "unknown"), ("units", "unknown")):
            bad = deepcopy(scene)
            bad["world_gauge"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_planning_geometry(bad, self.map)

    def test_failures_and_bounded_exhaustion(self):
        cases = [(None, self.start, [3., 0., 0.], FailureReason.NO_MAP),
                 (self.map, [13., 0., 0., 0.], self.start, FailureReason.START_BLOCKED),
                 (self.map, self.start, [13., 0., 0.], FailureReason.GOAL_BLOCKED),
                 (self.map, self.start, [3., 0., 0., 1.], FailureReason.INVALID_INPUT)]
        for map, start, goal, reason in cases:
            result = plan_and_track(map, start, goal)
            self.assertEqual(result["status"], "FAILURE")
            self.assertEqual(result["reason"], reason)
            self.assertEqual(result["actions"].shape, (0, 2))
        result = plan_and_track(self.map, self.start, [8., 2., 1.], max_expansions=2, max_retries=1)
        self.assertEqual(result["reason"], FailureReason.SEARCH_EXHAUSTED)
        self.assertLessEqual(result["metadata"]["expansions"], 2)
        self.assertEqual(result["metadata"]["attempts"], 2)

    def test_swept_footprint_cannot_tunnel_thin_wall(self):
        # Start/goal are free on opposite sides; no path through this full-height wall.
        self.map["obstacles"] = [[3.8, -8., 3.81, 8.]]
        self.assertTrue(collision_free(self.start, self.map))
        self.assertTrue(collision_free([7., 0., 0., 0.], self.map))
        result = plan_and_track(self.map, self.start, [7., 0., 0.], max_expansions=30, max_retries=0)
        self.assertEqual(result["reason"], FailureReason.SEARCH_EXHAUSTED)


if __name__ == "__main__":
    unittest.main()
