"""CPU pinhole geometry tests, independent of the retired episode interface."""

import json
import sys
import unittest

import numpy as np

from gaussian_park_world.world_model.geometry import (
    CAMERA_IDS, backproject_points, invert_transform, project_points, transform_points,
    validate_intrinsics, validate_transform, yaw_transform,
)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Geometry-only inputs retain the original 16 poses and four-camera rig.
        rng = np.random.default_rng(20261014)
        timestamps = np.arange(16, dtype=np.float64) * .1
        commands = np.column_stack((.4 * np.where(np.arange(15) < 8, 1, -1), np.full(15, .12)))
        state = np.zeros((16, 3), dtype=np.float64)
        state[0] = [2., -1., .4]
        for t in range(15):
            dt = timestamps[t + 1] - timestamps[t]
            state[t + 1, :2] = state[t, :2] + dt * commands[t, 0] * np.array([np.cos(state[t, 2]), np.sin(state[t, 2])])
            state[t + 1, 2] = state[t, 2] + dt * commands[t, 1]
        poses = np.stack([yaw_transform(row[2], [row[0], row[1], 0.]) for row in state])
        front_rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        rig = np.stack([yaw_transform(yaw, translation) for yaw, translation in zip(
            [0., np.pi, np.pi / 2, -np.pi / 2], [[1., 0., .8], [-1., 0., .8], [0., .5, .8], [0., -.5, .8]])])
        rig[:, :3, :3] = rig[:, :3, :3] @ front_rotation
        K = np.broadcast_to(np.array([[80., 0., 64.], [0., 80., 48.], [0., 0., 1.]]), (4, 3, 3)).copy()
        size = np.array([128, 96], dtype=np.int64)
        points = []
        for v in range(len(CAMERA_IDS)):
            optical = np.column_stack((rng.uniform(-.6, .6, 2), rng.uniform(-.3, .3, 2), [5., 6.]))
            points.extend(transform_points(poses[0] @ rig[v], optical))
        points = np.asarray(points)
        uv, depth, valid = np.empty((16, 4, 8, 2)), np.empty((16, 4, 8)), np.empty((16, 4, 8), dtype=bool)
        for t in range(16):
            for v in range(4):
                optical = transform_points(invert_transform(poses[t] @ rig[v]), points)
                uv[t, v], depth[t, v], valid[t, v] = project_points(optical, K[v], size)
        cls.arrays = dict(world_points_gt_m=points, T_world_from_vehicle=poses,
                          T_vehicle_from_optical=rig, camera_K=K, image_size_wh=size,
                          observed_uv_px=uv, observed_depth_z_m=depth, observed_valid=valid)

    def test_geometry_metrics_all_frames_and_camera_pairs(self):
        a = self.arrays
        points = a["world_points_gt_m"]
        errors = dict(world_vehicle_roundtrip_max_m=0., optical_world_roundtrip_max_m=0.,
                      backprojection_max_m=0., cross_camera_roundtrip_max_m=0.,
                      reprojection_max_px=0., cross_camera_reprojection_max_px=0.)
        comparisons, valid_counts = 0, np.zeros(4, dtype=np.int64)
        def metric(name, left, right):
            measured = float(np.max(np.linalg.norm(left - right, axis=-1)))
            self.assertTrue(np.isfinite(measured), name)
            errors[name] = max(errors[name], measured)
        for t in range(16):
            T_world_from_vehicle = a["T_world_from_vehicle"][t]
            vehicle_points = transform_points(invert_transform(T_world_from_vehicle), points)
            metric("world_vehicle_roundtrip_max_m", transform_points(T_world_from_vehicle, vehicle_points), points)
            for source in range(4):
                T_world_from_optical_source = T_world_from_vehicle @ a["T_vehicle_from_optical"][source]
                source_points = transform_points(invert_transform(T_world_from_optical_source), points)
                recovered_world = transform_points(T_world_from_optical_source, source_points)
                recovered_optical = transform_points(invert_transform(T_world_from_optical_source), recovered_world)
                metric("optical_world_roundtrip_max_m", recovered_optical, source_points)
                uv, depth, visible = project_points(source_points, a["camera_K"][source], a["image_size_wh"])
                np.testing.assert_array_equal(visible, a["observed_valid"][t, source])
                np.testing.assert_allclose(uv, a["observed_uv_px"][t, source], atol=1e-9, rtol=0)
                np.testing.assert_allclose(depth, a["observed_depth_z_m"][t, source], atol=1e-10, rtol=0)
                valid_counts[source] += visible.sum()
                self.assertGreater(visible.sum(), 0)
                stored_uv = a["observed_uv_px"][t, source, visible]
                stored_depth = a["observed_depth_z_m"][t, source, visible]
                reconstructed = backproject_points(stored_uv, stored_depth, a["camera_K"][source], a["image_size_wh"])
                metric("backprojection_max_m", reconstructed, source_points[visible])
                reprojected, _, reprojected_valid = project_points(reconstructed, a["camera_K"][source], a["image_size_wh"])
                self.assertTrue(reprojected_valid.all())
                metric("reprojection_max_px", reprojected, stored_uv)
                for target in range(4):
                    if target == source:
                        continue
                    T_world_from_optical_target = T_world_from_vehicle @ a["T_vehicle_from_optical"][target]
                    target_points = transform_points(invert_transform(T_world_from_optical_target), recovered_world)
                    recovered_source = transform_points(invert_transform(T_world_from_optical_source),
                                                        transform_points(T_world_from_optical_target, target_points))
                    metric("cross_camera_roundtrip_max_m", recovered_source, source_points)
                    uv_target, _, target_visible = project_points(target_points, a["camera_K"][target], a["image_size_wh"])
                    np.testing.assert_array_equal(target_visible, a["observed_valid"][t, target])
                    metric("cross_camera_reprojection_max_px", uv_target[target_visible],
                           a["observed_uv_px"][t, target, target_visible])
                    comparisons += int(target_visible.sum())
        for name, error in errors.items():
            self.assertTrue(np.isfinite(error), name)
            self.assertLess(error, 1e-9 if name.endswith("px") else 1e-5, name)
        np.testing.assert_array_equal(valid_counts, [32, 32, 32, 32])
        self.assertEqual(comparisons, 384)
        print("GEOMETRY_RESULTS " + json.dumps(dict(seed=20261014, frames=16, **errors,
              visible_samples_per_camera=valid_counts.tolist(), cross_camera_pixel_comparisons=comparisons), sort_keys=True))

    def test_known_pinhole_and_positive_z_visibility(self):
        K, size = self.arrays["camera_K"][0], self.arrays["image_size_wh"]
        points = np.array([[1., .5, 5.], [0., 0., -1.], [0., 0., 0.],
                           [np.nan, 0., 1.], [0., np.inf, 1.], [100., 0., 1.]])
        uv, depth, mask = project_points(points, K, size)
        np.testing.assert_array_equal(mask, [True, False, False, False, False, False])
        np.testing.assert_array_equal(uv[0], [80., 56.])
        self.assertEqual(depth[0], 5.)
        self.assertTrue(np.isnan(uv[1:]).all() and np.isnan(depth[1:]).all())
        np.testing.assert_allclose(backproject_points(uv[:1], depth[:1], K, size), points[:1], atol=1e-12, rtol=0)

    def test_camera_axes_and_composition_order(self):
        a = self.arrays
        directions = ([1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.])
        for v, direction in enumerate(directions):
            T_vehicle_from_optical = a["T_vehicle_from_optical"][v]
            origin = transform_points(T_vehicle_from_optical, np.zeros((1, 3)))[0]
            forward = transform_points(T_vehicle_from_optical, np.array([[0., 0., 1.]]))[0] - origin
            down = transform_points(T_vehicle_from_optical, np.array([[0., 1., 0.]]))[0] - origin
            np.testing.assert_allclose(forward, direction, atol=1e-12, rtol=0)
            np.testing.assert_allclose(down, [0, 0, -1], atol=1e-12, rtol=0)
            optical = np.array([[.2, -.1, 4.]])
            composed = transform_points(a["T_world_from_vehicle"][7] @ T_vehicle_from_optical, optical)
            sequential = transform_points(a["T_world_from_vehicle"][7], transform_points(T_vehicle_from_optical, optical))
            np.testing.assert_allclose(composed, sequential, atol=1e-12, rtol=0)

    def test_invalid_transforms_and_point_dimensions(self):
        reflection = np.diag([1., -1., 1., 1.])
        bad = [np.eye(3), reflection, np.full((4, 4), np.nan)]
        nonrigid = np.eye(4)
        nonrigid[0, 0] = 2.
        bad.append(nonrigid)
        malformed = np.eye(4)
        malformed[3, 0] = .1
        bad.append(malformed)
        for T in bad:
            with self.subTest(transform=T), self.assertRaises(ValueError):
                validate_transform(T)
        # A handedness reflection is reversible but is not a proper SE(3) rotation.
        points = np.array([[1., 2., 3.]])
        np.testing.assert_array_equal(points @ reflection[:3, :3].T @ reflection[:3, :3].T, points)
        self.assertEqual(np.linalg.det(reflection[:3, :3]), -1.)
        for points in (np.zeros(3), np.zeros((3, 4)), np.full((2, 3), np.inf)):
            with self.subTest(points=points), self.assertRaises(ValueError):
                transform_points(np.eye(4), points)
        with self.assertRaises(ValueError):
            project_points(np.zeros((3, 2)), self.arrays["camera_K"][0], [128, 96])

    def test_invalid_intrinsics_depth_and_pixels(self):
        K, size = self.arrays["camera_K"][0], self.arrays["image_size_wh"]
        for row, col, value in [(0, 0, -1.), (1, 1, 0.), (0, 2, 128.),
                                (0, 1, 1.), (1, 0, 1.), (2, 2, 2.), (0, 0, np.nan)]:
            bad = K.copy()
            bad[row, col] = value
            with self.subTest(K=bad), self.assertRaises(ValueError):
                validate_intrinsics(bad, size)
        for bad_size in ([0, 96], [128., 96.], [128], [128, 96, 3]):
            with self.subTest(size=bad_size), self.assertRaises(ValueError):
                validate_intrinsics(K, bad_size)
        with self.assertRaises(ValueError):
            validate_intrinsics(np.eye(4), size)
        for bad_depth in (-1., 0., np.nan, np.inf):
            with self.subTest(depth=bad_depth), self.assertRaises(ValueError):
                backproject_points([[64., 48.]], [bad_depth], K, size)
        for bad_uv in ([[128., 48.]], [[-1., 48.]], [[np.nan, 48.]], [[64., 48., 0.]]):
            with self.subTest(uv=bad_uv), self.assertRaises(ValueError):
                backproject_points(bad_uv, [5.], K, size)
        with self.assertRaises(ValueError):
            backproject_points([[64., 48.]], [[5.]], K, size)

    def test_cpu_only_imports(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "gsplat", "carla", "scipy")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
