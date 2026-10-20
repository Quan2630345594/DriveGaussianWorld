"""CPU pinhole geometry: meters, column-vector transforms, proper SE(3) only."""

import numpy as np


CAMERA_IDS = ("front", "back", "left", "right")


def yaw_transform(yaw_rad, translation_m):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T_target_from_source = np.eye(4)
    T_target_from_source[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T_target_from_source[:3, 3] = translation_m
    return T_target_from_source


def validate_transform(T_target_from_source):
    T = np.asarray(T_target_from_source, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Transform must be finite [4,4]")
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-10, rtol=0)):
        raise ValueError("Expected proper SE(3); reflections are convention changes")
    return T


def invert_transform(T_target_from_source):
    T = validate_transform(T_target_from_source)
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ T[:3, 3]
    return inverse


def transform_points(T_target_from_source, points_source_m):
    T = validate_transform(T_target_from_source)
    points = np.asarray(points_source_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite [P,3]")
    return points @ T[:3, :3].T + T[:3, 3]


def validate_intrinsics(K, image_size_wh):
    size = np.asarray(image_size_wh)
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
        raise ValueError("Image size must be positive integer [width,height]")
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(K[[0, 1], [1, 0]], 0, atol=1e-10, rtol=0)
            or not 0 <= K[0, 2] < size[0] or not 0 <= K[1, 2] < size[1]):
        raise ValueError("Expected zero-skew pinhole K in pixel units")
    return K


def project_points(points_optical_m, K, image_size_wh):
    """Return pixels, axial Z-depth and mask; invalid rows are NaN, not pixels."""
    K = validate_intrinsics(K, image_size_wh)
    points = np.asarray(points_optical_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Optical points must have shape [P,3]")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    uv = np.full((len(points), 2), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        uv[valid] = points[valid, :2] / points[valid, 2:3]
        uv[valid] = uv[valid] * K[[0, 1], [0, 1]] + K[:2, 2]
    valid &= (np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1)
              & (uv < np.asarray(image_size_wh)).all(axis=1))
    uv[~valid] = np.nan
    depth_z_m = np.where(valid, points[:, 2], np.nan)
    return uv, depth_z_m, valid


def backproject_points(uv_px, depth_z_m, K, image_size_wh):
    K = validate_intrinsics(K, image_size_wh)
    uv, depth = np.asarray(uv_px, dtype=np.float64), np.asarray(depth_z_m, dtype=np.float64)
    if (uv.ndim != 2 or uv.shape[1] != 2 or depth.shape != (len(uv),)
            or not np.isfinite(uv).all() or not np.isfinite(depth).all()
            or np.any(depth <= 0) or np.any(uv < 0)
            or np.any(uv >= np.asarray(image_size_wh))):
        raise ValueError("Backprojection requires in-image pixels and finite positive Z-depth")
    xy = (uv - K[:2, 2]) / K[[0, 1], [0, 1]] * depth[:, None]
    return np.column_stack((xy, depth))
