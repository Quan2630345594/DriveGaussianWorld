"""Synthetic world Gaussian geometry only: no rasterization, training or GWM."""

import numpy as np

from .geometry import CAMERA_IDS, invert_transform, project_points, transform_points, validate_intrinsics, validate_transform, yaw_transform


SEED = 707001
PERMUTATION_SEED = 707002
MAX_GAUSSIANS = 32
PROVENANCE = "synthetic_debug_fixture_only"
GAUSSIAN_FIELDS = {"provenance", "ids", "centers_world_m", "covariances_world_m2", "opacities", "colors_rgb"}
CAMERA_FIELDS = {"role", "T_world_from_vehicle", "T_vehicle_from_optical", "T_world_from_optical"}


def validate_gaussians(gaussians):
    """Reject bad schema/attributes; return a per-row finite/SPD geometry mask."""
    if (not isinstance(gaussians, dict) or set(gaussians) != GAUSSIAN_FIELDS
            or gaussians["provenance"] != PROVENANCE):
        raise ValueError("Expected synthetic Gaussian state only; no future/map/target fields")
    ids = gaussians["ids"]
    if (not isinstance(ids, np.ndarray) or ids.ndim != 1 or ids.dtype != np.int64
            or not 1 <= len(ids) <= MAX_GAUSSIANS or len(np.unique(ids)) != len(ids)):
        raise ValueError("Expected a small nonempty set of unique fixture IDs")
    n = len(ids)
    for key, shape in (("centers_world_m", (n, 3)), ("covariances_world_m2", (n, 3, 3)),
                       ("opacities", (n,)), ("colors_rgb", (n, 3))):
        value = gaussians[key]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float64:
            raise ValueError(f"Invalid Gaussian shape/dtype: {key}")
    for key in ("opacities", "colors_rgb"):
        value = gaussians[key]
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError(f"Expected finite activated [0,1] attributes: {key}")
    covariance = gaussians["covariances_world_m2"]
    valid = np.isfinite(gaussians["centers_world_m"]).all(axis=1) & np.isfinite(covariance).all(axis=(1, 2))
    for i in np.flatnonzero(valid):
        if not np.allclose(covariance[i], covariance[i].T, atol=1e-12, rtol=0):
            valid[i] = False
            continue
        try:
            eigenvalues = np.linalg.eigvalsh(covariance[i])
            valid[i] = np.isfinite(eigenvalues).all() and np.all(eigenvalues > 0)
        except np.linalg.LinAlgError:
            valid[i] = False
    return valid


def current_camera_pose(T_world_from_vehicle, T_vehicle_from_optical):
    """Named, single-time descriptor; all matrices use column-vector semantics."""
    vehicle = validate_transform(T_world_from_vehicle).copy()
    rig = validate_transform(T_vehicle_from_optical).copy()
    return dict(role="current_camera_pose", T_world_from_vehicle=vehicle,
                T_vehicle_from_optical=rig, T_world_from_optical=vehicle @ rig)


def _world_to_optical(camera_pose):
    if (not isinstance(camera_pose, dict) or set(camera_pose) != CAMERA_FIELDS
            or camera_pose["role"] != "current_camera_pose"):
        raise ValueError("Need one named current camera descriptor, not future poses/targets")
    vehicle = validate_transform(camera_pose["T_world_from_vehicle"])
    rig = validate_transform(camera_pose["T_vehicle_from_optical"])
    camera = validate_transform(camera_pose["T_world_from_optical"])
    if not np.allclose(camera, vehicle @ rig, atol=1e-10, rtol=0):
        raise ValueError("Camera direction/composition mismatch: require T_world_from_vehicle @ T_vehicle_from_optical")
    return invert_transform(camera)


def pinhole_jacobian(points_optical_m, K, image_size):
    """Perspective derivative [N,2,3] in pixel/m, for finite positive Z only."""
    K = validate_intrinsics(K, image_size)
    points = np.asarray(points_optical_m, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or np.any(points[:, 2] <= 0)):
        raise ValueError("Jacobian needs finite optical XYZ with positive Z")
    z = points[:, 2]
    J = np.zeros((len(points), 2, 3), dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        J[:, 0, 0], J[:, 1, 1] = K[0, 0] / z, K[1, 1] / z
        J[:, 0, 2] = -K[0, 0] * (points[:, 0] / z) / z
        J[:, 1, 2] = -K[1, 1] * (points[:, 1] / z) / z
    if not np.isfinite(J).all():
        raise ValueError("Jacobian overflowed")
    return J


def project_gaussians(gaussians, camera_pose, K, image_size):
    """Pure current-view projection; validity means center-frustum/SPD, not occlusion."""
    geometry_valid = validate_gaussians(gaussians)
    T_optical_from_world = _world_to_optical(camera_pose)
    K = validate_intrinsics(K, image_size)
    n = len(gaussians["ids"])
    optical = np.full((n, 3), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        optical[geometry_valid] = transform_points(T_optical_from_world, gaussians["centers_world_m"][geometry_valid])
    uv, depth, in_image = project_points(optical, K, image_size)
    valid = geometry_valid & in_image
    output = dict(ids=gaussians["ids"].copy(), valid_mask=valid,
                  geometry_valid_mask=geometry_valid.copy(), means_uv_px=np.full((n, 2), np.nan),
                  depth_z_m=np.full(n, np.nan), jacobians_uv_per_m=np.full((n, 2, 3), np.nan),
                  covariances_uv_px2=np.full((n, 2, 2), np.nan),
                  opacities=gaussians["opacities"].copy(), colors_rgb=gaussians["colors_rgb"].copy())
    R_optical_from_world = T_optical_from_world[:3, :3]
    for i in np.flatnonzero(valid):
        try:
            J = pinhole_jacobian(optical[i:i + 1], K, image_size)[0]
            with np.errstate(over="ignore", invalid="ignore"):
                optical_covariance = R_optical_from_world @ gaussians["covariances_world_m2"][i] @ R_optical_from_world.T
                screen_covariance = J @ optical_covariance @ J.T
            if (not np.isfinite(screen_covariance).all()
                    or not np.allclose(screen_covariance, screen_covariance.T, atol=1e-12, rtol=0)):
                valid[i] = False
                continue
            eigenvalues = np.linalg.eigvalsh(screen_covariance)
            if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
                valid[i] = False
                continue
        except (ValueError, np.linalg.LinAlgError):
            valid[i] = False
            continue
        output["means_uv_px"][i], output["depth_z_m"][i] = uv[i], depth[i]
        output["jacobians_uv_per_m"][i], output["covariances_uv_px2"][i] = J, screen_covariance
    return output


def static_multiview_observation(gaussians, T_world_from_vehicle, rig, K, image_size):
    """Observe the same supplied static world at one current vehicle pose only."""
    if not isinstance(rig, dict) or set(rig) != set(CAMERA_IDS):
        raise ValueError("Expected front/back/left/right static calibration")
    return {view: project_gaussians(gaussians, current_camera_pose(T_world_from_vehicle, rig[view]), K, image_size)
            for view in CAMERA_IDS}


def transform_world_subset(gaussians, selected_ids, T_world_after_from_world_before):
    """Externally imposed object motion on a copied subset, not learned dynamics."""
    if not validate_gaussians(gaussians).all():
        raise ValueError("Object-motion fixture requires valid finite SPD Gaussians")
    ids = np.asarray(selected_ids)
    if (ids.ndim != 1 or ids.dtype != np.int64 or not len(ids) or len(np.unique(ids)) != len(ids)
            or not np.isin(ids, gaussians["ids"]).all()):
        raise ValueError("Expected unique existing fixture IDs for explicit object motion")
    T = validate_transform(T_world_after_from_world_before)
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in gaussians.items()}
    mask = np.isin(result["ids"], ids)
    result["centers_world_m"][mask] = transform_points(T, result["centers_world_m"][mask])
    R = T[:3, :3]
    result["covariances_world_m2"][mask] = R @ result["covariances_world_m2"][mask] @ R.T
    if not validate_gaussians(result).all():
        raise ValueError("Explicit motion produced invalid geometry")
    return result


def make_static_fixture():
    """Sixteen known synthetic Gaussians; full world is disclosed ORACLE fixture."""
    rng = np.random.default_rng(SEED)
    initial_state = np.array([2., -1., .4])
    T_world_from_vehicle = yaw_transform(initial_state[2], [initial_state[0], initial_state[1], 0.])
    front_rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    rig = {}
    for view, yaw, translation in zip(CAMERA_IDS, [0., np.pi, np.pi / 2, -np.pi / 2],
                                     [[1., 0., .8], [-1., 0., .8], [0., .5, .8], [0., -.5, .8]]):
        T = yaw_transform(yaw, translation)
        T[:3, :3] = T[:3, :3] @ front_rotation
        rig[view] = T
    targets, centers = {}, []
    for view in CAMERA_IDS:
        optical = np.column_stack((rng.uniform(-.8, .8, 4), rng.uniform(-.4, .4, 4), rng.uniform(4., 8., 4)))
        if view == "front":
            optical[3] = [.78 * 6., 0., 6.]  # Center near image edge to exercise mask changes.
        targets[view] = optical.copy()
        centers.extend(transform_points(T_world_from_vehicle @ rig[view], optical))
    A = rng.normal(0., .03, (16, 3, 3))
    covariances = A @ A.transpose(0, 2, 1) + np.diag([.0025, .0064, .01])
    gaussians = dict(provenance=PROVENANCE, ids=np.arange(7000, 7016, dtype=np.int64),
                     centers_world_m=np.asarray(centers), covariances_world_m2=covariances,
                     opacities=rng.uniform(.2, .9, 16), colors_rgb=rng.uniform(0., 1., (16, 3)))
    if not validate_gaussians(gaussians).all():
        raise ValueError("Invalid generated Gaussian fixture")
    return dict(gaussians=gaussians, initial_state=initial_state, T_world_from_vehicle=T_world_from_vehicle,
                rig=rig, K=np.array([[80., 0., 64.], [0., 80., 48.], [0., 0., 1.]]),
                image_size=np.array([128, 96], dtype=np.int64), optical_targets=targets)
