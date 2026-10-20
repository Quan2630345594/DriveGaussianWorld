"""Bounded synthetic current-view visibility oracle, not RGB-to-3D reconstruction.

Only the five fields accepted by validate_observation belong in a model reader.
Diagnostics contain teacher information and must remain on the evaluator side.
"""

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from .gaussians import FEATURE_LAYOUT, pack_gaussians
from ..world_model.gaussian_observation_reference import (
    LOG_KERNEL_CUTOFF, MAX_HEIGHT, MAX_WIDTH, render_reference_observation,
)
from ..world_model.gaussian_state_contract import (
    MAX_GAUSSIANS, PROVENANCE, current_camera_pose, project_gaussians,
)
from ..world_model.geometry import validate_intrinsics, validate_transform


MODEL_FIELDS = frozenset({"features", "gaussian_mask", "source_time_s",
                          "available_at_s", "timestamp_s"})


def _time(value, name, shape):
    value = np.asarray(value)
    if (value.shape not in ((), shape) or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite numeric with shape {shape} or scalar")
    return np.broadcast_to(value, shape).astype(np.float64, copy=True)


def validate_observation(payload):
    """Validate ONLY a minimal model payload; reject metadata/teacher aliases.

    One frame, native neutral14, zero invalid rows, no inferred official mask
    binding. Dataset time/frame axes must be handled explicitly by its reader.
    """
    if not isinstance(payload, dict) or set(payload) != MODEL_FIELDS:
        raise ValueError("model payload must contain exactly the five observation fields")
    features, mask = payload["features"], payload["gaussian_mask"]
    if (not isinstance(features, np.ndarray) or features.ndim != 2
            or features.shape[1] != 14 or features.dtype.kind != "f"
            or not 1 <= len(features) <= MAX_GAUSSIANS
            or not np.isfinite(features).all()):
        raise ValueError("features must be finite floating [Nmax,14], Nmax in [1,32]")
    n = len(features)
    if not isinstance(mask, np.ndarray) or mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError("gaussian_mask must be bool [Nmax]")
    t = _time(payload["timestamp_s"], "timestamp_s", ())
    times = []
    for name in ("source_time_s", "available_at_s"):
        value = payload[name]
        if not isinstance(value, np.ndarray) or value.shape != (n,):
            raise ValueError(f"{name} must be an array [Nmax]")
        times.append(_time(value, name, (n,)))
        if np.any(value[~mask] != 0):
            raise ValueError("masked timestamps must be zero")
    source, available = times
    if np.any(source[mask] > available[mask]) or np.any(available[mask] > t):
        raise ValueError("require source_time_s <= available_at_s <= timestamp_s")
    if np.any(features[~mask] != 0):
        raise ValueError("masked features must be zero")
    rows = features[mask]
    pack_gaussians(dict(means=rows[:, :3], scales=rows[:, 3:6],
                        quats=rows[:, 6:10], colors=rows[:, 10:13], opacities=rows[:, 13]),
                   dtype=np.float64)


def extract_observation(canonical_arrays, current_camera_world, K, image_size,
                        timestamp_s, source_time_s, available_at_s, *,
                        source_classification,
                        provenance="SYNTHETIC_TEACHER_VISIBILITY_ORACLE",
                        max_gaussians=32, contribution_threshold=1e-4):
    """Extract world-meter neutral14 with current-time synthetic availability.

    current_camera_world is T_world_from_optical; size is integer (W,H).
    Teacher source/availability clocks may be scalars or actual per-row clocks;
    every row is checked before visibility, including offscreen/zero-alpha rows.
    Selected observations are newly constructed at t, hence both output clocks
    equal t, not the teacher's earlier source clock. No row truncation/cloning.

    Selection uses max_pixel(T * alpha) > contribution_threshold, not raw mass
    or frustum membership. Diagnostics cover ALL teacher rows in canonical
    content order, never original IDs. Depth is alpha-weighted center Z, not
    physical ground truth. Black-background RGB/alpha use the existing reference.
    """
    if source_classification != "SYNTHETIC_PLANNING_FIXTURE":
        raise ValueError("only explicitly classified SYNTHETIC_PLANNING_FIXTURE is permitted")
    if provenance != "SYNTHETIC_TEACHER_VISIBILITY_ORACLE":
        raise ValueError("unsupported observation provenance")
    if (isinstance(max_gaussians, (bool, np.bool_))
            or not isinstance(max_gaussians, (int, np.integer))
            or not 1 <= max_gaussians <= MAX_GAUSSIANS):
        raise ValueError("max_gaussians must be an integer in [1,32]")
    threshold = _time(contribution_threshold, "contribution_threshold", ())
    if threshold < 0 or threshold > 1:
        raise ValueError("contribution_threshold must be in [0,1]")
    size = np.asarray(image_size)
    if (size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0)
            or size[0] > MAX_WIDTH or size[1] > MAX_HEIGHT):
        raise ValueError("image_size must be integer [W,H], at most 128x96")
    camera = current_camera_pose(validate_transform(current_camera_world), np.eye(4))
    K = validate_intrinsics(K, size)
    rows = pack_gaussians(canonical_arrays, dtype=np.float64)
    n = len(rows)
    if n > MAX_GAUSSIANS:
        raise ValueError("teacher oracle is bounded to 32 Gaussians")
    cov = np.asarray(canonical_arrays.get("covariances"))
    if cov.shape != (n, 3, 3) or cov.dtype.kind not in "fiu" or not np.isfinite(cov).all():
        raise ValueError("finite canonical covariances [N,3,3] are required")
    rotation = Rotation.from_quat(rows[:, [7, 8, 9, 6]]).as_matrix() if n else np.empty((0, 3, 3))
    with np.errstate(over="ignore", invalid="ignore"):
        expected = (rotation * (rows[:, 3:6] ** 2)[:, None, :]) @ rotation.transpose(0, 2, 1)
    if (not np.allclose(cov, cov.transpose(0, 2, 1), atol=0, rtol=1e-10)
            or np.any(np.linalg.eigvalsh(cov) <= 0)
            or not np.isfinite(expected).all()
            or not np.allclose(cov, expected, atol=1e-30, rtol=1e-8)):
        raise ValueError("covariances must be symmetric SPD and agree with scales/unit wxyz")
    t = float(_time(timestamp_s, "timestamp_s", ()))
    source = _time(source_time_s, "source_time_s", (n,))
    available = _time(available_at_s, "available_at_s", (n,))
    if np.any(source > available) or np.any(available > t):
        raise ValueError("teacher clocks require source_time_s <= available_at_s <= timestamp_s")

    q = rows[:, 6:10]
    first = np.argmax(q != 0, axis=1)
    q[q[np.arange(n), first] < 0] *= -1
    rows[rows == 0] = 0.
    # Covariance is independently stored by the scene contract, so it also
    # breaks ties between equal feature rows with different valid roundoff.
    hash_cov = np.array(cov, dtype="<f8", copy=True)
    hash_cov[hash_cov == 0] = 0.
    content = [row.tobytes() + covariance.tobytes() for row, covariance in
               zip(np.ascontiguousarray(rows, dtype="<f8"), hash_cov)]
    order = np.array(sorted(range(n), key=lambda i: (hashlib.sha256(content[i]).digest(), content[i])), dtype=np.int64)
    rows, cov = rows[order], np.asarray(cov[order], dtype=np.float64)
    world = dict(provenance=PROVENANCE, ids=np.arange(n, dtype=np.int64),
                 centers_world_m=rows[:, :3].copy(), covariances_world_m2=cov,
                 opacities=rows[:, 13].copy(), colors_rgb=rows[:, 10:13].copy())
    width, height = map(int, size)
    mass, maximum, raw_mass = np.zeros(n), np.zeros(n), np.zeros(n)
    raw_max = np.zeros(n)
    depth_numerator = np.zeros((height, width))
    transmittance = np.ones(height * width)
    if n:
        reference = render_reference_observation(world, camera, K, size)
        projection = project_gaussians(world, camera, K, size)
        v, u = np.indices((height, width), dtype=np.float64)
        pixels = np.column_stack((u.ravel(), v.ravel()))
        # Reuse the reference's numerical skip decisions; its IDs are content ranks.
        for row in reference["sorted_ids"]:
            factor = np.linalg.cholesky(projection["covariances_uv_px2"][row])
            whitened = np.linalg.solve(factor, (pixels - projection["means_uv_px"][row]).T)
            with np.errstate(over="ignore", invalid="ignore"):
                squared = np.einsum("ij,ij->j", whitened, whitened)
            kernel = np.zeros_like(transmittance)
            active = squared <= -2 * LOG_KERNEL_CUTOFF
            kernel[active] = np.exp(-.5 * squared[active])
            alpha = np.clip(projection["opacities"][row] * kernel, 0., 1.)
            contribution = transmittance * alpha
            mass[row], maximum[row] = contribution.sum(), contribution.max()
            raw_mass[row], raw_max[row] = alpha.sum(), alpha.max()
            depth_numerator += (contribution * projection["depth_z_m"][row]).reshape(height, width)
            transmittance *= 1. - alpha
        projected = projection["valid_mask"].copy()
        renderable = reference["renderable_mask"].copy()
    else:
        reference = dict(rgb=np.zeros((height, width, 3)), alpha=np.zeros((height, width)))
        projected = renderable = np.zeros(0, dtype=bool)
    selected = maximum > threshold
    count = int(selected.sum())
    if count > max_gaussians:
        raise ValueError("max_gaussians is smaller than the visible count; truncation is forbidden")
    features = np.zeros((max_gaussians, 14), dtype=np.float64)
    features[:count] = rows[selected]
    mask = np.arange(max_gaussians) < count
    payload = dict(features=features, gaussian_mask=mask,
                   source_time_s=np.where(mask, t, 0.), available_at_s=np.where(mask, t, 0.),
                   timestamp_s=t)
    validate_observation(payload)
    depth = np.divide(depth_numerator, reference["alpha"], out=np.zeros_like(depth_numerator),
                      where=reference["alpha"] > 0)
    center_depth = (rows[:, :3] - camera["T_world_from_optical"][:3, 3]) @ camera["T_world_from_optical"][:3, 2]
    payload["diagnostics"] = dict(
        rgb=reference["rgb"], alpha=reference["alpha"], depth_z=depth, selected_count=count,
        per_gaussian=dict(visible_mass=mass, max_visible_contribution=maximum,
                          raw_mass=raw_mass, max_raw_contribution=raw_max,
                          center_depth_z_m=center_depth, projected_valid_mask=projected,
                          renderable_mask=renderable, selected_mask=selected))
    payload["metadata"] = dict(
        provenance=provenance, source_classification=source_classification,
        format="vendor-neutral14", layout=FEATURE_LAYOUT.copy(), coordinate_frame="world",
        units={"means": "meters", "scales": "meters", "depth_z": "meters", "timestamps": "seconds"},
        domains={"scales": "activated_positive", "quaternion": "unit_wxyz_canonical_sign",
                 "colors": "activated_rgb_0_1_NOT_SH", "opacity": "activated_0_1"},
        visibility="APPROXIMATE_CENTER_Z_ALPHA_COMPOSITING", current_camera_only=True,
        rgb_to_3d_reconstruction=False, teacher_oracle=True, diagnostics_model_input=False,
        depth_semantics="per_pixel_alpha_weighted_center_Z_NOT_PHYSICAL_GT_zero_if_empty",
        threshold_statistic="max_pixel_T_times_alpha_strictly_greater", contribution_threshold=float(threshold),
        timestamp_policy="synthetic_observation_constructed_at_t; caller_teacher_clocks_checked_all_rows",
        ordering="center_Z_then_sha256_canonical_content_rank; output_content_rank",
        padding="zero_invalid_first_party_mask_only", official_sh_feed=False,
        official_mask_binding="OFFICIAL_MASK_BINDING_PENDING")
    return payload
