"""Tiny CPU 2D Gaussian-opacity reference, not the production 3DGS renderer."""

import numpy as np

from .gaussian_state_contract import project_gaussians


MAX_WIDTH, MAX_HEIGHT = 128, 96
MAX_CONDITION = 1e10
MIN_VARIANCE_PX2 = 1e-12
LOG_KERNEL_CUTOFF = -700.
RANGE_ATOL = 1e-12


def render_reference_observation(gaussians, camera_pose, K, image_size, background_rgb=(0., 0., 0.),
                                 *, max_condition=MAX_CONDITION, min_variance_px2=MIN_VARIANCE_PX2):
    """One current camera; integer pixel centers, near-to-far center-Z sorting.

    Invalid/offscreen centers follow Tier 5 masks. Numerically unsuitable screen
    covariances are skipped explicitly, without covariance jitter. Returned debug
    masks are in input-row order; sorted_ids uses depth then unique fixture ID.
    """
    size = np.asarray(image_size)
    if (size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0)
            or size[0] > MAX_WIDTH or size[1] > MAX_HEIGHT):
        raise ValueError("Reference image must be positive integer [W,H], at most 128x96")
    background = np.asarray(background_rgb)
    if (background.shape != (3,) or background.dtype.kind not in "fiu"
            or not np.isfinite(background).all() or np.any(background < 0) or np.any(background > 1)):
        raise ValueError("Background must be finite real RGB[3] in [0,1]")
    background = background.astype(np.float64, copy=False)
    for setting, minimum in ((max_condition, 1.), (min_variance_px2, np.nextafter(0., 1.))):
        if (not np.isscalar(setting) or isinstance(setting, (bool, np.bool_))
                or not np.isfinite(setting) or setting < minimum):
            raise ValueError("Invalid finite condition/positive variance threshold")

    projection = project_gaussians(gaussians, camera_pose, K, size)
    width, height = map(int, size)
    v, u = np.indices((height, width), dtype=np.float64)
    pixels = np.column_stack((u.ravel(), v.ravel()))
    color = np.zeros((height * width, 3), dtype=np.float64)
    transmittance = np.ones(height * width, dtype=np.float64)
    renderable = np.zeros(len(projection["ids"]), dtype=bool)
    rows = np.flatnonzero(projection["valid_mask"])
    order = rows[np.lexsort((projection["ids"][rows], projection["depth_z_m"][rows]))]
    sorted_ids, skipped = [], []
    for row in order:
        covariance = projection["covariances_uv_px2"][row]
        try:
            if (not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)):
                raise ValueError("Invalid screen covariance")
            eigenvalues = np.linalg.eigvalsh(covariance)
            if not np.isfinite(eigenvalues).all() or eigenvalues[0] < min_variance_px2:
                raise ValueError("Screen covariance below declared variance threshold")
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                condition = eigenvalues[-1] / eigenvalues[0]
            if not np.isfinite(condition) or condition > max_condition:
                raise ValueError("Ill-conditioned screen covariance")
            factor = np.linalg.cholesky(covariance)
            delta = pixels - projection["means_uv_px"][row]
            whitened = np.linalg.solve(factor, delta.T)
            with np.errstate(over="ignore", invalid="ignore"):
                mahalanobis_sq = np.einsum("ij,ij->j", whitened, whitened)
            if not np.isfinite(mahalanobis_sq).all():
                raise ValueError("Nonfinite screen kernel")
        except (ValueError, np.linalg.LinAlgError):
            skipped.append(projection["ids"][row])
            continue
        renderable[row] = True
        sorted_ids.append(projection["ids"][row])
        kernel = np.zeros_like(transmittance)
        active = mahalanobis_sq <= -2 * LOG_KERNEL_CUTOFF
        kernel[active] = np.exp(-.5 * mahalanobis_sq[active])
        alpha = np.clip(projection["opacities"][row] * kernel, 0., 1.)
        color += (transmittance * alpha)[:, None] * projection["colors_rgb"][row]
        transmittance *= 1. - alpha
    color += transmittance[:, None] * background
    accumulated_alpha = 1. - transmittance
    for value in (color, transmittance, accumulated_alpha):
        if not np.isfinite(value).all() or np.any(value < -RANGE_ATOL) or np.any(value > 1. + RANGE_ATOL):
            raise ValueError("Nonfinite or out-of-range reference composition")
    return dict(rgb=color.reshape(height, width, 3), alpha=accumulated_alpha.reshape(height, width),
                transmittance=transmittance.reshape(height, width),
                projected_valid_mask=projection["valid_mask"].copy(), renderable_mask=renderable,
                sorted_ids=np.array(sorted_ids, dtype=np.int64), skipped_screen_ids=np.array(skipped, dtype=np.int64))
