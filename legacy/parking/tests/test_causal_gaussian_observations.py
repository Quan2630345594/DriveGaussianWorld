"""CPU-only tests of the synthetic oracle and its strict model boundary."""

import numpy as np
import pytest

from gaussian_park_world.gwm_adapter.observations import (
    MODEL_FIELDS, extract_observation, validate_observation,
)
from gaussian_park_world.scenes.contract import normalize_gaussians
from gaussian_park_world.world_model.gaussian_observation_reference import render_reference_observation
from gaussian_park_world.world_model.gaussian_state_contract import PROVENANCE, current_camera_pose


K = np.array([[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]])
SIZE = (9, 7)


def scene(means=((0., 0., 2.), (0., 0., 4.)), scales=None, opacities=None):
    means = np.asarray(means, dtype=float).reshape(-1, 3)
    n = len(means)
    if n == 0:
        return dict(means=means, scales=np.empty((0, 3)), quats=np.empty((0, 4)),
                    covariances=np.empty((0, 3, 3)), colors=np.empty((0, 3)), opacities=np.empty(0))
    return normalize_gaussians(
        means, np.full((n, 3), .02) if scales is None else scales,
        np.tile([1., 0., 0., 0.], (n, 1)),
        np.tile([[1., .25, .125], [.125, .5, 1.]], (n, 1))[:n],
        np.ones(n) if opacities is None else opacities)


def extract(arrays=None, **kwargs):
    options = dict(source_classification="SYNTHETIC_PLANNING_FIXTURE")
    options.update(kwargs)
    return extract_observation(scene() if arrays is None else arrays,
                               options.pop("camera", np.eye(4)), options.pop("K", K),
                               options.pop("image_size", SIZE), options.pop("timestamp_s", 5.),
                               options.pop("source_time_s", 1.), options.pop("available_at_s", 2.),
                               **options)


def minimal(result):
    return {key: result[key] for key in MODEL_FIELDS}


def test_coincident_opaque_near_hides_far_despite_both_frustum_candidates():
    result = extract()
    diagnostic = result["diagnostics"]
    per = diagnostic["per_gaussian"]
    near, far = np.argsort(per["center_depth_z_m"])
    assert per["projected_valid_mask"].all()
    assert per["raw_mass"][far] >= 1.
    assert per["max_visible_contribution"][near] == 1.
    assert per["max_visible_contribution"][far] < 1e-4
    assert diagnostic["selected_count"] == 1
    assert result["features"][0, 2] == 2.


def test_opaque_grid_suppresses_tiny_far_contribution_not_only_center():
    # Six opaque foreground splats cover every integer pixel in this tiny view.
    means = [(u, v, 2.) for v in range(2) for u in range(3)] + [(1., .5, 4.)]
    k = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    scales = np.full((7, 3), .3)
    scales[-1] = .5
    result = extract(scene(means, scales=scales), K=k, image_size=(3, 2))
    per = result["diagnostics"]["per_gaussian"]
    far = np.argmax(per["center_depth_z_m"])
    assert per["projected_valid_mask"].all()
    assert per["raw_mass"][far] > 0
    assert per["visible_mass"][far] == 0
    assert not per["selected_mask"][far]


def test_rgb_alpha_and_weighted_depth_match_actual_reference_unique_z():
    arrays = scene(scales=np.array([[.4, .3, .2], [.9, .6, .2]]), opacities=[.8, .7])
    world = dict(provenance=PROVENANCE, ids=np.array([91, 2], dtype=np.int64),
                 centers_world_m=arrays["means"], covariances_world_m2=arrays["covariances"],
                 colors_rgb=arrays["colors"], opacities=arrays["opacities"])
    reference = render_reference_observation(world, current_camera_pose(np.eye(4), np.eye(4)), K, SIZE)
    result = extract(arrays)
    diagnostic = result["diagnostics"]
    np.testing.assert_array_equal(diagnostic["rgb"], reference["rgb"])
    np.testing.assert_array_equal(diagnostic["alpha"], reference["alpha"])
    assert np.all(diagnostic["depth_z"][reference["alpha"] > 0] >= 2. - 1e-10)
    assert np.all(diagnostic["depth_z"] <= 4. + 1e-10)
    np.testing.assert_allclose(diagnostic["per_gaussian"]["visible_mass"].sum(), reference["alpha"].sum())


def test_equal_depth_permutation_quaternion_sign_and_unused_ids_are_invariant():
    arrays = scene(means=((0., 0., 2.), (0., 0., 2.)),
                   scales=np.full((2, 3), .3), opacities=[.8, .7])
    arrays["ids"] = np.array([999, -99])
    before = {key: value.copy() for key, value in arrays.items()}
    other = {key: value[::-1].copy() for key, value in arrays.items()}
    other["quats"] *= -1
    other["ids"][:] = [17, 12]
    left, right = extract(arrays), extract(other)
    for key in MODEL_FIELDS:
        np.testing.assert_array_equal(left[key], right[key])
    for key in ("rgb", "alpha", "depth_z"):
        np.testing.assert_array_equal(left["diagnostics"][key], right["diagnostics"][key])
    for key in left["diagnostics"]["per_gaussian"]:
        np.testing.assert_array_equal(left["diagnostics"]["per_gaussian"][key],
                                      right["diagnostics"]["per_gaussian"][key])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


def test_padding_clocks_metadata_and_minimal_reader():
    result = extract(source_time_s=np.array([0., 1.]), available_at_s=np.array([1., 3.]))
    mask = result["gaussian_mask"]
    assert result["features"].shape == (32, 14)
    assert mask.shape == (32,) and mask.dtype == bool
    assert set(result) == MODEL_FIELDS | {"metadata", "diagnostics"}
    assert np.all(result["features"][~mask] == 0)
    for key in ("source_time_s", "available_at_s"):
        assert result[key].shape == (32,)
        assert np.all(result[key][mask] == 5.)
        assert np.all(result[key][~mask] == 0.)
    validate_observation(minimal(result))
    with pytest.raises(ValueError, match="exactly"):
        validate_observation(result)
    assert result["metadata"]["official_mask_binding"] == "OFFICIAL_MASK_BINDING_PENDING"
    assert not result["metadata"]["official_sh_feed"]
    assert not result["metadata"]["rgb_to_3d_reconstruction"]
    assert result["metadata"]["coordinate_frame"] == "world"


@pytest.mark.parametrize("arrays", [scene(opacities=[0., 0.]), scene(means=((0., 0., -2.),)), scene(means=())])
def test_empty_observations_are_valid(arrays):
    result = extract(arrays)
    assert not result["gaussian_mask"].any()
    assert result["diagnostics"]["selected_count"] == 0
    assert np.all(result["diagnostics"]["alpha"] == 0)
    assert np.all(result["diagnostics"]["depth_z"] == 0)
    validate_observation(minimal(result))


@pytest.mark.parametrize("options", [
    {"source_time_s": [1., 6.]}, {"available_at_s": [2., 6.]},
    {"source_time_s": [3., 1.], "available_at_s": [2., 2.]},
    {"source_time_s": None}, {"available_at_s": "unknown"},
    {"timestamp_s": np.nan}, {"source_time_s": [1., np.inf]},
    {"source_classification": "REAL"}, {"source_classification": "UNKNOWN"},
    {"provenance": "REAL_RGB"},
])
def test_future_or_unknown_poison_rejected_even_for_hidden_zero_alpha(options):
    with pytest.raises(ValueError):
        extract(scene(opacities=[1., 0.]), **options)


@pytest.mark.parametrize("fault", ["asymmetric", "indefinite", "inconsistent", "quaternion", "scale", "nan"])
def test_malformed_geometry_never_silently_skipped_offscreen(fault):
    arrays = scene(means=((0., 0., -2.),))
    if fault == "asymmetric":
        arrays["covariances"][0, 0, 1] = .1
    elif fault == "indefinite":
        arrays["covariances"][0, 0, 0] = -1
    elif fault == "inconsistent":
        arrays["covariances"] *= 2
    elif fault == "quaternion":
        arrays["quats"] *= 2
    elif fault == "scale":
        arrays["scales"] *= -1
    else:
        arrays["means"][0, 0] = np.nan
    with pytest.raises(ValueError):
        extract(arrays)


def test_numeric_screen_skip_matches_reference():
    arrays = scene(means=((0., 0., 2.),), scales=[[1e-8, 1e-8, 1e-8]])
    result = extract(arrays, contribution_threshold=0.)
    assert result["diagnostics"]["per_gaussian"]["projected_valid_mask"].all()
    assert not result["diagnostics"]["per_gaussian"]["renderable_mask"].any()
    assert not result["gaussian_mask"].any()


@pytest.mark.parametrize("options", [
    {"max_gaussians": 0}, {"max_gaussians": 33}, {"max_gaussians": True},
    {"image_size": (129, 96)}, {"image_size": (9., 7.)},
    {"camera": np.diag([-1., 1., 1., 1.])}, {"camera": np.zeros((2, 4, 4))},
    {"K": np.array([[8., 1., 4.], [0., 8., 3.], [0., 0., 1.]])},
    {"contribution_threshold": -1.}, {"contribution_threshold": np.inf},
])
def test_native_bounds_and_current_camera_validation(options):
    with pytest.raises(ValueError):
        extract(**options)


def test_insufficient_capacity_rejected_not_truncated_and_teacher_count_bounded():
    with pytest.raises(ValueError, match="truncation"):
        extract(scene(means=((-0.5, 0., 2.), (.5, 0., 2.))), max_gaussians=1)
    with pytest.raises(ValueError, match="bounded"):
        extract(scene(means=np.tile([0., 0., 2.], (33, 1))))


@pytest.mark.parametrize("fault", ["future_pose", "features_padding", "time_padding", "future", "mask", "quaternion"])
def test_model_validator_rejects_leakage_and_malformed_payload(fault):
    payload = minimal(extract())
    if fault == "future_pose":
        payload["future_poses"] = np.eye(4)
    elif fault == "features_padding":
        payload["features"][-1, 0] = 1.
    elif fault == "time_padding":
        payload["source_time_s"][-1] = 1.
    elif fault == "future":
        payload["available_at_s"][0] = 6.
    elif fault == "mask":
        payload["gaussian_mask"] = payload["gaussian_mask"].astype(int)
    else:
        payload["features"][0, 6:10] *= 2
    with pytest.raises(ValueError):
        validate_observation(payload)


def test_synthetic_classification_is_mandatory_not_inferred_from_provenance():
    with pytest.raises(TypeError, match="source_classification"):
        extract_observation(scene(), np.eye(4), K, SIZE, 5., 1., 2.)


def test_anisotropic_wxyz_rotation_and_world_camera_units():
    angle = .7
    q = np.array([[np.cos(angle / 2), 0., 0., np.sin(angle / 2)]])
    arrays = normalize_gaussians([[10., -3., 4.]], [[.3, .1, .2]], q,
                                 [[.2, .4, .8]], [.9])
    camera = np.eye(4)
    camera[:3, 3] = [10., -3., 2.]
    camera[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    result = extract(arrays, camera=camera)
    row = result["features"][result["gaussian_mask"]][0]
    np.testing.assert_array_equal(row[:3], [10., -3., 4.])
    np.testing.assert_allclose(row[3:6], [.3, .1, .2])
    np.testing.assert_allclose(row[6:10], q[0])
    np.testing.assert_array_equal(result["diagnostics"]["per_gaussian"]["center_depth_z_m"], [2.])


def test_covariance_roundoff_breaks_equal_feature_ties_without_input_row_ids():
    arrays = scene(means=((0., 0., 2.), (0., 0., 2.)), scales=np.full((2, 3), .3))
    arrays["colors"][1] = arrays["colors"][0]
    arrays["covariances"][1] *= 1. + 1e-9
    other = {key: value[::-1].copy() for key, value in arrays.items()}
    left, right = extract(arrays), extract(other)
    for key in ("rgb", "alpha", "depth_z"):
        np.testing.assert_array_equal(left["diagnostics"][key], right["diagnostics"][key])
    for key in left["diagnostics"]["per_gaussian"]:
        np.testing.assert_array_equal(left["diagnostics"]["per_gaussian"][key],
                                      right["diagnostics"]["per_gaussian"][key])


def test_native_max_image_size_and_all_32_uncloned_rows():
    arrays = scene(means=[((i % 8 - 3.5) * .2, (i // 8 - 1.5) * .2, 2.) for i in range(32)])
    k = np.array([[80., 0., 64.], [0., 80., 48.], [0., 0., 1.]])
    result = extract(arrays, K=k, image_size=(128, 96))
    assert result["gaussian_mask"].all()
    assert result["diagnostics"]["rgb"].shape == (96, 128, 3)
    assert len(np.unique(result["features"][:, :3], axis=0)) == 32
    validate_observation(minimal(result))
