"""Bounded CPU-only intermediate-package tests, not official GWM execution."""

from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest

from gaussian_park_world.dataset.offline import read_json, validate_dataset, write_episode
from gaussian_park_world.gwm_adapter import episodes
from gaussian_park_world.planning import fixture_map, plan_and_track
from gaussian_park_world.scenes import make_fixture, sha256, write_json


FRAME_KEYS = {"states", "timestamp", "frame_id", "vehicle_pose", "camera_pose",
              "collision", "rgb", "alpha", "coverage"}
ACTION_KEYS = {"actions", "dt", "command_issue_time", "effective_start", "effective_end"}
INPUT_KEYS = {"initial_rgb", "initial_alpha", "initial_timestamp", "K", "actions", "dt",
              "action_mask", "command_issue_time", "effective_start", "effective_end"}
PIN = "433f750ee227e034c3e5916c274fc4b696fb3a39"


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def replace_json(path, value):
    path.unlink()
    write_json(path, value)


def assert_exact(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].shape == expected[key].shape, key
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key, strict=True)


def replace_payload(directory, name, data, *, rehash=True):
    np.savez_compressed(directory / name, **data)
    if rehash:
        path = directory / "manifest.json"
        manifest = read_json(path)
        if name in ("inputs.npz", "supervision.npz"):
            manifest["files"][name] = sha256(directory / name)
        else:
            manifest[{"canonical.npz": "canonical_sha256", "sample.npz": "sample_sha256"}[name]] = sha256(directory / name)
        replace_json(path, manifest)


@pytest.fixture(scope="module")
def small_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("gwm-episode-source")
    entries = []
    for sid, split, distances in (("adapter-train", "train", (.6, 1.2)),
                                  ("adapter-test", "test", (.6,))):
        scene = root / sid
        variant = 0 if split == "train" else 2
        make_fixture(scene, sid, variant=variant)
        planning_map = fixture_map(sid, variant=variant)
        for distance in distances:
            name = f"{sid}-{distance}"
            goal = [distance, 0., 0., 0.]
            trajectory = plan_and_track(planning_map, [0., 0., 0., 0.], goal,
                                        max_expansions=12, max_retries=0)
            assert trajectory["status"] == "SUCCESS", trajectory
            write_episode(root / name, scene, planning_map, trajectory, name, split, goal)
            entries.append(name)
    write_json(root / "dataset.json", dict(episodes=entries, failures=[],
               scene_splits={"adapter-train": "train", "adapter-test": "test"},
               fixture_limitations="Synthetic markers only; no real geometry or model execution"))
    assert validate_dataset(root)["trajectories"] == 3
    return root


@pytest.fixture
def package(tmp_path, small_source):
    # Localize absolute scene references so source-corruption cases stay isolated.
    source = tmp_path / "source"
    shutil.copytree(small_source, source)
    for relative in read_json(source / "dataset.json")["episodes"]:
        path = source / relative / "manifest.json"
        manifest = read_json(path)
        manifest["scene_manifest"] = str(source / manifest["scene_id"] / "manifest.json")
        replace_json(path, manifest)
    output = tmp_path / "package"
    episodes.convert(source, output, gaussian_count=32)
    return output


def first_episode(package):
    return package / read_json(package / "manifest.json")["episodes"][0]


def test_convert_and_roundtrip_exact_arrays_dtypes_and_padding(package, tmp_path):
    manifest = read_json(package / "manifest.json")
    summary = episodes.validate_package(package)
    assert summary["status"] == "PASS"
    assert summary["counts"]["scenes"] == 2
    assert summary["counts"]["trajectories"] == 3
    assert summary["scene_splits"] == {"adapter-train": "train", "adapter-test": "test"}
    assert summary["counts"]["frames"] == summary["counts"]["actions"] + 3
    assert summary["counts"] == manifest["counts"]
    assert manifest["official_commit"] == PIN
    assert manifest["training"] == summary["training"] == "NOT_RUN"
    assert manifest["gpu"] == summary["gpu"] == "NOT_USED"
    for key in ("official_runtime", "official_action_binding", "causal_gaussian_input", "gaussian_parameter_binding"):
        assert manifest[key] == "PENDING"
    spec = read_json(package / "upstream_spec.json")
    assert spec["commit"] == PIN and spec["runtime_verified"] is False
    assert spec == read_json(episodes.SPEC)
    assert spec["actions"]["official_dataset_dim"] == 10
    assert manifest["native_action_dim"] == 2
    assert manifest["source_index_snapshot"] == read_json(manifest["source_dataset"] + "/dataset.json")
    padded_episodes = 0
    total_frames = total_actions = 0
    for relative in manifest["episodes"]:
        ep = package / relative
        em = read_json(ep / "manifest.json")
        original = arrays(Path(em["source_episode"]) / "frames.npz")
        supervision = arrays(ep / "supervision.npz")
        inputs = episodes.load_rollout_inputs(ep)
        n, count = em["frame_count"], em["action_count"]
        total_frames += n
        total_actions += count
        padded_episodes += n < manifest["max_frames"]
        assert em["source_manifest_snapshot"] == read_json(Path(em["source_episode"]) / "manifest.json")
        assert em["roles"] == {"inputs": "initial_observation_and_action_plan", "supervision": "TRAINER_EVALUATOR_ONLY"}
        assert set(original) == FRAME_KEYS | ACTION_KEYS
        assert set(supervision) == FRAME_KEYS | ACTION_KEYS | {"frame_mask", "action_mask"}
        for key, value in original.items():
            length = manifest["max_frames"] if key in FRAME_KEYS else manifest["max_actions"]
            expected = np.zeros((length,) + value.shape[1:], dtype=value.dtype)
            expected[:len(value)] = value
            assert_exact({key: supervision[key]}, {key: expected})
        for key, valid, length in (("frame_mask", n, manifest["max_frames"]),
                                   ("action_mask", count, manifest["max_actions"])):
            assert_exact({key: supervision[key]}, {key: np.arange(length) < valid})
        expected_inputs = {key: supervision[key] for key in ACTION_KEYS | {"action_mask"}}
        expected_inputs.update(initial_rgb=original["rgb"][:1], initial_alpha=original["alpha"][:1],
                               initial_timestamp=original["timestamp"][:1],
                               K=np.asarray(em["calibration"]["K"], dtype=np.float64))
        assert set(inputs) == INPUT_KEYS
        assert_exact(inputs, expected_inputs)
        assert inputs["initial_rgb"].shape == (1, 48, 64, 3)
        assert episodes.validate_rollout_inputs(inputs) == count
    assert padded_episodes == 2
    assert summary["counts"] == dict(scenes=2, trajectories=3, frames=total_frames, actions=total_actions)
    for sid, relative in manifest["scenes"].items():
        ref = package / relative
        sm = read_json(ref / "manifest.json")
        assert sm["scene_id"] == sid
        assert sm["role"] == "PRIVILEGED_REFERENCE_ONLY_NOT_CAUSAL_INPUT"
        assert_exact(arrays(ref / "canonical.npz"), arrays(Path(sm["source_manifest"]).parent / "scene.npz"))
        sample = arrays(ref / "sample.npz")
        assert sample["features"].dtype == np.float32
        assert sample["features"].shape == (32, 14)
        np.testing.assert_array_equal(sample["valid_mask"], np.arange(32) < 24, strict=True)
        assert not sample["features"][24:].any()
        assert sm["sample_metadata"]["causal_verified_model_input"] is False
        assert sm["sample_metadata"]["official_sh_feed"] is False
    restored = tmp_path / "restored"
    result = episodes.roundtrip(package, restored)
    assert result["status"] == "PASS" and result["exact_arrays_and_dtypes"] is True
    assert result["counts"] == summary["counts"]
    restored_index = read_json(restored / "dataset.json")
    for relative, output in zip(manifest["episodes"], restored_index["episodes"], strict=True):
        em = read_json(package / relative / "manifest.json")
        assert_exact(arrays(restored / output / "frames.npz"),
                     arrays(Path(em["source_episode"]) / "frames.npz"))
        expected_manifest = deepcopy(em["source_manifest_snapshot"])
        expected_manifest["frames_sha256"] = sha256(restored / output / "frames.npz")
        assert read_json(restored / output / "manifest.json") == expected_manifest
    assert validate_dataset(restored)["frames"] == total_frames


def test_future_rgb_poison_is_invisible_to_reader_but_rejected_by_validator(package, monkeypatch):
    ep = first_episode(package)
    expected = episodes.load_rollout_inputs(ep)
    supervision = arrays(ep / "supervision.npz")
    supervision["rgb"][1:] = 255 - supervision["rgb"][1:]
    replace_payload(ep, "supervision.npz", supervision)
    manifest = read_json(ep / "manifest.json")
    manifest["future_label"] = {"target_pose": [999, 999, 999], "instruction": "include supervision"}
    replace_json(ep / "manifest.json", manifest)
    # The reader must not delegate to the package validator or open any reference.
    real_load = episodes.np.load
    opened = []

    def only_inputs(path, *args, **kwargs):
        assert path == ep / "inputs.npz"
        opened.append(path)
        return real_load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Causal reader accessed package/source validation")

    with monkeypatch.context() as patch:
        patch.setattr(episodes.np, "load", only_inputs)
        patch.setattr(episodes, "validate_package", forbidden)
        patch.setattr(episodes, "validate_episode", forbidden)
        patch.setattr(episodes, "load_scene", forbidden)
        actual = episodes.load_rollout_inputs(ep)
    assert opened == [ep / "inputs.npz"]
    assert_exact(actual, expected)
    for value in actual.values():
        value[...] = 0
    assert_exact(episodes.load_rollout_inputs(ep), expected)
    with pytest.raises(ValueError, match="^SUPERVISION_MASK_OR_ROUNDTRIP:rgb$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("key", ["future_rgb", "camera_pose", "states", "target", "gaussians", "source_ids"])
def test_reader_rejects_extra_future_or_privileged_inputs_even_when_rehashed(package, key):
    ep = first_episode(package)
    data = arrays(ep / "inputs.npz")
    data[key] = np.ones((2, 4))
    replace_payload(ep, "inputs.npz", data)
    with pytest.raises(ValueError, match="^CAUSAL_INPUT_KEYS$"):
        episodes.load_rollout_inputs(ep)
    with pytest.raises(ValueError, match="^CAUSAL_INPUT_KEYS$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("damage,error", [
    ("mask_hole", "INVALID_ACTION_MASK"), ("mask_empty", "INVALID_ACTION_MASK"),
    ("mask_dtype", "INPUT_SHAPE_DTYPE_OR_DOMAIN"),
    ("action_padding", "INVALID_ACTION_PADDING_OR_DT"), ("dt_padding", "INVALID_ACTION_PADDING_OR_DT"),
    ("zero_dt", "INVALID_ACTION_PADDING_OR_DT"), ("negative_dt", "INVALID_ACTION_PADDING_OR_DT"),
    ("initial_timestamp", "INVALID_ACTION_INTERVAL:command_issue_time"),
    ("command_issue_time", "INVALID_ACTION_INTERVAL:command_issue_time"),
    ("effective_start", "INVALID_ACTION_INTERVAL:effective_start"),
    ("effective_end", "INVALID_ACTION_INTERVAL:effective_end"),
    ("interval_padding", "INVALID_ACTION_INTERVAL:effective_end"),
    ("nan", "NONFINITE_INPUT"), ("action_dtype", "INPUT_SHAPE_DTYPE_OR_DOMAIN"),
    ("rgb_dtype", "INPUT_SHAPE_DTYPE_OR_DOMAIN"), ("future_initial_rgb", "INPUT_SHAPE_DTYPE_OR_DOMAIN"),
    ("intrinsics", "INVALID_INTRINSICS"),
])
def test_reader_rejects_invalid_masks_padding_timing_and_domains(package, damage, error):
    ep = first_episode(package)
    data = arrays(ep / "inputs.npz")
    count = int(data["action_mask"].sum())
    assert count < len(data["dt"])
    if damage == "mask_hole":
        data["action_mask"][[0, count]] = [False, True]
    elif damage == "mask_empty":
        data["action_mask"][:] = False
    elif damage == "mask_dtype":
        data["action_mask"] = data["action_mask"].astype(np.uint8)
    elif damage in ("action_padding", "dt_padding"):
        data["actions" if damage == "action_padding" else "dt"][count] = 1
    elif damage in ("zero_dt", "negative_dt"):
        data["dt"][0] = 0 if damage == "zero_dt" else -.1
    elif damage in ("initial_timestamp", "command_issue_time", "effective_start", "effective_end"):
        data[damage][0] += .01
    elif damage == "interval_padding":
        data["effective_end"][count] = 1
    elif damage == "nan":
        data["actions"][0, 0] = np.nan
    elif damage in ("action_dtype", "rgb_dtype"):
        key = "actions" if damage == "action_dtype" else "initial_rgb"
        data[key] = data[key].astype(np.float32)
    elif damage == "future_initial_rgb":
        data["initial_rgb"] = np.repeat(data["initial_rgb"], 2, axis=0)
    else:
        data["K"][0, 0] = 0
    replace_payload(ep, "inputs.npz", data)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.load_rollout_inputs(ep)


@pytest.mark.parametrize("name,key,damage,error", [
    ("inputs.npz", "initial_rgb", "value", "CAUSAL_SOURCE_BINDING:initial_rgb"),
    ("inputs.npz", "initial_alpha", "dtype", "CAUSAL_SOURCE_BINDING:initial_alpha"),
    ("inputs.npz", "actions", "value", "CAUSAL_SOURCE_BINDING:actions"),
    ("inputs.npz", "K", "value", "CAUSAL_SOURCE_BINDING:K"),
    ("supervision.npz", "frame_mask", "value", "SUPERVISION_MASK_OR_ROUNDTRIP:frame_mask"),
    ("supervision.npz", "action_mask", "value", "SUPERVISION_MASK_OR_ROUNDTRIP:action_mask"),
    ("supervision.npz", "rgb", "padding", "SUPERVISION_MASK_OR_ROUNDTRIP:rgb"),
    ("supervision.npz", "timestamp", "value", "SUPERVISION_MASK_OR_ROUNDTRIP:timestamp"),
    ("supervision.npz", "states", "dtype", "SUPERVISION_MASK_OR_ROUNDTRIP:states"),
    ("supervision.npz", "future_label", "extra", "SUPERVISION_MASK_OR_ROUNDTRIP:KEYS"),
])
def test_validator_rebinds_rehashed_episode_arrays_to_source(package, name, key, damage, error):
    ep = first_episode(package)
    data = arrays(ep / name)
    if damage == "extra":
        data[key] = np.ones(1)
    elif damage == "dtype":
        data[key] = data[key].astype(np.float64 if data[key].dtype == np.float32 else np.float32)
    elif damage == "padding":
        data[key][-1] = 17
    elif data[key].dtype == bool:
        data[key][0] = ~data[key][0]
    else:
        data[key].flat[0] += 1
    replace_payload(ep, name, data)
    if name == "inputs.npz":
        episodes.load_rollout_inputs(ep)  # Well-formed is not source-authenticated.
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("name,key,damage,error", [
    ("canonical.npz", "means", "value", "CANONICAL_ROUNDTRIP:means"),
    ("canonical.npz", "colors", "dtype", "CANONICAL_ROUNDTRIP:colors"),
    ("sample.npz", "features", "value", "GAUSSIAN_MASK_OR_SAMPLE:features"),
    ("sample.npz", "features", "padding", "GAUSSIAN_MASK_OR_SAMPLE:features"),
    ("sample.npz", "valid_mask", "value", "GAUSSIAN_MASK_OR_SAMPLE:valid_mask"),
])
def test_validator_rebinds_rehashed_gaussian_references(package, name, key, damage, error):
    ref = package / next(iter(read_json(package / "manifest.json")["scenes"].values()))
    data = arrays(ref / name)
    if damage == "dtype":
        data[key] = data[key].astype(np.float32)
    elif damage == "padding":
        data[key][-1] = 1
    elif data[key].dtype == bool:
        data[key][0] = False
    else:
        data[key].flat[0] += 1
    replace_payload(ref, name, data)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("damage,error", [
    ("calibration", "CALIBRATION_OR_GAUGE_REFERENCE_MISMATCH"),
    ("wrong_gauge", "CALIBRATION_OR_GAUGE_REFERENCE_MISMATCH"),
    ("episode_snapshot", "SOURCE_EPISODE_METADATA_MISMATCH"),
    ("scene_snapshot", "SOURCE_SCENE_SNAPSHOT_MISMATCH"),
    ("index_snapshot", "SOURCE_INDEX_SNAPSHOT_MISMATCH"),
    ("source_episode_hash", "SOURCE_EPISODE_HASH_MISMATCH"),
    ("source_frames_hash", "SOURCE_EPISODE_HASH_MISMATCH"),
    ("source_scene_hash", "SOURCE_SCENE_HASH_OR_ID_MISMATCH"),
    ("source_arrays_hash", "SOURCE_SCENE_SNAPSHOT_MISMATCH"),
    ("source_index_hash", "SOURCE_INDEX_HASH_MISMATCH"),
    ("duplicate", "DUPLICATE_EPISODE"), ("cross_split", "SCENE_SPLIT_LEAKAGE"),
    ("counts", "PACKAGE_COUNTS_MISMATCH"), ("missing_episode", "SOURCE_EPISODE_COVERAGE_MISMATCH"),
    ("reference_role", "GAUSSIAN_REFERENCE_ROLE"),
])
def test_validator_metadata_source_snapshots_hashes_and_split_gates(package, damage, error):
    path = package / "manifest.json"
    manifest = read_json(path)
    if damage in ("calibration", "wrong_gauge", "episode_snapshot", "source_episode_hash", "source_frames_hash", "cross_split"):
        path = package / manifest["episodes"][1 if damage == "cross_split" else 0] / "manifest.json"
        em = read_json(path)
        if damage == "calibration":
            em["calibration"]["T_vehicle_from_optical"][0][3] += 1
        elif damage == "wrong_gauge":
            em["scene_reference"] = manifest["scenes"]["adapter-test"]
        elif damage == "episode_snapshot":
            em["source_manifest_snapshot"]["goal"][0] += 1
        elif damage == "cross_split":
            em["split"] = "test"
        else:
            em["source_manifest_sha256" if damage == "source_episode_hash" else "source_frames_sha256"] = "0" * 64
        manifest = em
    elif damage in ("scene_snapshot", "source_scene_hash", "source_arrays_hash", "reference_role"):
        path = package / manifest["scenes"]["adapter-train"] / "manifest.json"
        manifest = read_json(path)
        if damage == "scene_snapshot":
            manifest["source_manifest_snapshot"]["world_gauge"]["metric_scale"] = "unknown"
        elif damage == "reference_role":
            manifest["role"] = "CAUSAL_INPUT"
        else:
            manifest["source_manifest_sha256" if damage == "source_scene_hash" else "source_arrays_sha256"] = "0" * 64
    elif damage == "index_snapshot":
        manifest["source_index_snapshot"]["failures"].append({"reason": "fabricated"})
    elif damage == "source_index_hash":
        manifest["source_index_sha256"] = "0" * 64
    elif damage == "duplicate":
        manifest["episodes"].append(manifest["episodes"][0])
    elif damage == "missing_episode":
        manifest["episodes"].pop()
    else:
        manifest["counts"]["frames"] += 1
    replace_json(path, manifest)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("damage,error", [
    ("index", "SOURCE_INDEX_HASH_MISMATCH"),
    ("episode_manifest", "SOURCE_EPISODE_HASH_MISMATCH"),
    ("episode_frames", "SOURCE_EPISODE_HASH_MISMATCH"),
    ("scene_manifest", "SOURCE_SCENE_HASH_OR_ID_MISMATCH"),
    ("scene_arrays", "SCENE_HASH_MISMATCH"),
    ("rehashed_index", "SOURCE_INDEX_SNAPSHOT_MISMATCH"),
    ("rehashed_episode", "SOURCE_EPISODE_METADATA_MISMATCH"),
    ("rehashed_scene", "SOURCE_SCENE_SNAPSHOT_MISMATCH"),
])
def test_actual_source_mutations_are_not_hidden_by_package_snapshots(package, damage, error):
    package_manifest = read_json(package / "manifest.json")
    ep = first_episode(package)
    em = read_json(ep / "manifest.json")
    ref = package / em["scene_reference"]
    sm = read_json(ref / "manifest.json")
    if damage == "episode_frames":
        path = Path(em["source_episode"]) / "frames.npz"
        data = arrays(path)
        data["rgb"][1:] ^= 1
        np.savez_compressed(path, **data)
    elif damage == "scene_arrays":
        path = Path(sm["source_manifest"]).parent / "scene.npz"
        data = arrays(path)
        data["means"][0, 0] += .01
        np.savez_compressed(path, **data)
    else:
        if damage in ("index", "rehashed_index"):
            path = Path(package_manifest["source_dataset"]) / "dataset.json"
        elif damage in ("episode_manifest", "rehashed_episode"):
            path = Path(em["source_episode"]) / "manifest.json"
        else:
            path = Path(sm["source_manifest"])
        original = read_json(path)
        original["audit_note"] = "Actual source changed after conversion"
        replace_json(path, original)
        if damage == "rehashed_index":
            package_manifest["source_index_sha256"] = sha256(path)
            replace_json(package / "manifest.json", package_manifest)
        elif damage == "rehashed_episode":
            em["source_manifest_sha256"] = sha256(path)
            replace_json(ep / "manifest.json", em)
        elif damage == "rehashed_scene":
            sm["source_manifest_sha256"] = sha256(path)
            replace_json(ref / "manifest.json", sm)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("damage,error", [
    ("gauge", "Scene and map must share scene_id and metric gauge"),
    ("calibration", "CAMERA_CALIBRATION_MISMATCH"),
    ("scene_binding", "SCENE_GAUGE_BINDING_MISMATCH"),
])
def test_validator_checks_source_semantics_after_hash_and_snapshot_updates(package, damage, error):
    ep = first_episode(package)
    em = read_json(ep / "manifest.json")
    path = Path(em["source_episode"]) / "manifest.json"
    original = read_json(path)
    if damage == "gauge":
        original["planning_map"]["gauge_id"] = "unrelated-metric-gauge"
    elif damage == "calibration":
        original["renderer"]["T_vehicle_from_optical"][0][3] += .1
        em["calibration"] = deepcopy(original["renderer"])
    else:
        manifest = read_json(package / "manifest.json")
        other = read_json(package / manifest["scenes"]["adapter-test"] / "manifest.json")
        original["scene_manifest_sha256"] = other["source_manifest_sha256"]
    replace_json(path, original)
    em["source_manifest_sha256"] = sha256(path)
    em["source_manifest_snapshot"] = original
    replace_json(ep / "manifest.json", em)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("official_commit", "0" * 40), ("format", "OFFICIAL_RUNTIME"),
    ("official_runtime", "VERIFIED"), ("official_action_binding", "VERIFIED"),
    ("causal_gaussian_input", "VERIFIED"), ("gaussian_parameter_binding", "VERIFIED"),
    ("native_action_dim", 10), ("training", "RUN"), ("gpu", "USED"),
])
def test_package_cannot_claim_official_bindings_training_or_gpu(package, key, value):
    path = package / "manifest.json"
    manifest = read_json(path)
    manifest[key] = value
    replace_json(path, manifest)
    with pytest.raises(ValueError, match="^PACKAGE_CONTRACT$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("key,value", [("runtime_verified", True), ("commit", "0" * 40)])
def test_rehashed_upstream_spec_cannot_claim_runtime_or_change_pin(package, key, value):
    path = package / "upstream_spec.json"
    spec = read_json(path)
    spec[key] = value
    replace_json(path, spec)
    manifest = read_json(package / "manifest.json")
    manifest["upstream_spec_sha256"] = sha256(path)
    replace_json(package / "manifest.json", manifest)
    with pytest.raises(ValueError, match="^UPSTREAM_RUNTIME_NOT_VERIFIED$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("name", ["inputs.npz", "supervision.npz"])
def test_unrehashed_payload_corruption_is_rejected(package, name):
    ep = first_episode(package)
    data = arrays(ep / name)
    data["initial_rgb" if name == "inputs.npz" else "rgb"].flat[0] ^= 1
    replace_payload(ep, name, data, rehash=False)
    if name == "inputs.npz":
        with pytest.raises(ValueError, match="^INPUT_HASH_MISMATCH$"):
            episodes.load_rollout_inputs(ep)
    with pytest.raises(ValueError, match=f"^EPISODE_FILE_HASH_MISMATCH:{name}$"):
        episodes.validate_package(package)


@pytest.mark.parametrize("width,error", [(2, "OFFICIAL_ACTION_DIMENSION_MISMATCH"),
                                         (10, "OFFICIAL_ACTION_BINDING_PENDING")])
def test_vehicle_actions_are_not_robot_actions_even_at_matching_width(width, error):
    with pytest.raises(ValueError, match=f"^{error}:"):
        episodes.require_official_action_binding(np.zeros((1, 3, width), dtype=np.float32))


def test_image_channel_range_conversion_is_exact_native_resolution():
    rgb = np.arange(2 * 3 * 48 * 64 * 3, dtype=np.int64).astype(np.uint8).reshape(2, 3, 48, 64, 3)
    before = rgb.copy()
    result = episodes.rgb_to_gwm_images(rgb)
    assert result.shape == (2, 3, 3, 48, 64)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, rgb.transpose(0, 1, 4, 2, 3).astype(np.float32) / np.float32(255), strict=True)
    assert result.min() == 0 and result.max() == 1
    assert not np.shares_memory(result, rgb)
    np.testing.assert_array_equal(rgb, before, strict=True)
    for invalid in (rgb.astype(np.float32), rgb[0], rgb[..., :2]):
        with pytest.raises(ValueError, match="Expected uint8 RGB"):
            episodes.rgb_to_gwm_images(invalid)


def test_conversion_and_roundtrip_outputs_are_exclusive(package):
    manifest = read_json(package / "manifest.json")
    before = sha256(package / "manifest.json")
    with pytest.raises(FileExistsError):
        episodes.convert(manifest["source_dataset"], package, gaussian_count=32)
    with pytest.raises(FileExistsError):
        episodes.roundtrip(package, package)
    assert sha256(package / "manifest.json") == before


@pytest.mark.parametrize("phase", ["convert", "validate"])
@pytest.mark.parametrize("alias", ["arrays", "gauge"])
def test_different_scene_ids_cannot_hide_cross_split_source_aliases(small_source, tmp_path, monkeypatch, phase, alias):
    source = tmp_path / "source"
    shutil.copytree(small_source, source)
    train = read_json(source / "adapter-train" / "manifest.json")
    path = source / "adapter-test" / "manifest.json"
    scene = read_json(path)
    if alias == "arrays":
        # Exact variant-0 copy under a different scene ID and gauge.
        shutil.copyfile(source / "adapter-train" / "scene.npz", source / "adapter-test" / "scene.npz")
        scene["array_sha256"] = train["array_sha256"]
    else:
        scene["world_gauge"]["id"] = train["world_gauge"]["id"]
    replace_json(path, scene)
    for relative in read_json(source / "dataset.json")["episodes"]:
        path = source / relative / "manifest.json"
        original = read_json(path)
        scene_path = source / original["scene_id"] / "manifest.json"
        original["scene_manifest"] = str(scene_path)
        original["scene_manifest_sha256"] = sha256(scene_path)
        if alias == "gauge" and original["split"] == "test":
            original["planning_map"]["gauge_id"] = train["world_gauge"]["id"]
        replace_json(path, original)
    output = tmp_path / "package"
    if phase == "validate":
        # Produce a well-formed leaky package without disabling any other checks.
        with monkeypatch.context() as patch:
            patch.setattr(episodes, "_split_binding", lambda *args: None)
            episodes.convert(source, output, gaussian_count=32)
        with pytest.raises(ValueError, match="^SCENE_SPLIT_LEAKAGE$"):
            episodes.validate_package(output)
    else:
        with pytest.raises(ValueError, match="^SCENE_SPLIT_LEAKAGE$"):
            episodes.convert(source, output, gaussian_count=32)
        assert not output.exists()


@pytest.mark.parametrize("damage,error", [
    ("absolute", "UNSAFE_PACKAGE_PATH"), ("parent", "UNSAFE_PACKAGE_PATH"),
    ("symlink", "UNSAFE_PACKAGE_PATH"), ("manifest_symlink", "UNSAFE_PACKAGE_PATH"),
    ("frames_symlink", "UNSAFE_PACKAGE_PATH"), ("episodes", "EPISODE_LIMIT_1024"),
    ("frame_count", "FRAME_LIMIT_4096"), ("action_count", "FRAME_LIMIT_4096"),
    ("zero", "FRAME_LIMIT_4096"), ("boolean", "FRAME_LIMIT_4096"),
    ("fraction", "FRAME_LIMIT_4096"),
])
def test_convert_preflight_precedes_expensive_validation(small_source, tmp_path, monkeypatch, damage, error):
    source = tmp_path / "source"
    source.mkdir()
    index = deepcopy(read_json(small_source / "dataset.json"))
    if damage in ("absolute", "parent", "symlink"):
        if damage == "symlink":
            (source / "escape").symlink_to(small_source, target_is_directory=True)
        index["episodes"] = [str(small_source) if damage == "absolute" else
                             "../escape" if damage == "parent" else "escape"]
    elif damage == "episodes":
        index["episodes"] = ["missing"] * 1025
    else:
        index["episodes"] = ["episode"]
        directory = source / "episode"
        directory.mkdir()
        original = deepcopy(read_json(small_source / read_json(small_source / "dataset.json")["episodes"][0] / "manifest.json"))
        if damage == "manifest_symlink":
            (directory / "manifest.json").symlink_to(small_source / "dataset.json")
        else:
            if damage == "frames_symlink":
                (directory / "frames.npz").symlink_to(small_source / "dataset.json")
            else:
                original[damage if damage in ("frame_count", "action_count") else "frame_count"] = {
                    "zero": 0, "boolean": True, "fraction": 1.5,
                }.get(damage, 4097)
            write_json(directory / "manifest.json", original)
    write_json(source / "dataset.json", index)

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight accessed expensive validation or arrays")

    monkeypatch.setattr(episodes, "validate_dataset", forbidden)
    monkeypatch.setattr(episodes, "load_scene", forbidden)
    monkeypatch.setattr(episodes.np, "load", forbidden)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.convert(source, tmp_path / "output", gaussian_count=32)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("damage,error", [
    ("source_escape", "UNSAFE_PACKAGE_PATH"), ("package_escape", "UNSAFE_PACKAGE_PATH"),
    ("scene_escape", "UNSAFE_PACKAGE_PATH"), ("reference_escape", "UNSAFE_PACKAGE_PATH"),
    ("source_count", "EPISODE_LIMIT_1024"), ("package_count", "EPISODE_LIMIT_1024"),
    ("coverage", "SOURCE_EPISODE_COVERAGE_MISMATCH"), ("frames", "FRAME_LIMIT_4096"),
])
def test_package_preflight_precedes_source_walk(package, monkeypatch, damage, error):
    path = package / "manifest.json"
    manifest = read_json(path)
    if damage in ("source_escape", "source_count"):
        index_path = Path(manifest["source_dataset"]) / "dataset.json"
        index = read_json(index_path)
        index["episodes"] = ["../escape"] if damage == "source_escape" else ["missing"] * 1025
        replace_json(index_path, index)
        manifest["source_index_sha256"] = sha256(index_path)
        manifest["source_index_snapshot"] = index
    elif damage in ("package_escape", "package_count"):
        manifest["episodes"] = ["../escape"] if damage == "package_escape" else ["missing"] * 1025
    elif damage == "scene_escape":
        manifest["scenes"]["adapter-test"] = "../escape"
    else:
        ep_path = first_episode(package) / "manifest.json"
        em = read_json(ep_path)
        if damage == "reference_escape":
            em["scene_reference"] = "../escape"
        elif damage == "coverage":
            em["source_episode"] = str(package.parent / "unintended-source")
        else:
            em["frame_count"] = 4097
        replace_json(ep_path, em)
    replace_json(path, manifest)

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight followed a source episode or loaded arrays")

    monkeypatch.setattr(episodes, "validate_episode", forbidden)
    monkeypatch.setattr(episodes, "load_scene", forbidden)
    monkeypatch.setattr(episodes.np, "load", forbidden)
    with pytest.raises(ValueError, match=f"^{error}$"):
        episodes.validate_package(package)
