"""Small CPU-only episode contract tests; no training or full scene pipeline."""

from copy import deepcopy
import json
import shutil

import numpy as np
import pytest

from gaussian_park_world.data_pipeline import build_episodes
from gaussian_park_world.dataset.offline import (
    load_scene, read_json, render_frames, rollout_inputs, validate_dataset,
    validate_episode, write_episode,
)
from gaussian_park_world.planning import fixture_map, plan_and_track
from gaussian_park_world.scenes import make_fixture, sha256, write_json


def replace_json(path, value):
    path.unlink()
    write_json(path, value)


def read_frames(episode):
    with np.load(episode / "frames.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def replace_frames(episode, arrays, *, rehash=True):
    np.savez_compressed(episode / "frames.npz", **arrays)
    if rehash:
        manifest = read_json(episode / "manifest.json")
        manifest["frames_sha256"] = sha256(episode / "frames.npz")
        replace_json(episode / "manifest.json", manifest)


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline-dataset")
    scene = root / "scene"
    make_fixture(scene, "small-forward")
    planning_map = fixture_map("small-forward")
    trajectories = {}
    for distance in (.6, 1.2):
        name = f"forward_{distance}"
        goal = [distance, 0., 0., 0.]
        trajectory = plan_and_track(planning_map, [0., 0., 0., 0.], goal,
                                    max_expansions=12, max_retries=0)
        assert trajectory["status"] == "SUCCESS", trajectory
        write_episode(root / name, scene, planning_map, trajectory, name, "train", goal)
        trajectories[name] = trajectory
    write_json(root / "dataset.json", dict(episodes=list(trajectories), failures=[]))
    return root, trajectories


@pytest.fixture
def dataset(tmp_path, small_dataset):
    source, _ = small_dataset
    root = tmp_path / "dataset"
    shutil.copytree(source, root)
    # Episode scene references are absolute; keep mutations local to each test.
    for name in ("forward_0.6", "forward_1.2"):
        path = root / name / "manifest.json"
        manifest = read_json(path)
        manifest["scene_manifest"] = str(root / "scene" / "manifest.json")
        replace_json(path, manifest)
    return root


def test_small_write_validate_and_exclusive_output(dataset, small_dataset):
    summary = validate_dataset(dataset)
    assert summary["status"] == "FIXTURE_ONLY"
    assert summary["training"] == "NOT_RUN"
    assert summary["scenes"] == 1
    assert summary["trajectories"] == 2
    assert summary["scene_splits"] == {"small-forward": "train"}
    assert summary["failures"] == []
    total = 0
    for distance in (.6, 1.2):
        name = f"forward_{distance}"
        episode = dataset / name
        manifest = read_json(episode / "manifest.json")
        arrays = read_frames(episode)
        row = validate_episode(episode)
        total += row["frames"]
        assert row["frames"] == len(arrays["actions"]) + 1
        assert 0 < row["min_coverage"] <= row["max_coverage"] <= 1
        assert manifest["frames_sha256"] == sha256(episode / "frames.npz")
        assert manifest["scene_manifest_sha256"] == sha256(dataset / "scene" / "manifest.json")
        assert manifest["renderer"]["backend"] == "existing_numpy_gaussian_reference_not_production_3dgs"
        assert manifest["training"] == "NOT_RUN"
        np.testing.assert_allclose(arrays["states"][-1], [distance, 0, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(arrays["actions"][-1], [0, 0])
        np.testing.assert_allclose(arrays["timestamp"], np.r_[0., np.cumsum(arrays["dt"])])
        before = sha256(episode / "frames.npz")
        with pytest.raises(FileExistsError):
            write_episode(episode, dataset / "scene", fixture_map("small-forward"),
                          small_dataset[1][name], name, "train", [distance, 0, 0, 0])
        assert sha256(episode / "frames.npz") == before
    assert summary["frames"] == total


@pytest.mark.parametrize("damage,error", [
    ("state", "ACTION_STATE_MISMATCH"),
    ("action", "ACTION_STATE_MISMATCH"),
    ("speed_limit", "ACTION_LIMIT"),
    ("steering_limit", "ACTION_LIMIT"),
    ("timestamp", "ACTION_STATE_TIME_MISMATCH:timestamp"),
    ("command_issue_time", "ACTION_STATE_TIME_MISMATCH:command_issue_time"),
    ("effective_start", "ACTION_STATE_TIME_MISMATCH:effective_start"),
    ("effective_end", "ACTION_STATE_TIME_MISMATCH:effective_end"),
    ("camera_pose", "CAMERA_POSE_MISMATCH"),
    ("vehicle_pose", "CAMERA_POSE_MISMATCH"),
    ("alpha_range", "ARRAY_RANGE_OR_DTYPE"),
    ("rgb_dtype", "ARRAY_RANGE_OR_DTYPE"),
    ("collision_dtype", "ARRAY_RANGE_OR_DTYPE"),
    ("dt", "ARRAY_RANGE_OR_DTYPE"),
    ("frame_id", "ARRAY_RANGE_OR_DTYPE"),
    ("nan_state", "ARRAY_SCHEMA:states"),
    ("initial_overspeed", "STATE_SPEED_LIMIT"),
    ("missing_action", "ARRAY_SCHEMA:actions"),
    ("state_shape", "ARRAY_SCHEMA:states"),
    ("collision", "COLLISION_OR_OUT_OF_BOUNDS"),
    ("empty_view", "EMPTY_VIEW_OR_COVERAGE_MISMATCH"),
    ("coverage", "EMPTY_VIEW_OR_COVERAGE_MISMATCH"),
])
def test_frame_corruption_hash_then_semantic_validation(dataset, damage, error):
    episode = dataset / "forward_0.6"
    arrays = read_frames(episode)
    if damage == "state":
        arrays["states"][1, 0] += .2
    elif damage == "action":
        arrays["actions"][0, 0] = .5
    elif damage in ("speed_limit", "steering_limit"):
        arrays["actions"][0, 0 if damage == "speed_limit" else 1] = 2
    elif damage in ("timestamp", "command_issue_time", "effective_start", "effective_end"):
        arrays[damage][0] += .01
    elif damage in ("camera_pose", "vehicle_pose"):
        arrays[damage][0, 0, 3] += .1
    elif damage == "alpha_range":
        arrays["alpha"][0, 0, 0] = 1.1
    elif damage in ("rgb_dtype", "collision_dtype"):
        key = damage.removesuffix("_dtype")
        arrays[key] = arrays[key].astype(float)
    elif damage == "dt":
        arrays["dt"][0] = 0
    elif damage == "frame_id":
        arrays["frame_id"][0] = 9
    elif damage == "nan_state":
        arrays["states"][0, 0] = np.nan
    elif damage == "initial_overspeed":
        arrays["states"][0, 3] = 100
    elif damage == "missing_action":
        del arrays["actions"]
    elif damage == "state_shape":
        arrays["states"] = arrays["states"][:, :3]
    elif damage == "collision":
        arrays["collision"][0] = True
    elif damage == "empty_view":
        arrays["alpha"][0] = 0
        arrays["coverage"][0] = 0
    else:
        arrays["coverage"][0] = 1
    replace_frames(episode, arrays, rehash=False)
    with pytest.raises(ValueError, match="^FRAMES_HASH_MISMATCH$"):
        validate_episode(episode)
    replace_frames(episode, arrays)
    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_episode(episode)


@pytest.mark.parametrize("field,value,error", [
    ("schema_version", 2, "EPISODE_SCHEMA"),
    ("split", "validation", "EPISODE_SCHEMA"),
    ("action_count", 0, "FRAME_ACTION_LENGTH"),
    ("scene_id", "other-scene", "SCENE_ID_OR_SOURCE_MISMATCH"),
    ("source_classification", "REAL", "SCENE_ID_OR_SOURCE_MISMATCH"),
    ("goal_reached", False, "GOAL_NOT_REACHED"),
    ("goal_reached", 1, "GOAL_NOT_REACHED"),
    ("goal", [10., 0., 0., 0.], "GOAL_NOT_REACHED"),
    ("goal", [0., 0.], "INVALID_GOAL"),
    ("goal", [[.6, 0., 0.]], "INVALID_GOAL"),
    ("goal", [.6, 0., 0., 0., 0.], "INVALID_GOAL"),
    ("goal", [.6, 0., 0., .1], "INVALID_GOAL"),
])
def test_manifest_semantic_corruption(dataset, field, value, error):
    episode = dataset / "forward_0.6"
    manifest = read_json(episode / "manifest.json")
    manifest[field] = value
    replace_json(episode / "manifest.json", manifest)
    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_episode(episode)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_goal_rejected(dataset, value):
    episode = dataset / "forward_0.6"
    path = episode / "manifest.json"
    manifest = read_json(path)
    manifest["goal"][0] = value
    path.unlink()
    # Deliberately bypass the strict writer to simulate a corrupt input manifest.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream)
    with pytest.raises(ValueError, match="^INVALID_GOAL$"):
        validate_episode(episode)


def test_three_component_goal_supported(dataset):
    episode = dataset / "forward_0.6"
    manifest = read_json(episode / "manifest.json")
    manifest["goal"] = manifest["goal"][:3]
    replace_json(episode / "manifest.json", manifest)
    assert validate_episode(episode)["trajectory_id"] == "forward_0.6"


@pytest.mark.parametrize("section,field,value,error", [
    ("renderer", "T_vehicle_from_optical", np.eye(4).tolist(), "CAMERA_CALIBRATION_MISMATCH"),
    ("renderer", "K", np.eye(3).tolist(), "CAMERA_CALIBRATION_MISMATCH"),
    ("renderer", "image_size_wh", [1, 1], "CAMERA_CALIBRATION_MISMATCH"),
    ("renderer", "source_sha256", "0" * 64, "RENDERER_PROVENANCE_MISMATCH"),
    ("renderer", "backend", "unknown", "RENDERER_PROVENANCE_MISMATCH"),
    ("execution", "wheelbase", 3., "UNSUPPORTED_VEHICLE_OR_EXECUTION"),
    ("execution", "speed_snap", 1, "UNSUPPORTED_VEHICLE_OR_EXECUTION"),
    ("execution", "speed_snap", False, "UNSUPPORTED_VEHICLE_OR_EXECUTION"),
    ("timing", "command_latency", "UNKNOWN", "UNSUPPORTED_TIMING"),
    ("timing", "interval", "[t+dt,t+2dt)", "UNSUPPORTED_TIMING"),
])
def test_calibration_provenance_and_execution_tampering(dataset, section, field, value, error):
    episode = dataset / "forward_0.6"
    manifest = read_json(episode / "manifest.json")
    manifest[section][field] = value
    replace_json(episode / "manifest.json", manifest)
    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_episode(episode)


@pytest.mark.parametrize("damage,error", [
    ("goal_shape", "INVALID_GOAL"),
    ("nan_goal", "Out of range float values are not JSON compliant: nan"),
    ("moving_goal", "INVALID_GOAL"),
    ("initial_overspeed", "STATE_SPEED_LIMIT"),
    ("wheelbase", "UNSUPPORTED_VEHICLE_OR_EXECUTION"),
    ("split", "EPISODE_SCHEMA"),
])
def test_rejected_write_is_not_published(dataset, small_dataset, damage, error):
    trajectory = deepcopy(small_dataset[1]["forward_0.6"])
    goal, split = [.6, 0., 0., 0.], "train"
    if damage == "goal_shape":
        goal = goal[:2]
    elif damage == "nan_goal":
        goal[0] = np.nan
    elif damage == "moving_goal":
        goal[3] = .1
    elif damage == "initial_overspeed":
        trajectory["states"][0, 3] = 100
    elif damage == "wheelbase":
        trajectory["metadata"]["wheelbase"] = 3.
    else:
        split = "invalid"
    output = dataset / "rejected"
    with pytest.raises(ValueError, match=f"^{error}$"):
        write_episode(output, dataset / "scene", fixture_map("small-forward"),
                      trajectory, "rejected", split, goal)
    assert not (output / "manifest.json").exists()
    assert (output / "frames.npz").is_file()
    if damage != "nan_goal":
        assert (output / "pending_manifest.json").is_file()
    before = {path.name: sha256(path) for path in output.iterdir()}
    with pytest.raises(FileNotFoundError):
        validate_episode(output)
    with pytest.raises(FileExistsError):
        write_episode(output, dataset / "scene", fixture_map("small-forward"),
                      trajectory, "rejected", "train", [.6, 0., 0., 0.])
    assert {path.name: sha256(path) for path in output.iterdir()} == before


def test_scene_hash_chain_and_semantic_validation(dataset):
    episode = dataset / "forward_0.6"
    scene = dataset / "scene"
    manifest = read_json(scene / "manifest.json")
    with np.load(scene / "scene.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["opacities"][0] = 2
    np.savez(scene / "scene.npz", **arrays)
    with pytest.raises(ValueError, match="^SCENE_HASH_MISMATCH$"):
        validate_episode(episode)
    manifest["array_sha256"] = sha256(scene / "scene.npz")
    replace_json(scene / "manifest.json", manifest)
    with pytest.raises(ValueError, match="^SCENE_MANIFEST_HASH_MISMATCH$"):
        validate_episode(episode)
    episode_manifest = read_json(episode / "manifest.json")
    episode_manifest["scene_manifest_sha256"] = sha256(scene / "manifest.json")
    replace_json(episode / "manifest.json", episode_manifest)
    with pytest.raises(ValueError, match="invalid opacities range"):
        validate_episode(episode)


@pytest.mark.parametrize("damage,error", [
    ("empty", "EMPTY_DATASET"),
    ("duplicate", "DUPLICATE_EPISODE"),
    ("split", "SCENE_SPLIT_LEAKAGE"),
])
def test_dataset_identity_and_scene_split_gate(dataset, damage, error):
    index = read_json(dataset / "dataset.json")
    if damage == "empty":
        index["episodes"] = []
    elif damage == "duplicate":
        index["episodes"].append(index["episodes"][0])
    else:
        path = dataset / "forward_1.2" / "manifest.json"
        manifest = read_json(path)
        manifest["split"] = "test"
        replace_json(path, manifest)
    replace_json(dataset / "dataset.json", index)
    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_dataset(dataset)


def test_renderer_rejects_empty_views_and_unknown_backend(dataset):
    states = np.array([[0., 0., np.pi, 0.]])
    with pytest.raises(ValueError, match="^EMPTY_VIEW$"):
        render_frames(dataset / "scene", states)
    with pytest.raises(ValueError, match="^UNSUPPORTED_RENDERER$"):
        render_frames(dataset / "scene", states, backend="not-a-renderer")


@pytest.mark.parametrize("damage,error", [
    ("unknown_free", "Need evidenced ground and occupied unknown space"),
    ("ground", "Need evidenced ground and occupied unknown space"),
    ("gauge", "Scene and map must share scene_id and metric gauge"),
    ("unknown_obstacle", "SWEPT_COLLISION"),
])
def test_episode_planning_map_fails_closed(dataset, damage, error):
    episode = dataset / "forward_0.6"
    manifest = read_json(episode / "manifest.json")
    planning_map = manifest["planning_map"]
    if damage == "unknown_free":
        planning_map["unknown_occupied"] = False
    elif damage == "ground":
        planning_map["ground"]["trusted"] = False
    elif damage == "gauge":
        planning_map["gauge_id"] = "unrelated-gauge"
    else:
        planning_map["unknown_rectangles"] = [[0., -.1, .1, .1]]
    replace_json(episode / "manifest.json", manifest)
    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_episode(episode)


@pytest.mark.parametrize("metric_scale", ["unknown", "verified"])
def test_real_scene_geometry_blocked_before_episode_write(dataset, small_dataset, metric_scale):
    scene = dataset / "scene"
    manifest = read_json(scene / "manifest.json")
    manifest["source_classification"] = "DGGT_RECONSTRUCTION_UNKNOWN_METRIC_GAUGE"
    manifest["world_gauge"]["metric_scale"] = metric_scale
    manifest["planning_status"] = (
        "BLOCKED_UNKNOWN_METRIC_GAUGE" if metric_scale == "unknown" else "BLOCKED_GEOMETRY")
    manifest["provenance"]["source_sha256"] = {"scene.npz": sha256(scene / "scene.npz")}
    replace_json(scene / "manifest.json", manifest)
    load_scene(scene)  # A valid visualization scene is not trusted planning geometry.
    output = dataset / "blocked"
    trajectory = small_dataset[1]["forward_0.6"]
    with pytest.raises(ValueError, match="supported metric fixture evidence"):
        write_episode(output, scene, fixture_map("small-forward"), trajectory,
                      "blocked", "train", [.6, 0, 0, 0])
    assert not output.exists()
    with pytest.raises(ValueError, match="^REAL_SCENE_PLANNING=BLOCKED_GEOMETRY$"):
        render_frames(scene, trajectory["states"][:1])
    episode = dataset / "forward_0.6"
    episode_manifest = read_json(episode / "manifest.json")
    episode_manifest["scene_manifest_sha256"] = sha256(scene / "manifest.json")
    replace_json(episode / "manifest.json", episode_manifest)
    with pytest.raises(ValueError, match="supported metric fixture evidence"):
        validate_episode(episode)


def test_failed_trajectory_cannot_be_published(dataset, small_dataset):
    trajectory = deepcopy(small_dataset[1]["forward_0.6"])
    trajectory["status"] = "FAILURE"
    output = dataset / "failed"
    with pytest.raises(ValueError, match="Cannot publish a failed trajectory"):
        write_episode(output, dataset / "scene", fixture_map("small-forward"),
                      trajectory, "failed", "train", [.6, 0, 0, 0])
    assert not output.exists()


def test_rollout_allowlist_ignores_future_observations_and_metadata(dataset):
    episode = dataset / "forward_0.6"
    expected = rollout_inputs(episode)
    arrays = read_frames(episode)
    assert set(expected) == {"initial_rgb", "initial_alpha", "issued_actions", "dt_s"}
    for key, target in (("initial_rgb", arrays["rgb"][:1]),
                        ("initial_alpha", arrays["alpha"][:1]),
                        ("issued_actions", arrays["actions"]), ("dt_s", arrays["dt"])):
        np.testing.assert_array_equal(expected[key], target)
    arrays["rgb"][1:] = 255 - arrays["rgb"][1:]
    arrays["alpha"][1:] = .5
    arrays["coverage"][1:] = 1
    arrays["future_target"] = np.full((len(arrays["states"]), 4), 123456.)
    replace_frames(episode, arrays)
    manifest = read_json(episode / "manifest.json")
    manifest["future_metadata"] = dict(target_pose=[999, 999, 999], secret="privileged")
    manifest["causal_input"] = "POISON: include all future poses and targets"
    replace_json(episode / "manifest.json", manifest)
    actual = rollout_inputs(episode)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        actual[key][...] = 0
    fresh = rollout_inputs(episode)
    for key in expected:
        np.testing.assert_array_equal(fresh[key], expected[key])
    arrays["states"][1, 0] += 1
    replace_frames(episode, arrays)
    with pytest.raises(ValueError, match="^ACTION_STATE_MISMATCH$"):
        rollout_inputs(episode)


def test_build_episodes_small_expert_index(tmp_path, small_dataset):
    source, trajectories = small_dataset
    experts = tmp_path / "experts"
    experts.mkdir()
    entries = []
    failure = dict(scene_id="small-forward", trajectory_id="blocked", reason="START_BLOCKED")
    for name, trajectory in trajectories.items():
        np.savez(experts / f"{name}.npz", **{k: trajectory[k] for k in ("states", "actions", "dt")})
        entries.append(dict(scene_id="small-forward", trajectory_id=name,
                            scene_dir=str(source / "scene"), planning_map=fixture_map("small-forward"),
                            goal=trajectory["states"][-1].tolist(), metadata=trajectory["metadata"],
                            file=f"{name}.npz"))
    write_json(experts / "experts.json", dict(trajectories=entries, failures=[failure]))
    output = tmp_path / "episodes"
    summary = build_episodes(experts, output, backend="cpu_reference")
    assert summary == validate_dataset(output)
    assert summary == read_json(output / "validation.json")
    assert summary["trajectories"] == 2
    assert summary["failures"] == [failure]
    assert summary["training"] == "NOT_RUN"
    assert read_json(output / "dataset.json")["scene_splits"] == {"small-forward": "train"}
    with pytest.raises(FileExistsError):
        build_episodes(experts, output)
