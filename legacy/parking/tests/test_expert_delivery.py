"""Small CPU-only delivery tests; no renderer, observation or GWM execution."""

import json

import numpy as np
import pytest

from gaussian_park_world.planning.delivery import generate, validate_experts, _digest, _geometry, _check_arrays
from gaussian_park_world.planning.expert import SOURCE, integrate
from gaussian_park_world.dataset.offline import load_scene, vehicle_pose
from gaussian_park_world.scenes.contract import make_fixture, sha256


def _write(path, value):
    # Tests deliberately rewrite artifacts to exercise fail-closed validation.
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


REAL_SOURCE = "DGGT_RECONSTRUCTION_UNKNOWN_METRIC_GAUGE"


def _make_real(source, scene_id="real"):
    make_fixture(source, scene_id)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["source_classification"] = REAL_SOURCE
    manifest["provenance"]["source_sha256"] = {"original": "0" * 64}
    manifest["world_gauge"].update(metric_scale="unknown", units="unknown", confidence="unknown")
    manifest["planning_status"] = "BLOCKED_UNKNOWN_METRIC_GAUGE"
    _write(source / "manifest.json", manifest)


@pytest.fixture
def delivery(tmp_path):
    source = tmp_path / "scene"
    make_fixture(source, "fixture", variant=4)
    output = tmp_path / "experts"
    requests = [dict(trajectory_id="forward", start=[-5., 0., 0., 0.], goal=[-1.4, 0., 0., 0.]),
                dict(trajectory_id="blocked", start=[14., 0., 0., 0.], goal=[0., 0., 0., 0.])]
    result = generate(source, output, requests=requests, budget=100, retries=0)
    assert result["trajectories"] == 1
    assert result["failure_reasons"] == {"START_BLOCKED": 1}
    return source, output


def _index(output):
    return json.loads((output / "experts.json").read_text())


def _mutate_npz(output, mutate):
    index = _index(output)
    entry = index["trajectories"][0]
    path = output / entry["file"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    mutate(arrays)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    entry["sha256"] = sha256(path)
    _write(output / "experts.json", index)


def test_delivery_contract_and_no_overwrite(delivery):
    source, output = delivery
    index = _index(output)
    result = validate_experts(output)
    assert result["valid"] and result["frames"] == result["actions"] + 1
    assert result["source_classification"] == SOURCE
    assert result["source_classifications"] == result["trajectory_source_classifications"] == [SOURCE]
    assert index["source_scenes"][0]["source_classification"] == SOURCE
    entry = index["trajectories"][0]
    with np.load(output / entry["file"], allow_pickle=False) as arrays:
        assert arrays["states"].shape == (result["frames"], 4)
        assert arrays["actions"].shape == (result["actions"], 2)
        assert arrays["camera_pose"][0, 2, 3] == 1.
    assert entry["scene_manifest_sha256"] == sha256(source / "manifest.json")
    before = sha256(output / "experts.json")
    with pytest.raises(FileExistsError):
        generate(source, output)
    assert sha256(output / "experts.json") == before


@pytest.mark.parametrize("key,index,value,error", [
    ("actions", (0, 0), .5, "ACTION_REPLAY_MISMATCH"),
    ("actions", (0, 0), 1.1, "VEHICLE_LIMITS"),
    ("actions", (0, 1), .56, "VEHICLE_LIMITS"),
    ("dt", 0, 0., "INVALID_DT"),
    ("dt", 0, -.1, "INVALID_DT"),
    ("dt", 0, np.nan, "INVALID_ARRAY"),
    ("timestamp", 1, .5, "INVALID_TIMING"),
    ("effective_end", 0, .2, "INVALID_TIMING"),
    ("command_issue_time", 0, .1, "INVALID_TIMING"),
    ("frame_id", 1, 0, "INVALID_TIMING"),
    ("vehicle_pose", (1, 0, 3), 99., "VEHICLE_POSE_MISMATCH"),
    ("camera_pose", (1, 2, 3), 2., "CAMERA_POSE_MISMATCH"),
])
def test_reject_tampered_arrays_even_with_updated_hash(delivery, key, index, value, error):
    _, output = delivery
    _mutate_npz(output, lambda a: a[key].__setitem__(index, value))
    with pytest.raises(ValueError, match=error):
        validate_experts(output)


@pytest.mark.parametrize("change,error", [
    (lambda i: i["counts"].__setitem__("actions", 0), "INDEX_COUNT_MISMATCH"),
    (lambda i: i["trajectories"].append(i["trajectories"][0]), "REQUEST_OUTCOME_MISMATCH"),
    (lambda i: i["trajectories"][0].__setitem__("trajectory_id", "other"), "REQUEST_OUTCOME_MISMATCH"),
    (lambda i: i["trajectories"][0].__setitem__("file", "../other/output.npz"), "CROSS_OUTPUT_REFERENCE"),
    (lambda i: i["trajectories"][0].__setitem__("sha256", "0" * 64), "TRAJECTORY_HASH_MISMATCH"),
    (lambda i: i["trajectories"][0]["metadata"].__setitem__("wheelbase", 3.), "INVALID_VEHICLE_METADATA"),
    (lambda i: i["trajectories"][0]["statistics"].__setitem__("forward_count", 0), "STATISTICS_MISMATCH"),
    (lambda i: i["source_scenes"][0]["world_gauge"].__setitem__("id", "wrong"), "SOURCE_SCENE_GAUGE_MISMATCH"),
    (lambda i: i["requests"][0].__setitem__("scene_id", "wrong"), "INVALID_OR_DUPLICATE_REQUEST"),
    (lambda i: i["failures"][0].__setitem__("reason", "MADE_UP"), "INVALID_FAILURE_REASON"),
])
def test_reject_index_corruption(delivery, change, error):
    _, output = delivery
    index = _index(output)
    change(index)
    _write(output / "experts.json", index)
    with pytest.raises(ValueError, match=error):
        validate_experts(output)


def test_stale_source_manifest(delivery):
    source, output = delivery
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["provenance"]["purpose"] = "changed"
    _write(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="STALE_SCENE_MANIFEST_HASH"):
        validate_experts(output)


def test_stale_source_arrays(delivery):
    source, output = delivery
    with (source / "scene.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SCENE_HASH_MISMATCH"):
        validate_experts(output)


@pytest.mark.parametrize("field,value", [("gauge_id", "wrong"), ("scene_id", "other"),
                                        ("unknown_occupied", False)])
def test_wrong_map_scene_gauge_unknown(delivery, field, value):
    _, output = delivery
    index = _index(output)
    descriptor = index["maps"][0]
    descriptor["planning_map"][field] = value
    descriptor["geometry_sha256"] = _digest(_geometry(descriptor["planning_map"]))
    snapshot = {k: v for k, v in descriptor.items() if k not in ("file", "sha256")}
    _write(output / descriptor["file"], snapshot)
    descriptor["sha256"] = sha256(output / descriptor["file"])
    _write(output / "experts.json", index)
    with pytest.raises(ValueError):
        validate_experts(output)


@pytest.mark.parametrize("endpoint,value", [("start", [-4., 0., 0., 0.]),
                                           ("goal", [8., 0., 0., 0.]),
                                           ("goal", [-1.4, 0., 0., 1.]),
                                           ("goal", [-1.4, 0., 1., 0.])])
def test_invalid_start_goal(delivery, endpoint, value):
    _, output = delivery
    index = _index(output)
    index["requests"][0][endpoint] = value
    index["trajectories"][0][endpoint] = value
    _write(output / "experts.json", index)
    with pytest.raises(ValueError, match="INVALID_START_GOAL_STOP"):
        validate_experts(output)


def test_reject_collision_with_replay_consistent_shift(delivery):
    _, output = delivery
    _mutate_npz(output, lambda a: a["states"].__setitem__((slice(None), 1), 4.5))
    index = _index(output)
    for entry in (index["requests"][0], index["trajectories"][0]):
        entry["start"][1] = 4.5
        entry["goal"][1] = 4.5
    _write(output / "experts.json", index)
    with pytest.raises(ValueError, match="BLOCKED_ENDPOINT"):
        validate_experts(output)


def test_sweep_rejects_tunneling_between_free_endpoints(delivery):
    source, output = delivery
    entry = _index(output)["trajectories"][0]
    scene, _ = load_scene(source)
    start = np.array([-5., 0., 0., 0.])
    actions = np.array([[1., 0.], [0., 0.]])
    dt = np.array([12., .1])
    end = integrate(start, actions[0], dt[0])
    states = np.array([start, end, integrate(end, actions[1], dt[1])])
    times = np.r_[0., np.cumsum(dt)]
    poses = np.array([vehicle_pose(s) for s in states])
    arrays = dict(states=states, actions=actions, dt=dt, timestamp=times,
                  frame_id=np.arange(3), command_issue_time=times[:-1],
                  effective_start=times[:-1], effective_end=times[1:], vehicle_pose=poses,
                  camera_pose=poses @ np.asarray(scene["cameras"][0]["T_world_from_optical"]))
    entry["start"] = start.tolist()
    entry["goal"] = states[-1].tolist()
    # Local checker test: map gate separately rejects noncanonical snapshots.
    entry["planning_map"]["unknown_rectangles"] = [[0., -8., .01, 8.]]
    with pytest.raises(ValueError, match="SWEPT_COLLISION"):
        _check_arrays(arrays, entry, scene)


def test_batch_three_distinct_layouts_single_scene(tmp_path):
    source, output = tmp_path / "scene", tmp_path / "experts"
    make_fixture(source, "fixture", variant=2)
    request = dict(trajectory_id="short", start=[-5., 0., 0., 0.], goal=[-3.8, 0., 0., 0.])
    result = generate(source, output, batch=True, requests=[request], budget=50, retries=0)
    assert result["trajectories"] == 3 and result["geometry_layouts"] == 3
    index = _index(output)
    assert len({m["planning_map"]["gauge_id"] for m in index["maps"]}) == 1
    assert len({m["geometry_sha256"] for m in index["maps"]}) == 3
    duplicate = index["maps"][2]
    for key in ("layout", "name", "planning_map", "geometry_sha256"):
        duplicate[key] = index["maps"][0][key]
    _write(output / duplicate["file"], {k: v for k, v in duplicate.items() if k not in ("file", "sha256")})
    duplicate["sha256"] = sha256(output / duplicate["file"])
    _write(output / "experts.json", index)
    with pytest.raises(ValueError, match="BATCH_LAYOUTS_NOT_UNIQUE"):
        validate_experts(output)


def test_real_and_unknown_gauge_fail_closed_without_planner(tmp_path, monkeypatch):
    source = tmp_path / "scene"
    _make_real(source)

    def forbidden(*args, **kwargs):
        pytest.fail("Real scene must not reach planner or renderer")

    monkeypatch.setattr("gaussian_park_world.planning.delivery.plan_and_track", forbidden)
    for batch in (False, True):
        output = tmp_path / f"gate_{batch}"
        result = generate(source, output, batch=batch)
        assert result["trajectories"] == result["geometry_layouts"] == 0
        assert result["failure_reasons"] == {"BLOCKED_GEOMETRY": 1}
        assert result["source_classification"] == REAL_SOURCE
        assert result["source_classifications"] == [REAL_SOURCE]
        assert result["trajectory_source_classifications"] == []
        assert _index(output)["source_scenes"][0]["source_classification"] == REAL_SOURCE
        assert not list(output.glob("*.npz"))


@pytest.mark.parametrize("field,value", [
    ("source_classification", SOURCE),
    ("source_classifications", [SOURCE]),
    ("trajectory_source_classifications", [SOURCE]),
])
def test_real_gate_rejects_forged_classification(tmp_path, field, value):
    source, output = tmp_path / "scene", tmp_path / "experts"
    _make_real(source)
    generate(source, output, batch=True)
    index = _index(output)
    index[field] = value
    _write(output / "experts.json", index)
    with pytest.raises(ValueError, match="SOURCE_CLASSIFICATION_MISMATCH"):
        validate_experts(output)


@pytest.fixture
def mixed_delivery(tmp_path):
    sources, output = tmp_path / "scenes", tmp_path / "experts"
    sources.mkdir()
    make_fixture(sources / "fixture", "fixture")
    _make_real(sources / "real")
    request = dict(trajectory_id="short", start=[-5., 0., 0., 0.], goal=[-3.8, 0., 0., 0.])
    generate(sources, output, requests=[request], budget=50, retries=0)
    return output


def test_mixed_source_and_accepted_trajectory_classifications(mixed_delivery):
    index = _index(mixed_delivery)
    result = validate_experts(mixed_delivery)
    assert result["source_classification"] == "MIXED"
    assert result["source_classifications"] == sorted([SOURCE, REAL_SOURCE])
    assert result["trajectory_source_classifications"] == [SOURCE]
    assert result["trajectories"] == 1
    assert result["failure_reasons"] == {"BLOCKED_GEOMETRY": 1}
    assert {s["scene_id"]: s["source_classification"] for s in index["source_scenes"]} == {
        "fixture": SOURCE, "real": REAL_SOURCE}
    assert index["trajectories"][0]["source_classification"] == SOURCE


@pytest.mark.parametrize("change", [
    lambda i: i.__setitem__("source_classification", SOURCE),
    lambda i: i.__setitem__("source_classifications", [SOURCE]),
    lambda i: i.__setitem__("trajectory_source_classifications", [REAL_SOURCE]),
    lambda i: i["source_scenes"][1].__setitem__("source_classification", SOURCE),
    lambda i: i["trajectories"][0].__setitem__("source_classification", REAL_SOURCE),
])
def test_mixed_source_rejects_forged_classification(mixed_delivery, change):
    index = _index(mixed_delivery)
    change(index)
    _write(mixed_delivery / "experts.json", index)
    with pytest.raises(ValueError, match="SOURCE_CLASSIFICATION_MISMATCH"):
        validate_experts(mixed_delivery)


def test_batch_covers_all_four_sources_with_three_layouts(tmp_path):
    sources, output = tmp_path / "scenes", tmp_path / "experts"
    sources.mkdir()
    for i in range(4):
        make_fixture(sources / f"fixture_{i}", f"fixture_{i}", variant=2 * i)
    request = dict(trajectory_id="short", start=[-5., 0., 0., 0.], goal=[-3.8, 0., 0., 0.])
    result = generate(sources, output, batch=True, requests=[request], budget=50, retries=0)
    index = _index(output)
    assert result["maps"] == result["requests"] == result["trajectories"] == 4
    assert result["geometry_layouts"] == 3 and result["failures"] == 0
    assert [m["layout"] for m in index["maps"]] == [0, 1, 2, 0]
    assert {e["scene_id"] for e in index["trajectories"]} == {f"fixture_{i}" for i in range(4)}
    assert validate_experts(output)["valid"]
    index["maps"].pop()
    index["requests"].pop()
    index["trajectories"].pop()
    _write(output / "experts.json", index)
    with pytest.raises(ValueError, match="UNACCOUNTED_SOURCE_SCENE"):
        validate_experts(output)


def test_unverified_fixture_camera_is_unavailable(tmp_path):
    source, output = tmp_path / "scene", tmp_path / "experts"
    make_fixture(source, "fixture")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["cameras"][0]["T_world_from_optical"][2][3] = 2.
    _write(source / "manifest.json", manifest)
    generate(source, output, requests=[dict(trajectory_id="short", start=[-5., 0., 0., 0.],
                                           goal=[-3.8, 0., 0., 0.])], budget=50, retries=0)
    entry = _index(output)["trajectories"][0]
    assert entry["camera_calibration"] == "unavailable"
    with np.load(output / entry["file"], allow_pickle=False) as arrays:
        assert "camera_pose" not in arrays
        assert "vehicle_pose" in arrays


def test_extra_unindexed_output_rejected(delivery):
    _, output = delivery
    (output / "extra.npz").touch()
    with pytest.raises(ValueError, match="UNINDEXED_OUTPUT_FILES"):
        validate_experts(output)


def test_existing_episode_builder_reads_delivery_without_rendering(delivery, tmp_path, monkeypatch):
    from gaussian_park_world import data_pipeline

    _, experts = delivery
    captured = []

    def capture(output, scene, map, trajectory, trajectory_id, split, goal, *, backend):
        assert set(trajectory) == {"states", "actions", "dt", "status", "metadata"}
        assert trajectory["states"].shape[0] == trajectory["actions"].shape[0] + 1
        assert trajectory["status"] == "SUCCESS"
        captured.append(trajectory_id)

    monkeypatch.setattr(data_pipeline, "write_episode", capture)
    monkeypatch.setattr(data_pipeline, "validate_dataset", lambda _: {"test_capture_only": True})
    data_pipeline.build_episodes(experts, tmp_path / "captured_episodes")
    assert captured == [_index(experts)["trajectories"][0]["trajectory_id"]]
