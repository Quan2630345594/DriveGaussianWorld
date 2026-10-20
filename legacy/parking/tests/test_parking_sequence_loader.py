"""Bounded CPU sequence-runtime tests, not an official model forward."""

from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from gaussian_park_world.dataset.offline import read_json, write_episode
from gaussian_park_world.gwm_adapter import episodes, parking
from gaussian_park_world.planning import fixture_map, plan_and_track
from gaussian_park_world.scenes import make_fixture, sha256, write_json


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def replace_json(path, value):
    path.unlink()
    write_json(path, value)


def assert_exact(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], strict=True, err_msg=key)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("parking-sequences")
    source = root / "source"
    source.mkdir()
    entries = []
    for sid, split, variant, distances in (
        ("parking-train", "train", 0, (.6, 1.2)),
        ("parking-test", "test", 2, (-.6,)),
    ):
        scene = source / sid
        make_fixture(scene, sid, variant=variant)
        planning_map = fixture_map(sid, variant=variant)
        for distance in distances:
            name = f"{sid}-{distance}"
            goal = [distance, 0., 0., 0.]
            trajectory = plan_and_track(planning_map, [0., 0., 0., 0.], goal,
                                        max_expansions=12, max_retries=0)
            assert trajectory["status"] == "SUCCESS", trajectory
            write_episode(source / name, scene, planning_map, trajectory, name, split, goal)
            entries.append(name)
    write_json(source / "dataset.json", dict(episodes=entries, failures=[],
               scene_splits={"parking-train": "train", "parking-test": "test"},
               fixture_limitations="Small synthetic CPU runtime fixture only"))
    episodes.convert(source, root / "package", gaussian_count=32)
    parking.prepare(root / "package", root / "observations", max_gaussians=32)
    parking.export_windows(root / "observations", root / "windows", history=3, horizon=4)
    return root


@pytest.fixture
def corpus(tmp_path, prepared):
    # Only mutable S19 artifacts are copied; the shared S17/S18 source stays untouched.
    observations = tmp_path / "observations"
    windows = tmp_path / "windows"
    shutil.copytree(prepared / "observations", observations)
    shutil.copytree(prepared / "windows", windows)
    path = windows / "manifest.json"
    manifest = read_json(path)
    manifest["observations"] = str(observations)
    manifest["observations_manifest_sha256"] = sha256(observations / "manifest.json")
    replace_json(path, manifest)
    return observations, windows


def test_source_counts_domains_and_scene_splits(prepared):
    observations, windows = prepared / "observations", prepared / "windows"
    summary = parking.validate_observations(observations)
    manifest = read_json(observations / "manifest.json")
    counts = summary["counts"]
    assert counts["scenes"] == 2 and counts["trajectories"] == 3
    assert counts["frames"] == counts["actions"] + 3
    assert counts["frames"] < 60  # Never render the full 270-frame acceptance corpus here.
    assert counts == read_json(prepared / "package" / "manifest.json")["counts"]
    assert manifest["gaussian_frame"] == "world_meters"
    assert manifest["gaussian_domain"] == "activated_RGB_NOT_SH_unit_wxyz"
    assert manifest["provenance"] == "SYNTHETIC_TEACHER_VISIBILITY_ORACLE_NOT_RGB_RECONSTRUCTION"
    wm = read_json(windows / "manifest.json")
    assert wm["action_order"] == ["signed_speed_mps", "steering_rad", "dt_s"]
    assert wm["gaussian_domain"] == "activated_RGB_NOT_SH_unit_wxyz_world_meters"
    assert wm["input_domain"] == "normalized_RGB_before_GaussianPredictor._process_obs_NOT_outer_forward"
    for result in (summary, parking.validate_windows(windows)):
        assert result["status"] == "PASS"
        assert result["official_runtime_binding"] == "PENDING"
        assert result["training"] == "NOT_RUN" and result["gpu"] == "NOT_USED"
    splits = {}
    for relative in manifest["episodes"]:
        directory = observations / relative
        em = read_json(directory / "manifest.json")
        source = prepared / "package" / em["source_episode"]
        assert em["source_manifest_sha256"] == sha256(source / "manifest.json")
        for name, digest in em["files"].items():
            assert digest == sha256(directory / name)
        splits.setdefault(em["scene_id"], set()).add(em["split"])
        contract = em["observation_contract"]
        assert contract["domains"]["colors"] == "activated_rgb_0_1_NOT_SH"
        assert contract["official_sh_feed"] is False
        assert contract["teacher_oracle"] is True
        assert contract["diagnostics_model_input"] is False
    assert splits == {"parking-train": {"train"}, "parking-test": {"test"}}
    train = parking.ParkingSequenceDataset(windows, "train")
    test = parking.ParkingSequenceDataset(windows, "test")
    assert {w["scene_id"] for w in train.windows}.isdisjoint(w["scene_id"] for w in test.windows)
    assert len(train) + len(test) == len(parking.ParkingSequenceDataset(windows))
    assert len(parking.ParkingSequenceDataset(windows, "val")) == 0
    with pytest.raises(ValueError, match="INVALID_SPLIT"):
        parking.ParkingSequenceDataset(windows, "validation")


def test_getitem_normalization_actions_padding_successors_and_nonoverlap(prepared):
    dataset = parking.ParkingSequenceDataset(prepared / "windows")
    seen = set()
    padded_history = padded_future = reverse = False
    for index, window in enumerate(dataset.windows):
        sample = dataset[index]
        assert set(sample) == {"inputs", "targets"}
        data, targets = sample["inputs"], sample["targets"]
        assert set(data) == parking.INPUT_FIELDS
        parking.validate_inputs(data)
        anchor = window["anchor"]
        directory = prepared / "observations" / window["episode"]
        em = read_json(directory / "manifest.json")
        source = arrays(prepared / "package" / em["source_episode"] / "supervision.npz")
        count = min(4, em["action_count"] - anchor)
        history = np.arange(anchor - 2, anchor + 1)
        np.testing.assert_array_equal(data["history_mask"], history >= 0)
        np.testing.assert_array_equal(data["action_mask"], np.arange(4) < count)
        np.testing.assert_array_equal(targets["time_mask"], data["action_mask"])
        for slot, frame in enumerate(history):
            if frame < 0:
                padded_history = True
                for key in ("rgb", "alpha", "gaussians", "gaussian_mask", "camera_pose",
                            "source_time_s", "available_at_s", "history_time_s"):
                    assert not data[key][slot].any(), key
            else:
                expected = source["rgb"][frame].transpose(2, 0, 1).astype(np.float32) / np.float32(255)
                np.testing.assert_array_equal(data["rgb"][slot], expected, strict=True)
                np.testing.assert_array_equal(data["camera_pose"][slot], source["camera_pose"][frame])
        assert data["rgb"].dtype == data["gaussians"].dtype == data["actions"].dtype == np.float32
        assert isinstance(data["current_time_s"], np.ndarray)
        assert data["current_time_s"].shape == () and data["current_time_s"].dtype == np.float64
        assert 0 <= data["rgb"].min() <= data["rgb"].max() <= 1
        expected = np.column_stack((source["actions"], source["dt"]))[anchor:anchor + count]
        np.testing.assert_array_equal(data["actions"][:count], expected.astype(np.float32), strict=True)
        reverse |= bool(np.any(data["actions"][:, 0] < 0))
        for key, source_key in (("dt_s", "dt"), ("command_issue_time", "command_issue_time"),
                                ("effective_start", "effective_start"), ("effective_end", "effective_end")):
            assert data[key].dtype == np.float64
            np.testing.assert_array_equal(data[key][:count], source[source_key][anchor:anchor + count], strict=True)
            assert not data[key][count:].any()
        np.testing.assert_array_equal(targets["frame_id"][:count], np.arange(anchor + 1, anchor + count + 1))
        for slot in range(count):
            frame = anchor + slot + 1
            identity = (window["episode"], frame)
            assert identity not in seen
            seen.add(identity)
            np.testing.assert_array_equal(targets["rgb"][slot],
                source["rgb"][frame].transpose(2, 0, 1).astype(np.float32) / np.float32(255))
            np.testing.assert_array_equal(targets["camera_pose"][slot], source["camera_pose"][frame])
            assert targets["timestamp_s"][slot] == source["timestamp"][frame]
            assert targets["timestamp_s"][slot] == data["effective_end"][slot]
            assert targets["timestamp_s"][slot] > data["current_time_s"]
        if count < 4:
            padded_future = True
            assert not data["actions"][count:].any()
            for value in targets.values():
                assert not value[count:].any()
    assert padded_history and padded_future and reverse
    assert len(seen) == dataset.manifest["counts"]["actions"]


def test_actual_dataloader_collates_cpu_tensors_and_preserves_clocks(prepared):
    loader = parking.make_dataloader(prepared / "windows", batch_size=2, split="train")
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert loader.num_workers == 0
    batch = next(iter(loader))
    samples = [loader.dataset[i] for i in range(2)]
    for role, fields in batch.items():
        for key, tensor in fields.items():
            assert isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
            np.testing.assert_array_equal(tensor.numpy(), np.stack([s[role][key] for s in samples]), strict=True)
    assert batch["inputs"]["rgb"].shape == (2, 3, 3, 48, 64)
    assert batch["inputs"]["gaussians"].shape == (2, 3, 32, 14)
    assert batch["inputs"]["actions"].shape == (2, 4, 3)
    for key in ("dt_s", "command_issue_time", "effective_start", "effective_end"):
        assert batch["inputs"][key].dtype == torch.float64
    for key in ("history_mask", "gaussian_mask", "action_mask"):
        assert batch["inputs"][key].dtype == torch.bool
    with pytest.raises(ValueError, match="EMPTY_BATCH"):
        parking.collate_parking([])
    with pytest.raises(ValueError, match="BATCH_ALLOWLIST"):
        parking.collate_parking([dict(samples[0], teacher={})])


@pytest.mark.parametrize("anchor_kind", ["initial", "later"])
def test_rollout_opens_only_allowed_archives_and_past_current_members(prepared, monkeypatch, anchor_kind):
    dataset = parking.ParkingSequenceDataset(prepared / "windows")
    index = 0 if anchor_kind == "initial" else next(i for i, w in enumerate(dataset.windows) if w["anchor"] >= 4)
    window = dataset.windows[index]
    expected = dataset.rollout_inputs(index)
    directory = prepared / "observations" / window["episode"]
    accessed, opened = [], []
    real_load, real_json, real_hash = np.load, parking.read_json, parking.sha256

    class PrefixArchive:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            frame, field = key.split(".")
            assert max(0, window["anchor"] - 2) <= int(frame[1:]) <= window["anchor"]
            assert field in parking.FRAME_FIELDS
            accessed.append(key)
            return self.archive[key]

    def guarded_load(path, *args, **kwargs):
        path = Path(path)
        assert path in {directory / "observations.npz", directory / "actions.npz"}
        assert kwargs.get("allow_pickle") is False
        opened.append(path.name)
        archive = real_load(path, *args, **kwargs)
        return PrefixArchive(archive) if path.name == "observations.npz" else archive

    def guarded_json(path):
        assert Path(path) in {directory / "manifest.json", prepared / "observations" / "manifest.json"}
        return real_json(path)

    def guarded_hash(path):
        assert Path(path) in {directory / "observations.npz", directory / "actions.npz"}
        return real_hash(path)

    def forbidden(*args, **kwargs):
        pytest.fail("Rollout accessed teacher/source/supervision/diagnostics or offline validator")

    with monkeypatch.context() as patch:
        patch.setattr(parking.np, "load", guarded_load)
        patch.setattr(parking, "read_json", guarded_json)
        patch.setattr(parking, "sha256", guarded_hash)
        for owner, name in ((parking, "validate_observations"), (parking, "load_scene"),
                            (parking, "extract_observation"), (episodes, "validate_package")):
            patch.setattr(owner, name, forbidden)
        actual = dataset.rollout_inputs(index)
    assert_exact(actual, expected)
    assert opened == ["observations.npz", "actions.npz"]
    frames = range(max(0, window["anchor"] - 2), window["anchor"] + 1)
    assert set(accessed) == {parking._key(frame, field) for frame in frames for field in parking.FRAME_FIELDS}
    assert len(accessed) == len(set(accessed))


@pytest.mark.parametrize("field,error", [
    ("rgb", "CURRENT_FRAME_SOURCE_BINDING:rgb"),
    ("camera_pose", "CURRENT_FRAME_SOURCE_BINDING:camera_pose"),
    ("features", "VISIBLE_SET_SOURCE:features"),
    ("available_at_s", "source_time_s <= available_at_s <= timestamp_s"),
])
def test_rehashed_future_poison_cannot_change_inputs_and_targets_require_valid_frames(corpus, field, error):
    observations, windows = corpus
    dataset = parking.ParkingSequenceDataset(windows)
    before = dataset[0]
    directory = observations / dataset.windows[0]["episode"]
    path = directory / "observations.npz"
    payload = arrays(path)
    key = parking._key(1, field)
    mask = payload[parking._key(1, "gaussian_mask")]
    assert mask.any()
    if field == "rgb":
        payload[key] ^= 255
    elif field == "camera_pose":
        payload[key][0, 3] += 10
    elif field == "features":
        payload[key][mask, 0] += 10
    else:
        payload[key][mask] = payload[parking._key(1, "timestamp_s")] + 10
    np.savez_compressed(path, **payload)
    manifest = read_json(directory / "manifest.json")
    manifest["files"][path.name] = sha256(path)
    replace_json(directory / "manifest.json", manifest)
    assert_exact(dataset.rollout_inputs(0), before["inputs"])
    if field == "available_at_s":
        with pytest.raises(ValueError, match=error):
            dataset[0]
    else:
        after = dataset[0]
        assert_exact(after["inputs"], before["inputs"])
        target_key = "gaussians" if field == "features" else field
        assert not np.array_equal(after["targets"][target_key][0], before["targets"][target_key][0])
    with pytest.raises(ValueError, match=error):
        parking.validate_observations(observations)


@pytest.mark.parametrize("damage", ["duplicate", "target_alias", "reorder", "split", "scene", "counts", "escape"])
def test_window_manifest_rejects_duplicates_aliases_reordering_and_split_edits(corpus, damage):
    _, windows = corpus
    path = windows / "manifest.json"
    manifest = read_json(path)
    if damage == "duplicate":
        manifest["windows"].append(manifest["windows"][0].copy())
    elif damage == "target_alias":
        manifest["windows"][1]["anchor"] = 1
    elif damage == "reorder":
        manifest["windows"].reverse()
    elif damage in ("split", "scene"):
        manifest["windows"][0]["split" if damage == "split" else "scene_id"] = "test" if damage == "split" else "other"
    elif damage == "escape":
        manifest["windows"][0]["episode"] = "../outside"
    else:
        manifest["counts"]["actions"] += 1
    replace_json(path, manifest)
    before = {p: sha256(p) for p in windows.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="WINDOW_DUPLICATION_SPLIT_OR_TARGET_ALIAS"):
        parking.ParkingSequenceDataset(windows)
    assert {p: sha256(p) for p in before} == before


@pytest.mark.parametrize("damage,error", [
    ("source_hash", "S19_SOURCE_HASH"), ("window_hash", "WINDOW_SOURCE_HASH"),
    ("episode_source", "EPISODE_SOURCE_BINDING"), ("split", "EPISODE_SOURCE_BINDING"),
    ("reorder", "EPISODE_SOURCE_BINDING"), ("escape", "UNSAFE_PACKAGE_PATH"),
    ("domain", "WINDOW_DOMAIN"),
])
def test_source_bindings_and_manifest_hash_failures_are_read_only(corpus, damage, error):
    observations, windows = corpus
    root = windows if damage in ("window_hash", "domain") else observations
    path = root / "manifest.json"
    manifest = read_json(path)
    if damage in ("episode_source", "split"):
        path = observations / manifest["episodes"][0] / "manifest.json"
        manifest = read_json(path)
        manifest["source_manifest_sha256" if damage == "episode_source" else "split"] = "0" * 64 if damage == "episode_source" else "test"
    elif damage == "reorder":
        manifest["episodes"].reverse()
    elif damage == "escape":
        manifest["episodes"][0] = "../outside"
    else:
        key = {"source_hash": "source_manifest_sha256", "window_hash": "observations_manifest_sha256", "domain": "gaussian_domain"}[damage]
        manifest[key] = "SH" if damage == "domain" else "0" * 64
    replace_json(path, manifest)
    before = {p: sha256(p) for root in corpus for p in root.rglob("*") if p.is_file()}
    validator = parking.validate_windows if root == windows else parking.validate_observations
    with pytest.raises(ValueError, match=error):
        validator(root)
    assert {p: sha256(p) for p in before} == before


@pytest.mark.parametrize("name", ["observations.npz", "actions.npz", "diagnostics.npz"])
def test_unrehashed_corruption_fails_without_writing(corpus, name):
    observations, windows = corpus
    dataset = parking.ParkingSequenceDataset(windows)
    directory = observations / dataset.windows[0]["episode"]
    path = directory / name
    payload = arrays(path)
    key = {"observations.npz": parking._key(0, "rgb"),
           "actions.npz": "actions", "diagnostics.npz": "depth_z"}[name]
    if name == "observations.npz":
        payload[key].flat[0] ^= 1
    else:
        payload[key].flat[0] += 1
    np.savez_compressed(path, **payload)
    before = sha256(path)
    with pytest.raises(ValueError, match="OBSERVATION_FILE_HASH:" + name):
        parking.validate_observations(observations)
    if name != "diagnostics.npz":
        with pytest.raises(ValueError, match="RUNTIME_INPUT_HASH:" + name):
            dataset.rollout_inputs(0)
    assert sha256(path) == before


@pytest.mark.parametrize("alias", ["gauge_id", "scene_arrays_sha256"])
def test_scene_aliases_cannot_cross_splits(corpus, alias):
    observations, _ = corpus
    manifest = read_json(observations / "manifest.json")
    first = read_json(observations / manifest["episodes"][0] / "manifest.json")
    path = observations / manifest["episodes"][-1] / "manifest.json"
    last = read_json(path)
    assert first["split"] != last["split"]
    last[alias] = first[alias]
    replace_json(path, last)
    with pytest.raises(ValueError, match="SCENE_SPLIT_LEAKAGE"):
        parking.validate_observations(observations)


@pytest.mark.parametrize("damage,error", [
    ("action10", "PARKING_ACTION_WIDTH_DOMAIN_OR_DT"),
    ("action2", "PARKING_ACTION_WIDTH_DOMAIN_OR_DT"),
    ("zero_dt", "PARKING_ACTION_WIDTH_DOMAIN_OR_DT"),
    ("negative_dt", "PARKING_ACTION_WIDTH_DOMAIN_OR_DT"),
    ("action_dtype", "PARKING_ACTION_WIDTH_DOMAIN_OR_DT"),
    ("dt_alignment", "ACTION_DT_ALIGNMENT"),
    ("command_issue_time", "ACTION_INTERVAL_ALIGNMENT:command_issue_time"),
    ("effective_start", "ACTION_INTERVAL_ALIGNMENT:effective_start"),
    ("effective_end", "ACTION_INTERVAL_ALIGNMENT:effective_end"),
    ("interval_dtype", "ACTION_INTERVAL_ALIGNMENT:effective_end"),
])
def test_full_validator_rejects_rehashed_malformed_action_archives(corpus, damage, error):
    observations, _ = corpus
    relative = read_json(observations / "manifest.json")["episodes"][0]
    directory = observations / relative
    path = directory / "actions.npz"
    data = arrays(path)
    if damage in ("action10", "action2"):
        data["actions"] = np.zeros((len(data["actions"]), 10 if damage == "action10" else 2))
    elif damage == "action_dtype":
        data["actions"] = data["actions"].astype(np.float32)
    elif damage == "interval_dtype":
        data["effective_end"] = data["effective_end"].astype(np.float32)
    elif damage in ("zero_dt", "negative_dt", "dt_alignment"):
        data["actions"][0, 2] = {"zero_dt": 0., "negative_dt": -.1, "dt_alignment": .123}[damage]
    else:
        data[damage][0] += .01
    np.savez_compressed(path, **data)
    manifest = read_json(directory / "manifest.json")
    manifest["files"][path.name] = sha256(path)
    replace_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match=error):
        parking.validate_observations(observations)


@pytest.mark.parametrize("damage,error", [
    ("extra", "PARKING_INPUT_ALLOWLIST"), ("nan", "PARKING_INPUT_FINITE_ARRAYS"),
    ("rgb_range", "HISTORY_SHAPE_DOMAIN_OR_MASK"), ("rgb_dtype", "HISTORY_SHAPE_DOMAIN_OR_MASK"),
    ("history_mask", "HISTORY_SHAPE_DOMAIN_OR_MASK"), ("gaussian_mask", "HISTORY_SHAPE_DOMAIN_OR_MASK"),
    ("masked_gaussians", "MASKED_HISTORY_VALUES"), ("future_history", "FUTURE_HISTORY"),
    ("source_future", "source_time_s <= available_at_s <= timestamp_s"),
    ("available_future", "source_time_s <= available_at_s <= timestamp_s"),
    ("action10", "PARKING_ACTION_WIDTH_OR_MASK"), ("action2", "PARKING_ACTION_WIDTH_OR_MASK"),
    ("action_dtype", "PARKING_ACTION_WIDTH_OR_MASK"), ("action_mask_dtype", "PARKING_ACTION_WIDTH_OR_MASK"),
    ("action_mask_empty", "INVALID_ACTION_MASK"), ("action_mask_hole", "INVALID_ACTION_MASK"),
    ("zero_dt", "PARKING_DT"), ("negative_dt", "PARKING_DT"), ("two_dt", "PARKING_DT"),
    ("dt_mismatch", "PARKING_DT"),
    ("command_issue_time", "PARKING_INTERVAL"), ("effective_start", "PARKING_INTERVAL"),
    ("effective_end", "PARKING_INTERVAL"), ("action_padding", "MASKED_ACTION_VALUES"),
    ("dt_padding", "PARKING_DT"),
])
def test_malformed_model_inputs_fail_closed(prepared, damage, error):
    dataset = parking.ParkingSequenceDataset(prepared / "windows")
    if damage in ("action_padding", "dt_padding"):
        index = next(i for i in range(len(dataset)) if not dataset.rollout_inputs(i)["action_mask"].all())
    else:
        index = 0
    data = dataset.rollout_inputs(index)
    valid = data["gaussian_mask"][-1]
    assert valid.any()
    if damage == "extra":
        data["teacher"] = np.zeros(1)
    elif damage == "nan":
        data["rgb"].flat[0] = np.nan
    elif damage == "rgb_range":
        data["rgb"].flat[0] = 255
    elif damage == "rgb_dtype":
        data["rgb"] = data["rgb"].astype(np.float64)
    elif damage in ("history_mask", "gaussian_mask", "action_mask_dtype"):
        key = "action_mask" if damage == "action_mask_dtype" else damage
        data[key] = data[key].astype(np.uint8)
    elif damage == "masked_gaussians":
        data["gaussian_mask"][0, 0] = True
    elif damage == "future_history":
        data["history_time_s"][-1] += 1
    elif damage in ("source_future", "available_future"):
        data["source_time_s" if damage == "source_future" else "available_at_s"][-1, valid] += 1
    elif damage in ("action10", "action2"):
        data["actions"] = np.zeros((4, 10 if damage == "action10" else 2), np.float32)
    elif damage == "action_dtype":
        data["actions"] = data["actions"].astype(np.float64)
    elif damage == "action_mask_empty":
        data["action_mask"][:] = False
    elif damage == "action_mask_hole":
        data["action_mask"][0] = False
    elif damage in ("zero_dt", "negative_dt"):
        data["dt_s"][0] = 0 if damage == "zero_dt" else -.1
        data["actions"][0, 2] = data["dt_s"][0]
    elif damage == "two_dt":
        data["dt_s"] = np.column_stack((data["dt_s"], data["dt_s"]))
    elif damage == "dt_mismatch":
        data["actions"][0, 2] *= 2
    elif damage in ("command_issue_time", "effective_start", "effective_end"):
        data[damage][0] += .01
    else:
        slot = int(data["action_mask"].sum())
        if damage == "action_padding":
            data["actions"][slot, :2] = 1
        else:
            data["dt_s"][slot] = 1
    with pytest.raises(ValueError, match=error):
        parking.validate_inputs(data)
    with pytest.raises(ValueError, match=error):
        parking.official_guard(data)


@pytest.mark.parametrize("profile", ["standalone_vae_config", "world_model_defaults"])
def test_official_guard_refuses_masks_and_unbound_runtime_without_gpu_forward(prepared, monkeypatch, profile):
    data = parking.ParkingSequenceDataset(prepared / "windows").rollout_inputs(0)

    def forbidden(*args, **kwargs):
        pytest.fail("Official refusal must not run a model or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    with pytest.raises(RuntimeError, match="^OFFICIAL_MASK_BINDING_PENDING:"):
        parking.official_guard(data, profile=profile)
    # A valid fully populated first-party payload still cannot claim official execution.
    for key in ("rgb", "alpha", "gaussians", "gaussian_mask", "source_time_s", "available_at_s",
                "camera_pose", "history_time_s", "history_mask"):
        data[key] = data[key][-1:].copy()
    valid = data["gaussian_mask"][0]
    for key in ("gaussians", "gaussian_mask", "source_time_s", "available_at_s"):
        data[key] = data[key][:, valid].copy()
    assert data["action_mask"].all()
    parking.validate_inputs(data)
    with pytest.raises(RuntimeError, match="^OFFICIAL_RUNTIME_BINDING_PENDING:"):
        parking.official_guard(data, profile=profile)
    with pytest.raises(ValueError, match="OFFICIAL_PROFILE_OR_PIN"):
        parking.official_guard(data, profile="invented")


def test_prepare_and_export_refuse_existing_paths_without_modification(prepared):
    for directory, operation in (
        (prepared / "observations", lambda: parking.prepare(prepared / "package", prepared / "observations")),
        (prepared / "windows", lambda: parking.export_windows(prepared / "observations", prepared / "windows")),
    ):
        before = sha256(directory / "manifest.json")
        with pytest.raises(FileExistsError):
            operation()
        assert sha256(directory / "manifest.json") == before


@pytest.mark.parametrize("field", sorted(parking.INPUT_FIELDS))
def test_input_exact_shapes_before_indexing(prepared, field):
    data = parking.ParkingSequenceDataset(prepared / "windows").rollout_inputs(0)
    data[field] = np.expand_dims(data[field], 0)
    with pytest.raises(ValueError):
        parking.validate_inputs(data)


@pytest.mark.parametrize("field", ["source_time_s", "available_at_s"])
def test_extra_clock_rows_rejected(prepared, field):
    data = parking.ParkingSequenceDataset(prepared / "windows").rollout_inputs(0)
    data[field] = np.concatenate((data[field], data[field][-1:]))
    with pytest.raises(ValueError, match="HISTORY_SHAPE"):
        parking.validate_inputs(data)


@pytest.mark.parametrize("field", ["rgb", "alpha", "gaussians", "gaussian_mask", "source_time_s",
                                 "available_at_s", "camera_pose", "history_time_s"])
def test_every_invalid_history_field_is_zero(prepared, field):
    data = parking.ParkingSequenceDataset(prepared / "windows").rollout_inputs(0)
    assert not data["history_mask"][0]
    data[field][0] = 1
    with pytest.raises(ValueError, match="MASKED_HISTORY_VALUES"):
        parking.validate_inputs(data)


@pytest.mark.parametrize("field", sorted(parking.INPUT_FIELDS))
def test_input_exact_dtypes(prepared, field):
    data = parking.ParkingSequenceDataset(prepared / "windows").rollout_inputs(0)
    dtype = np.float64 if data[field].dtype == np.float32 else np.float32
    data[field] = data[field].astype(dtype)
    with pytest.raises(ValueError):
        parking.validate_inputs(data)


@pytest.mark.parametrize("damage", ["teacher_K", "negative_focal", "skew", "principal_point", "K_bottom",
                                   "alpha2", "alpha_negative", "speed999", "reverse_speed", "steer",
                                   "pose_scale", "pose_reflection", "history_equal", "history_reverse",
                                   "history_hole", "scale_zero", "scale_negative", "quat_nonunit"])
def test_review_input_domains(prepared, damage):
    dataset = parking.ParkingSequenceDataset(prepared / "windows")
    index = next(i for i, w in enumerate(dataset.windows) if w["anchor"] >= 4)
    data = dataset.rollout_inputs(index)
    if damage == "teacher_K":
        data["K"] = np.zeros((32, 14), np.float64)
    elif damage in ("negative_focal", "skew", "principal_point", "K_bottom"):
        row, col, value = {"negative_focal": (0, 0, -1), "skew": (0, 1, 1),
                           "principal_point": (0, 2, data["rgb"].shape[-1]), "K_bottom": (2, 0, 1)}[damage]
        data["K"][row, col] = value
    elif damage.startswith("alpha"):
        data["alpha"][-1, 0, 0, 0] = 2 if damage == "alpha2" else -.1
    elif damage in ("speed999", "reverse_speed", "steer"):
        data["actions"][0, 1 if damage == "steer" else 0] = {"speed999": 999, "reverse_speed": -999, "steer": .56}[damage]
    elif damage.startswith("pose"):
        data["camera_pose"][-1, :3, 0] *= 2 if damage == "pose_scale" else -1
    elif damage in ("history_equal", "history_reverse"):
        data["history_time_s"][0] = data["history_time_s"][1] + (0 if damage == "history_equal" else .01)
    elif damage == "history_hole":
        data["history_mask"][1] = False
    else:
        row = np.flatnonzero(data["gaussian_mask"][-1])[0]
        if damage == "quat_nonunit":
            data["gaussians"][-1, row, 6:10] *= 2
        else:
            data["gaussians"][-1, row, 3] = 0 if damage == "scale_zero" else -1
    with pytest.raises(ValueError):
        parking.validate_inputs(data)


@pytest.mark.parametrize("field", sorted(parking.TARGET_FIELDS))
@pytest.mark.parametrize("damage", ["shape", "dtype", "padding"])
def test_target_exact_shapes_dtypes_and_zero_padding(prepared, field, damage):
    dataset = parking.ParkingSequenceDataset(prepared / "windows")
    index = next(i for i in range(len(dataset)) if not dataset.rollout_inputs(i)["action_mask"].all())
    sample = dataset[index]
    targets = sample["targets"]
    if damage == "shape":
        targets[field] = np.expand_dims(targets[field], 0)
    elif damage == "dtype":
        targets[field] = targets[field].astype(np.float64 if targets[field].dtype == np.float32 else np.float32)
    else:
        targets[field][np.flatnonzero(~sample["inputs"]["action_mask"])[0]] = 1
    with pytest.raises(ValueError):
        parking.validate_targets(targets, sample["inputs"])
    with pytest.raises(ValueError):
        parking.collate_parking([sample])


@pytest.mark.parametrize("damage", ["rgb_range", "alpha2", "scale_zero", "scale_negative", "quat_nonunit",
                                   "masked_gaussian", "mask_mismatch", "pose_scale", "pose_reflection",
                                   "past_time", "wrong_time", "frame_zero", "frame_gap", "nan"])
def test_mutated_batch_target_domains_rejected(prepared, damage):
    sample = parking.ParkingSequenceDataset(prepared / "windows")[0]
    targets = sample["targets"]
    row = np.flatnonzero(targets["gaussian_mask"][0])[0]
    if damage in ("rgb_range", "alpha2", "nan"):
        targets["alpha" if damage == "alpha2" else "rgb"].flat[0] = np.nan if damage == "nan" else 2
    elif damage in ("scale_zero", "scale_negative"):
        targets["gaussians"][0, row, 3] = 0 if damage == "scale_zero" else -1
    elif damage == "quat_nonunit":
        targets["gaussians"][0, row, 6:10] *= 2
    elif damage == "masked_gaussian":
        targets["gaussian_mask"][0, row] = False
    elif damage == "mask_mismatch":
        targets["time_mask"][0] = False
    elif damage.startswith("pose"):
        targets["camera_pose"][0, :3, 0] *= 2 if damage == "pose_scale" else -1
    elif damage in ("past_time", "wrong_time"):
        targets["timestamp_s"][0] = sample["inputs"]["current_time_s"] if damage == "past_time" else targets["timestamp_s"][0] + 1e-12
    else:
        targets["frame_id"][0 if damage == "frame_zero" else 1] = 0 if damage == "frame_zero" else 999
    with pytest.raises(ValueError):
        parking.collate_parking([sample])


@pytest.mark.parametrize("role", ["inputs", "targets"])
@pytest.mark.parametrize("field", ["teacher", "future_rgb", "future_gaussians", "diagnostics", "target_pose"])
def test_forbidden_future_keys_cannot_enter_batch(prepared, role, field):
    sample = parking.ParkingSequenceDataset(prepared / "windows")[0]
    sample[role][field] = np.zeros(1)
    with pytest.raises(ValueError, match="ALLOWLIST"):
        parking.collate_parking([sample])


@pytest.mark.parametrize("damage", ["clock", "scale_negative", "scale_underflow", "pose", "rgb_dtype", "timestamp"])
def test_future_frame_checked_before_and_after_cast(corpus, damage):
    observations, windows = corpus
    dataset = parking.ParkingSequenceDataset(windows)
    before = dataset.rollout_inputs(0)
    directory = observations / dataset.windows[0]["episode"]
    path = directory / "observations.npz"
    payload = arrays(path)
    mask = payload[parking._key(1, "gaussian_mask")]
    row = np.flatnonzero(mask)[0]
    if damage == "clock":
        payload[parking._key(1, "available_at_s")][mask] += 1
    elif damage.startswith("scale"):
        payload[parking._key(1, "features")][row, 3] = -1e-50 if damage == "scale_negative" else 1e-50
    elif damage == "pose":
        payload[parking._key(1, "camera_pose")][:3, 0] *= 2
    elif damage == "rgb_dtype":
        payload[parking._key(1, "rgb")] = payload[parking._key(1, "rgb")].astype(np.float32)
    else:
        payload[parking._key(1, "timestamp_s")] += 1e-12
    np.savez_compressed(path, **payload)
    manifest = read_json(directory / "manifest.json")
    manifest["files"][path.name] = sha256(path)
    replace_json(directory / "manifest.json", manifest)
    assert_exact(dataset.rollout_inputs(0), before)
    with pytest.raises(ValueError):
        dataset[0]
