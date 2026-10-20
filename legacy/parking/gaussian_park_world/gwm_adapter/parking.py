"""CPU parking observation/sequence runtime, not an official GWM model.

Preparation alone can open the synthetic teacher and current camera. Runtime
readers use lazy per-frame archive members, never teacher/diagnostic archives.
Windows have disjoint future targets and one issued command for each successor.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from ..dataset.offline import load_scene, read_json
from ..scenes.contract import sha256, write_json
from ..world_model.geometry import validate_intrinsics, validate_transform
from . import episodes as ep
from .observations import MODEL_FIELDS, extract_observation, validate_observation


OBS_FORMAT = "S19_CAUSAL_SYNTHETIC_TEACHER_OBSERVATIONS_V1"
WINDOW_FORMAT = "S19_PARKING_SEQUENCES_V1"
FRAME_FIELDS = MODEL_FIELDS | {"rgb", "alpha", "camera_pose"}
ACTION_FIELDS = {"actions", "command_issue_time", "effective_start", "effective_end"}
INPUT_FIELDS = {"rgb", "alpha", "gaussians", "gaussian_mask", "history_mask", "history_time_s",
                "source_time_s", "available_at_s", "camera_pose", "K", "current_time_s", "actions",
                "action_mask", "dt_s", "command_issue_time", "effective_start", "effective_end"}
TARGET_FIELDS = {"rgb", "alpha", "gaussians", "gaussian_mask", "time_mask", "timestamp_s",
                 "camera_pose", "frame_id"}


def _key(frame, field):
    return f"f{frame:04d}.{field}"


def _frame(archive, frame):
    return {name: archive[_key(frame, name)].copy() for name in FRAME_FIELDS}


def _check_frame(data):
    if set(data) != FRAME_FIELDS:
        raise ValueError("OBSERVATION_FIELDS")
    validate_observation({k: data[k] for k in MODEL_FIELDS})
    rgb, alpha = data["rgb"], data["alpha"]
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
            or alpha.shape != rgb.shape[:2] or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError("OBSERVATION_IMAGE_DOMAIN")
    validate_transform(data["camera_pose"])


def _check_actions(data, timestamps):
    if set(data) != ACTION_FIELDS:
        raise ValueError("ACTION_FIELDS")
    actions = data["actions"]
    if (actions.shape != (len(timestamps) - 1, 3) or actions.dtype != np.float64
            or not np.isfinite(actions).all() or np.any(actions[:, 2] <= 0)
            or np.any(abs(actions[:, 0]) > 1.) or np.any(abs(actions[:, 1]) > .55)):
        raise ValueError("PARKING_ACTION_WIDTH_DOMAIN_OR_DT")
    if (not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
            or not np.allclose(np.diff(timestamps), actions[:, 2], atol=1e-10, rtol=0)):
        raise ValueError("ACTION_DT_ALIGNMENT")
    for field, target in (("command_issue_time", timestamps[:-1]), ("effective_start", timestamps[:-1]),
                          ("effective_end", timestamps[1:])):
        if (data[field].shape != target.shape or data[field].dtype != np.float64
                or not np.allclose(data[field], target, atol=1e-10, rtol=0)):
            raise ValueError("ACTION_INTERVAL_ALIGNMENT:" + field)


def prepare(package, output, *, contribution_threshold=1e-4, max_gaussians=32):
    """Export every current frame using a disclosed bounded teacher oracle."""
    started = time.perf_counter()
    package, output = Path(package).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_summary = ep.validate_package(package)
    source_index = read_json(package / "manifest.json")
    # Reject non-synthetic or uncalibrated sources before opening a teacher.
    for relative in source_index["episodes"]:
        source = read_json(ep._inside(package, relative) / "manifest.json")
        if source["source_classification"] != "SYNTHETIC_PLANNING_FIXTURE":
            raise ValueError("REAL_SOURCE_TIME_AND_GEOMETRY_PENDING")
    output.mkdir()
    entries, selected_counts = [], []
    for i, relative in enumerate(source_index["episodes"]):
        source_dir = ep._inside(package, relative)
        source = read_json(source_dir / "manifest.json")
        original = source["source_manifest_snapshot"]
        scene, canonical = load_scene(Path(original["scene_manifest"]).parent)
        if scene["world_gauge"] != dict(id=scene["world_gauge"]["id"], metric_scale="verified",
                                        units="meters", axes="x_forward_y_left_z_up", confidence="synthetic_by_construction"):
            raise ValueError("SYNTHETIC_METRIC_GAUGE_REQUIRED")
        data = ep._arrays(source_dir / "supervision.npz")
        n, count = source["frame_count"], source["action_count"]
        calibration = source["calibration"]
        observations, diagnostics = {}, []
        metadata = None
        for frame in range(n):
            t = float(data["timestamp"][frame])
            current = extract_observation(canonical, data["camera_pose"][frame], calibration["K"],
                       calibration["image_size_wh"], t, t, t,
                       source_classification=scene["source_classification"],
                       max_gaussians=max_gaussians, contribution_threshold=contribution_threshold)
            rgb = np.rint(np.clip(current["diagnostics"]["rgb"], 0, 1) * 255).astype(np.uint8)
            if (not np.array_equal(rgb, data["rgb"][frame])
                    or not np.allclose(current["diagnostics"]["alpha"], data["alpha"][frame], atol=1e-7, rtol=0)):
                raise ValueError("CURRENT_ORACLE_RENDER_SOURCE_MISMATCH")
            payload = {k: np.asarray(current[k]) for k in MODEL_FIELDS}
            payload.update(rgb=data["rgb"][frame], alpha=data["alpha"][frame], camera_pose=data["camera_pose"][frame])
            _check_frame(payload)
            observations.update({_key(frame, k): value for k, value in payload.items()})
            diagnostics.append(current["diagnostics"])
            selected_counts.append(int(current["gaussian_mask"].sum()))
            metadata = current["metadata"]
        action_data = dict(actions=np.column_stack((data["actions"][:count], data["dt"][:count])),
                           **{k: data[k][:count] for k in ACTION_FIELDS - {"actions"}})
        _check_actions(action_data, data["timestamp"][:n])
        directory = output / f"{i:04d}"
        directory.mkdir()
        ep._save(directory / "observations.npz", observations)
        ep._save(directory / "actions.npz", action_data)
        diagnostic_arrays = {name: np.stack([d["per_gaussian"][name] for d in diagnostics])
                             for name in diagnostics[0]["per_gaussian"]}
        diagnostic_arrays["depth_z"] = np.stack([d["depth_z"] for d in diagnostics])
        ep._save(directory / "diagnostics.npz", diagnostic_arrays)
        write_json(directory / "manifest.json", dict(schema_version=1, scene_id=source["scene_id"],
                   trajectory_id=source["trajectory_id"], split=source["split"], frame_count=n, action_count=count,
                   source_episode=relative, source_manifest_sha256=sha256(source_dir / "manifest.json"),
                   scene_arrays_sha256=scene["array_sha256"], gauge_id=scene["world_gauge"]["id"],
                   calibration=calibration, observation_contract=metadata,
                   files={name: sha256(directory / name) for name in ("observations.npz", "actions.npz", "diagnostics.npz")}))
        entries.append(directory.name)
    write_json(output / "manifest.json", dict(schema_version=1, format=OBS_FORMAT, source_package=str(package),
               source_manifest_sha256=sha256(package / "manifest.json"), upstream_commit=ep.PIN,
               upstream_spec_sha256=sha256(ep.SPEC), episodes=entries, counts=source_summary["counts"],
               max_gaussians=max_gaussians, contribution_threshold=contribution_threshold,
               provenance="SYNTHETIC_TEACHER_VISIBILITY_ORACLE_NOT_RGB_RECONSTRUCTION",
               gaussian_frame="world_meters", gaussian_domain="activated_RGB_NOT_SH_unit_wxyz",
               first_party_masks="VALIDATED; never forwarded to unmodified official codec",
               official_runtime_binding="PENDING", real_scene_geometry="BLOCKED_GEOMETRY",
               real_expert_trajectories=0, training="NOT_RUN", gpu="NOT_USED"))
    summary = validate_observations(output)
    summary.update(selected_min=min(selected_counts), selected_max=max(selected_counts),
                   selected_total=sum(selected_counts), prepare_seconds=time.perf_counter() - started,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(output / "validation.json", summary)
    return summary


def _manifest(root, format):
    m = read_json(Path(root) / "manifest.json")
    if (m.get("schema_version") != 1 or m.get("format") != format or m.get("upstream_commit") != ep.PIN
            or m.get("official_runtime_binding") != "PENDING"
            or m.get("training") != "NOT_RUN" or m.get("gpu") != "NOT_USED"):
        raise ValueError("S19_CONTRACT")
    return m


def validate_observations(root):
    root = Path(root)
    m = _manifest(root, OBS_FORMAT)
    package = Path(m["source_package"])
    if sha256(package / "manifest.json") != m["source_manifest_sha256"] or sha256(ep.SPEC) != m["upstream_spec_sha256"]:
        raise ValueError("S19_SOURCE_HASH")
    ep.validate_package(package)
    source_index = read_json(package / "manifest.json")
    if len(m["episodes"]) != len(source_index["episodes"]):
        raise ValueError("EPISODE_COVERAGE")
    identities, aliases, scenes = set(), {}, set()
    frames = actions = 0
    for relative, source_relative in zip(m["episodes"], source_index["episodes"]):
        directory = ep._inside(root, relative)
        current = read_json(directory / "manifest.json")
        source_dir = ep._inside(package, source_relative)
        source = read_json(source_dir / "manifest.json")
        if (current["source_episode"] != source_relative or sha256(source_dir / "manifest.json") != current["source_manifest_sha256"]
                or any(current[k] != source[k] for k in ("scene_id", "trajectory_id", "split", "frame_count", "action_count"))
                or current["calibration"] != source["calibration"]):
            raise ValueError("EPISODE_SOURCE_BINDING")
        identity = (current["scene_id"], current["trajectory_id"])
        if identity in identities:
            raise ValueError("DUPLICATE_EPISODE")
        identities.add(identity)
        for key in (current["scene_id"], current["gauge_id"], current["scene_arrays_sha256"]):
            if key in aliases and aliases[key] != current["split"]:
                raise ValueError("SCENE_SPLIT_LEAKAGE")
            aliases[key] = current["split"]
        scenes.add(current["scene_id"])
        for name, digest in current["files"].items():
            if sha256(ep._inside(directory, name)) != digest:
                raise ValueError("OBSERVATION_FILE_HASH:" + name)
        n, count = current["frame_count"], current["action_count"]
        original = ep._arrays(source_dir / "supervision.npz")
        scene, canonical = load_scene(Path(source["source_manifest_snapshot"]["scene_manifest"]).parent)
        if (current["gauge_id"] != scene["world_gauge"]["id"] or current["scene_arrays_sha256"] != scene["array_sha256"]
                or scene["source_classification"] != "SYNTHETIC_PLANNING_FIXTURE"):
            raise ValueError("GAUGE_OR_SYNTHETIC_CLASSIFICATION")
        diagnostics = ep._arrays(directory / "diagnostics.npz")
        with np.load(directory / "observations.npz", allow_pickle=False) as archive:
            if set(archive.files) != {_key(i, k) for i in range(n) for k in FRAME_FIELDS}:
                raise ValueError("OBSERVATION_ARCHIVE_FIELDS")
            timestamps = []
            for i in range(n):
                obs = _frame(archive, i)
                _check_frame(obs)
                timestamps.append(float(obs["timestamp_s"]))
                for key in ("rgb", "alpha", "camera_pose"):
                    if not np.array_equal(obs[key], original[key][i]):
                        raise ValueError("CURRENT_FRAME_SOURCE_BINDING:" + key)
                t = float(original["timestamp"][i])
                if float(obs["timestamp_s"]) != t:
                    raise ValueError("CURRENT_FRAME_TIMESTAMP")
                expected = extract_observation(canonical, original["camera_pose"][i], current["calibration"]["K"],
                           current["calibration"]["image_size_wh"], t, t, t,
                           source_classification=scene["source_classification"], max_gaussians=m["max_gaussians"],
                           contribution_threshold=m["contribution_threshold"])
                ep._equal({k: obs[k] for k in MODEL_FIELDS}, {k: np.asarray(expected[k]) for k in MODEL_FIELDS}, "VISIBLE_SET_SOURCE")
                if current["observation_contract"] != expected["metadata"]:
                    raise ValueError("GAUSSIAN_DOMAIN_OR_ORACLE_CONTRACT")
                for key, value in expected["diagnostics"]["per_gaussian"].items():
                    if not np.array_equal(diagnostics[key][i], value):
                        raise ValueError("CONTRIBUTION_DIAGNOSTIC:" + key)
                if not np.array_equal(diagnostics["depth_z"][i], expected["diagnostics"]["depth_z"]):
                    raise ValueError("DEPTH_DIAGNOSTIC")
        actual_actions = ep._arrays(directory / "actions.npz")
        _check_actions(actual_actions, np.asarray(timestamps))
        expected_actions = dict(actions=np.column_stack((original["actions"][:count], original["dt"][:count])),
                                **{k: original[k][:count] for k in ACTION_FIELDS - {"actions"}})
        ep._equal(actual_actions, expected_actions, "LOSSLESS_PARKING_ACTIONS")
        frames += n
        actions += count
    counts = dict(scenes=len(scenes), trajectories=len(identities), frames=frames, actions=actions)
    if counts != m["counts"] or counts != source_index["counts"]:
        raise ValueError("S19_COUNTS")
    return dict(status="PASS", scope="CAUSAL_SYNTHETIC_ORACLE_ONLY", counts=counts,
                official_runtime_binding="PENDING", real_scene_geometry="BLOCKED_GEOMETRY", training="NOT_RUN", gpu="NOT_USED")


def export_windows(observations, output, *, history=2, horizon=4):
    observations, output = Path(observations).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if any(type(v) is not int or not 1 <= v <= 32 for v in (history, horizon)):
        raise ValueError("WINDOW_LIMIT_1_TO_32")
    validate_observations(observations)
    corpus = _manifest(observations, OBS_FORMAT)
    windows = []
    for relative in corpus["episodes"]:
        entry = read_json(ep._inside(observations, relative) / "manifest.json")
        windows.extend(dict(episode=relative, anchor=anchor, split=entry["split"], scene_id=entry["scene_id"],
                            trajectory_id=entry["trajectory_id"]) for anchor in range(0, entry["action_count"], horizon))
    output.mkdir()
    write_json(output / "manifest.json", dict(schema_version=1, format=WINDOW_FORMAT, upstream_commit=ep.PIN,
               observations=str(observations), observations_manifest_sha256=sha256(observations / "manifest.json"),
               history=history, horizon=horizon, windows=windows, counts=corpus["counts"],
               input_domain="normalized_RGB_before_GaussianPredictor._process_obs_NOT_outer_forward",
               gaussian_domain="activated_RGB_NOT_SH_unit_wxyz_world_meters", action_order=["signed_speed_mps", "steering_rad", "dt_s"],
               official_runtime_binding="PENDING", training="NOT_RUN", gpu="NOT_USED"))
    summary = validate_windows(output)
    write_json(output / "validation.json", summary)
    return summary


def validate_windows(root):
    root = Path(root)
    m = _manifest(root, WINDOW_FORMAT)
    corpus_root = Path(m["observations"])
    if sha256(corpus_root / "manifest.json") != m["observations_manifest_sha256"]:
        raise ValueError("WINDOW_SOURCE_HASH")
    corpus = _manifest(corpus_root, OBS_FORMAT)
    if any(type(m[k]) is not int or not 1 <= m[k] <= 32 for k in ("history", "horizon")):
        raise ValueError("WINDOW_LIMIT_1_TO_32")
    if m["gaussian_domain"] != "activated_RGB_NOT_SH_unit_wxyz_world_meters" or m["action_order"] != ["signed_speed_mps", "steering_rad", "dt_s"]:
        raise ValueError("WINDOW_DOMAIN")
    expected = []
    for relative in corpus["episodes"]:
        em = read_json(ep._inside(corpus_root, relative) / "manifest.json")
        expected.extend(dict(episode=relative, anchor=a, split=em["split"], scene_id=em["scene_id"],
                             trajectory_id=em["trajectory_id"]) for a in range(0, em["action_count"], m["horizon"]))
    if m["windows"] != expected or m["counts"] != corpus["counts"]:
        raise ValueError("WINDOW_DUPLICATION_SPLIT_OR_TARGET_ALIAS")
    return dict(status="PASS", windows=len(expected), counts=m["counts"], history=m["history"], horizon=m["horizon"],
                official_runtime_binding="PENDING", training="NOT_RUN", gpu="NOT_USED")


def validate_inputs(data):
    if set(data) != INPUT_FIELDS:
        raise ValueError("PARKING_INPUT_ALLOWLIST")
    if any(not isinstance(v, np.ndarray) or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("PARKING_INPUT_FINITE_ARRAYS")
    rgb, gs, hm, gm = (data[k] for k in ("rgb", "gaussians", "history_mask", "gaussian_mask"))
    if (hm.ndim != 1 or not hm.size or rgb.ndim != 4 or gs.ndim != 3
            or not 1 <= gs.shape[1] <= 32):
        raise ValueError("HISTORY_SHAPE_DOMAIN_OR_MASK")
    h, g = hm.size, gs.shape[1]
    height, width = rgb.shape[2:]
    history_specs = {"rgb": ((h, 3, height, width), np.float32),
                     "alpha": ((h, 1, height, width), np.float32),
                     "gaussians": ((h, g, 14), np.float32), "gaussian_mask": ((h, g), bool),
                     "history_mask": ((h,), bool), "history_time_s": ((h,), np.float64),
                     "source_time_s": ((h, g), np.float64), "available_at_s": ((h, g), np.float64),
                     "camera_pose": ((h, 4, 4), np.float64), "K": ((3, 3), np.float64),
                     "current_time_s": ((), np.float64)}
    if any(data[k].shape != shape or data[k].dtype != dtype for k, (shape, dtype) in history_specs.items()):
        raise ValueError("HISTORY_SHAPE_DOMAIN_OR_MASK")
    mask, actions = data["action_mask"], data["actions"]
    if mask.ndim != 1 or not mask.size:
        raise ValueError("PARKING_ACTION_WIDTH_OR_MASK")
    n = mask.size
    if mask.dtype != bool or actions.shape != (n, 3) or actions.dtype != np.float32:
        raise ValueError("PARKING_ACTION_WIDTH_OR_MASK")
    if data["dt_s"].shape != (n,) or data["dt_s"].dtype != np.float64:
        raise ValueError("PARKING_DT")
    if any(data[k].shape != (n,) or data[k].dtype != np.float64
           for k in ("command_issue_time", "effective_start", "effective_end")):
        raise ValueError("PARKING_INTERVAL")
    validate_intrinsics(data["K"], (width, height))
    if (not hm[-1] or np.any(np.diff(hm.astype(int)) < 0)
            or any(np.any((data[k] < 0) | (data[k] > 1)) for k in ("rgb", "alpha"))):
        raise ValueError("HISTORY_SHAPE_DOMAIN_OR_MASK")
    for key in ("rgb", "alpha", "gaussians", "gaussian_mask", "source_time_s", "available_at_s",
                "camera_pose", "history_time_s"):
        if np.any(data[key][~hm] != 0):
            raise ValueError("MASKED_HISTORY_VALUES:" + key)
    if (np.any(data["history_time_s"][hm] > data["current_time_s"])
            or data["history_time_s"][-1] != data["current_time_s"]
            or np.any(np.diff(data["history_time_s"][hm]) <= 0)):
        raise ValueError("FUTURE_HISTORY")
    for i in range(h):
        if hm[i]:
            validate_transform(data["camera_pose"][i])
        validate_observation(dict(features=gs[i], gaussian_mask=gm[i], source_time_s=data["source_time_s"][i],
                                  available_at_s=data["available_at_s"][i], timestamp_s=data["history_time_s"][i]))
    count = int(mask.sum())
    if not count or not np.array_equal(mask, np.arange(n) < count):
        raise ValueError("INVALID_ACTION_MASK")
    if np.any(data["dt_s"][:count] <= 0) or not np.array_equal(actions[:, 2], data["dt_s"].astype(np.float32)):
        raise ValueError("PARKING_DT")
    if (np.any(np.abs(actions[:count, 0]) > np.float32(1.))
            or np.any(np.abs(actions[:count, 1]) > np.float32(.55))
            or np.any(actions[:count, 2] <= 0)):
        raise ValueError("PARKING_ACTION_DOMAIN")
    for key in ("actions", "dt_s", "command_issue_time", "effective_start", "effective_end"):
        if np.any(data[key][count:] != 0):
            raise ValueError("MASKED_ACTION_VALUES")
    timeline = float(data["current_time_s"]) + np.r_[0., np.cumsum(data["dt_s"][:count])]
    for key, target in (("command_issue_time", timeline[:-1]), ("effective_start", timeline[:-1]), ("effective_end", timeline[1:])):
        if not np.allclose(data[key][:count], target, atol=1e-10, rtol=0):
            raise ValueError("PARKING_INTERVAL")


def validate_targets(targets, inputs):
    """Validate successor supervision separately from the model input allowlist."""
    if set(targets) != TARGET_FIELDS:
        raise ValueError("PARKING_TARGET_ALLOWLIST")
    if any(not isinstance(v, np.ndarray) or not np.isfinite(v).all() for v in targets.values()):
        raise ValueError("PARKING_TARGET_FINITE_ARRAYS")
    f = inputs["action_mask"].size
    height, width = inputs["rgb"].shape[-2:]
    n = inputs["gaussians"].shape[1]
    specs = {"rgb": ((f, 3, height, width), np.float32), "alpha": ((f, 1, height, width), np.float32),
             "gaussians": ((f, n, 14), np.float32), "gaussian_mask": ((f, n), bool),
             "time_mask": ((f,), bool), "timestamp_s": ((f,), np.float64),
             "camera_pose": ((f, 4, 4), np.float64), "frame_id": ((f,), np.int64)}
    if any(targets[k].shape != shape or targets[k].dtype != dtype for k, (shape, dtype) in specs.items()):
        raise ValueError("PARKING_TARGET_SHAPE_OR_DTYPE")
    mask = targets["time_mask"]
    if not np.array_equal(mask, inputs["action_mask"]):
        raise ValueError("PARKING_TARGET_MASK")
    if any(np.any(value[~mask] != 0) for value in targets.values()):
        raise ValueError("MASKED_TARGET_VALUES")
    if any(np.any((targets[k] < 0) | (targets[k] > 1)) for k in ("rgb", "alpha")):
        raise ValueError("PARKING_TARGET_IMAGE_DOMAIN")
    times, ids = targets["timestamp_s"][mask], targets["frame_id"][mask]
    if (not np.array_equal(times, inputs["effective_end"][mask])
            or np.any(times <= inputs["current_time_s"]) or np.any(np.diff(times) <= 0)):
        raise ValueError("T_PLUS_ONE_TARGET_ALIGNMENT")
    if np.any(ids <= 0) or np.any(np.diff(ids) != 1):
        raise ValueError("PARKING_TARGET_FRAME_IDS")
    for i in np.flatnonzero(mask):
        validate_transform(targets["camera_pose"][i])
        # Targets carry no source clocks; validate their activated features at frame time.
        clocks = np.where(targets["gaussian_mask"][i], targets["timestamp_s"][i], 0.)
        validate_observation(dict(features=targets["gaussians"][i], gaussian_mask=targets["gaussian_mask"][i],
                                  source_time_s=clocks, available_at_s=clocks, timestamp_s=targets["timestamp_s"][i]))


class ParkingSequenceDataset:
    """Map-style dataset; runtime opens current/past frame members only for inputs.

    __getitem__ returns {inputs,targets}. rollout_inputs is a separate path that
    never reads future frames or opens source supervision/teacher/diagnostics.
    Full offline provenance/oracle validation is deliberately a separate CLI.
    """

    def __init__(self, root, split=None):
        self.root = Path(root)
        validate_windows(self.root)
        self.manifest = _manifest(self.root, WINDOW_FORMAT)
        self.corpus = Path(self.manifest["observations"])
        if split not in (None, "train", "val", "test"):
            raise ValueError("INVALID_SPLIT")
        self.windows = [w for w in self.manifest["windows"] if split is None or w["split"] == split]

    def __len__(self):
        return len(self.windows)

    def _episode(self, index):
        w = self.windows[index]
        directory = ep._inside(self.corpus, w["episode"])
        m = read_json(directory / "manifest.json")
        for name in ("observations.npz", "actions.npz"):
            if sha256(directory / name) != m["files"][name]:
                raise ValueError("RUNTIME_INPUT_HASH:" + name)
        return w, directory, m

    def rollout_inputs(self, index):
        w, directory, m = self._episode(index)
        h, f = self.manifest["history"], self.manifest["horizon"]
        anchor = w["anchor"]
        width, height = m["calibration"]["image_size_wh"]
        corpus = _manifest(self.corpus, OBS_FORMAT)
        n = corpus["max_gaussians"]
        result = dict(rgb=np.zeros((h, 3, height, width), np.float32), alpha=np.zeros((h, 1, height, width), np.float32),
                      gaussians=np.zeros((h, n, 14), np.float32), gaussian_mask=np.zeros((h, n), bool),
                      source_time_s=np.zeros((h, n)), available_at_s=np.zeros((h, n)), history_mask=np.zeros(h, bool),
                      history_time_s=np.zeros(h), camera_pose=np.zeros((h, 4, 4)), K=np.asarray(m["calibration"]["K"], dtype=float),
                      current_time_s=np.array(0.), actions=np.zeros((f, 3), np.float32), action_mask=np.zeros(f, bool),
                      dt_s=np.zeros(f), command_issue_time=np.zeros(f), effective_start=np.zeros(f), effective_end=np.zeros(f))
        with np.load(directory / "observations.npz", allow_pickle=False) as archive:
            for slot, frame in enumerate(range(anchor - h + 1, anchor + 1)):
                if frame < 0:
                    continue
                data = _frame(archive, frame)
                _check_frame(data)
                result["rgb"][slot] = data["rgb"].transpose(2, 0, 1).astype(np.float32) / np.float32(255.)
                result["alpha"][slot, 0] = data["alpha"]
                result["gaussians"][slot] = data["features"].astype(np.float32)
                for name in ("gaussian_mask", "source_time_s", "available_at_s", "camera_pose"):
                    result[name][slot] = data[name]
                result["history_mask"][slot] = True
                result["history_time_s"][slot] = data["timestamp_s"]
        count = min(f, m["action_count"] - anchor)
        actions = ep._arrays(directory / "actions.npz")
        result["actions"][:count] = actions["actions"][anchor:anchor + count].astype(np.float32)
        result["dt_s"][:count] = actions["actions"][anchor:anchor + count, 2]
        result["action_mask"][:count] = True
        result["current_time_s"] = np.array(result["history_time_s"][-1])
        for name in ACTION_FIELDS - {"actions"}:
            result[name][:count] = actions[name][anchor:anchor + count]
        validate_inputs(result)
        return result

    def __getitem__(self, index):
        inputs = self.rollout_inputs(index)
        w, directory, m = self._episode(index)
        f = self.manifest["horizon"]
        h, width = inputs["rgb"].shape[-2:]
        n = inputs["gaussians"].shape[-2]
        targets = dict(rgb=np.zeros((f, 3, h, width), np.float32), alpha=np.zeros((f, 1, h, width), np.float32),
                       gaussians=np.zeros((f, n, 14), np.float32), gaussian_mask=np.zeros((f, n), bool),
                       time_mask=inputs["action_mask"].copy(), timestamp_s=np.zeros(f),
                       camera_pose=np.zeros((f, 4, 4)), frame_id=np.zeros(f, np.int64))
        with np.load(directory / "observations.npz", allow_pickle=False) as archive:
            for slot in np.flatnonzero(targets["time_mask"]):
                frame = w["anchor"] + 1 + int(slot)
                data = _frame(archive, frame)
                _check_frame(data)
                targets["rgb"][slot] = data["rgb"].transpose(2, 0, 1).astype(np.float32) / np.float32(255.)
                targets["alpha"][slot, 0] = data["alpha"]
                targets["gaussians"][slot] = data["features"].astype(np.float32)
                targets["gaussian_mask"][slot] = data["gaussian_mask"]
                targets["timestamp_s"][slot] = data["timestamp_s"]
                targets["camera_pose"][slot] = data["camera_pose"]
                targets["frame_id"][slot] = frame
        if not np.array_equal(targets["frame_id"][targets["time_mask"]],
                              w["anchor"] + 1 + np.flatnonzero(inputs["action_mask"])):
            raise ValueError("PARKING_TARGET_FRAME_IDS")
        validate_targets(targets, inputs)
        return dict(inputs=inputs, targets=targets)


def collate_parking(samples):
    """CPU tensors only. Masks are preserved, checked and never dropped."""
    import torch
    if not samples:
        raise ValueError("EMPTY_BATCH")
    for sample in samples:
        if set(sample) != {"inputs", "targets"}:
            raise ValueError("BATCH_ALLOWLIST")
        validate_inputs(sample["inputs"])
        validate_targets(sample["targets"], sample["inputs"])
    return {role: {k: torch.from_numpy(np.stack([s[role][k] for s in samples])) for k in samples[0][role]}
            for role in ("inputs", "targets")}


def make_dataloader(root, *, split=None, batch_size=2):
    from torch.utils.data import DataLoader
    return DataLoader(ParkingSequenceDataset(root, split), batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_parking)


def official_guard(inputs, *, profile="standalone_vae_config"):
    """Explicit refusal, not a stub forward or an RGB-to-SH relabelling."""
    validate_inputs(inputs)
    spec = read_json(ep.SPEC)
    if spec["commit"] != ep.PIN or profile not in ("standalone_vae_config", "world_model_defaults"):
        raise ValueError("OFFICIAL_PROFILE_OR_PIN")
    if not inputs["gaussian_mask"].all() or not inputs["history_mask"].all() or not inputs["action_mask"].all():
        raise RuntimeError("OFFICIAL_MASK_BINDING_PENDING: codec FPS/attention do not consume Gaussian masks")
    raise RuntimeError("OFFICIAL_RUNTIME_BINDING_PENDING: no direct precomputed Gaussian predictor API; RGB is not SH; dependencies absent")


def inspect_batch(windows, batch_size=2, split="train", *, all_batches=False):
    started = time.perf_counter()
    loader = make_dataloader(windows, split=None if split == "all" else split, batch_size=batch_size)
    iterator = iter(loader)
    batch = next(iterator)
    batches, samples = 1, len(batch["inputs"]["rgb"])
    actions = int(batch["inputs"]["action_mask"].sum())
    targets = int(batch["targets"]["time_mask"].sum())
    if all_batches:
        for current in iterator:
            batches += 1
            samples += len(current["inputs"]["rgb"])
            actions += int(current["inputs"]["action_mask"].sum())
            targets += int(current["targets"]["time_mask"].sum())
    spec = read_json(ep.SPEC)
    return dict(status="FIRST_PARTY_DATALOADER_PASS", windows=len(loader.dataset), batch_size=len(batch["inputs"]["rgb"]),
                shapes={role: {k: list(v.shape) for k, v in batch[role].items()} for role in batch},
                dtypes={role: {k: str(v.dtype) for k, v in batch[role].items()} for role in batch},
                devices=sorted({str(v.device) for role in batch.values() for v in role.values()}),
                consumed=dict(batches=batches, samples=samples, actions=actions, targets=targets),
                official_profiles={k: spec["codec"][k] for k in ("standalone_vae_config", "world_model_defaults")},
                image_boundary="normalized_before__process_obs; outer_predictor_forward_would_divide_again",
                inspect_seconds=time.perf_counter() - started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                official_runtime_binding="PENDING", training="NOT_RUN", gpu="NOT_USED")


def roundtrip(windows, output):
    m = _manifest(windows, WINDOW_FORMAT)
    validate_windows(windows)
    corpus = _manifest(m["observations"], OBS_FORMAT)
    validate_observations(m["observations"])
    dataset = ParkingSequenceDataset(windows)
    targets, command_count = set(), 0
    for i, w in enumerate(dataset.windows):
        sample = dataset[i]
        for slot in np.flatnonzero(sample["targets"]["time_mask"]):
            identity = (w["episode"], int(sample["targets"]["frame_id"][slot]))
            if identity in targets:
                raise ValueError("DUPLICATE_TARGET_ALIAS")
            targets.add(identity)
            command_count += 1
    if command_count != corpus["counts"]["actions"]:
        raise ValueError("ROUNDTRIP_ACTION_COVERAGE")
    restored = ep.roundtrip(corpus["source_package"], output)
    summary = dict(status="PASS", windows=len(dataset), counts=corpus["counts"], unique_successor_targets=len(targets),
                   exact_raw_episode_arrays=restored["exact_arrays_and_dtypes"],
                   model_precision="RGB/observed features/actions float32; raw Gaussian/actions/clock float64 preserved",
                   official_runtime_binding="PENDING", training="NOT_RUN", gpu="NOT_USED")
    write_json(Path(output) / "parking_roundtrip.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--package", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--threshold", type=float, default=1e-4)
    windows = commands.add_parser("export-windows")
    windows.add_argument("--observations", required=True)
    windows.add_argument("--output", required=True)
    windows.add_argument("--history", type=int, default=2)
    windows.add_argument("--horizon", type=int, default=4)
    validate = commands.add_parser("validate")
    validate.add_argument("--path", required=True)
    inspect = commands.add_parser("inspect-batch")
    inspect.add_argument("--windows", required=True)
    inspect.add_argument("--batch-size", type=int, default=2)
    inspect.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    inspect.add_argument("--all-batches", action="store_true")
    inspect.add_argument("--output")
    rt = commands.add_parser("roundtrip")
    rt.add_argument("--windows", required=True)
    rt.add_argument("--output", required=True)
    guard = commands.add_parser("official-check")
    guard.add_argument("--windows", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.package, args.output, contribution_threshold=args.threshold)
    elif args.command == "export-windows":
        result = export_windows(args.observations, args.output, history=args.history, horizon=args.horizon)
    elif args.command == "validate":
        m = read_json(Path(args.path) / "manifest.json")
        result = validate_observations(args.path) if m["format"] == OBS_FORMAT else validate_windows(args.path)
    elif args.command == "inspect-batch":
        result = inspect_batch(args.windows, args.batch_size, args.split, all_batches=args.all_batches)
        if args.output:
            write_json(args.output, result)
    elif args.command == "roundtrip":
        result = roundtrip(args.windows, args.output)
    else:
        try:
            official_guard(ParkingSequenceDataset(args.windows).rollout_inputs(0))
        except RuntimeError as error:
            print(json.dumps(dict(official_runtime_binding="PENDING", reason=str(error), training="NOT_RUN", gpu="NOT_USED")))
            return 2
        raise AssertionError("An unimplemented official runtime cannot succeed")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
