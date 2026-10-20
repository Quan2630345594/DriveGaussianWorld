"""Episode conversion without model execution or robot-action impersonation.

Full scene Gaussians and future frames are reference/supervision only. The
model-facing reader opens inputs.npz only and never constructs Gaussian inputs
from the privileged map. Native-resolution RGB is retained without resizing.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from gaussian_park_world.dataset.offline import load_scene, read_json, validate_dataset, validate_episode
from gaussian_park_world.scenes.contract import sha256, write_json
from .gaussians import sample_gaussians


PIN = "433f750ee227e034c3e5916c274fc4b696fb3a39"
SPEC = Path(__file__).with_name("upstream_spec.json")
FRAME_KEYS = {"states", "timestamp", "frame_id", "vehicle_pose", "camera_pose", "collision", "rgb", "alpha", "coverage"}
ACTION_KEYS = {"actions", "dt", "command_issue_time", "effective_start", "effective_end"}
INPUT_KEYS = {"initial_rgb", "initial_alpha", "initial_timestamp", "K", "actions", "dt", "action_mask",
              "command_issue_time", "effective_start", "effective_end"}


def _inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("UNSAFE_PACKAGE_PATH")
    target = (Path(root) / path).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("UNSAFE_PACKAGE_PATH")
    return target


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _frame_limits(manifest, frame_key="frame_count", action_key="action_count"):
    if any(type(manifest[k]) is not int or not 1 <= manifest[k] <= 4096
           for k in (frame_key, action_key)):
        raise ValueError("FRAME_LIMIT_4096")


def _split_binding(bindings, scene, split):
    for key in (("scene", scene["scene_id"]), ("arrays", scene["array_sha256"]),
                ("gauge", scene["world_gauge"]["id"])):
        if key in bindings and bindings[key] != split:
            raise ValueError("SCENE_SPLIT_LEAKAGE")
        bindings[key] = split


def _save(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _equal(actual, expected, reason):
    if set(actual) != set(expected):
        raise ValueError(reason + ":KEYS")
    for key in expected:
        if (actual[key].dtype != expected[key].dtype or actual[key].shape != expected[key].shape
                or not np.array_equal(actual[key], expected[key])):
            raise ValueError(reason + ":" + key)


def _padded(data, frames, actions):
    if set(data) != FRAME_KEYS | ACTION_KEYS:
        raise ValueError("UNSUPPORTED_EPISODE_ARRAYS")
    result = {}
    n, count = len(data["timestamp"]), len(data["dt"])
    if n > frames or count > actions or n != count + 1:
        raise ValueError("PADDING_LENGTH")
    for key, value in data.items():
        total = frames if key in FRAME_KEYS else actions
        result[key] = np.zeros((total,) + value.shape[1:], dtype=value.dtype)
        result[key][:len(value)] = value
    result["frame_mask"] = np.arange(frames) < n
    result["action_mask"] = np.arange(actions) < count
    return result


def _inputs(data, K, max_actions):
    padded = _padded(data, max_actions + 1, max_actions)
    return dict(initial_rgb=data["rgb"][:1].copy(), initial_alpha=data["alpha"][:1].copy(),
                initial_timestamp=data["timestamp"][:1].copy(), K=np.asarray(K, dtype=np.float64),
                **{k: padded[k] for k in ACTION_KEYS | {"action_mask"}})


def validate_rollout_inputs(data):
    """Strict allowlist for initial observation + factual proposed action plan.

    Future action intervals are allowed plans, not future pose measurements.
    No Gaussian map, future image, target, state, source IDs or poses are allowed.
    """
    if set(data) != INPUT_KEYS:
        raise ValueError("CAUSAL_INPUT_KEYS")
    if any(not isinstance(v, np.ndarray) or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("NONFINITE_INPUT")
    rgb, alpha, actions, dt, mask = (data[k] for k in ("initial_rgb", "initial_alpha", "actions", "dt", "action_mask"))
    if (rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3 or min(rgb.shape[1:3]) <= 0
            or rgb.dtype != np.uint8 or alpha.shape != rgb.shape[:-1]
            or np.any((alpha < 0) | (alpha > 1)) or data["initial_timestamp"].shape != (1,)
            or actions.ndim != 2 or actions.shape[1] != 2 or actions.dtype != np.float64
            or dt.shape != (len(actions),) or dt.dtype != np.float64
            or mask.shape != dt.shape or mask.dtype != bool):
        raise ValueError("INPUT_SHAPE_DTYPE_OR_DOMAIN")
    count = int(mask.sum())
    if count < 1 or not np.array_equal(mask, np.arange(len(mask)) < count):
        raise ValueError("INVALID_ACTION_MASK")
    if np.any(dt[:count] <= 0) or np.any(dt[count:] != 0) or np.any(actions[count:] != 0):
        raise ValueError("INVALID_ACTION_PADDING_OR_DT")
    K = data["K"]
    if (K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.array_equal(K[2], [0., 0., 1.])):
        raise ValueError("INVALID_INTRINSICS")
    timeline = data["initial_timestamp"][0] + np.r_[0., np.cumsum(dt[:count])]
    for key, expected in (("command_issue_time", timeline[:-1]), ("effective_start", timeline[:-1]),
                          ("effective_end", timeline[1:])):
        values = data[key]
        if (values.shape != dt.shape or np.any(values[count:] != 0)
                or not np.allclose(values[:count], expected, atol=1e-10, rtol=0)):
            raise ValueError("INVALID_ACTION_INTERVAL:" + key)
    return count


def rgb_to_gwm_images(rgb):
    """Verified image-only channel/range conversion; no resize or model call.

    Returns [B,T,3,H,W] float32 for [B,T,H,W,3] uint8. Native 64x48 fixture
    resolution is NOT silently presented as the DROID default 128x128.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 5 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB[B,T,H,W,3]")
    return rgb.transpose(0, 1, 4, 2, 3).astype(np.float32) / np.float32(255.)


def require_official_action_binding(actions):
    """Fail explicitly instead of padding vehicle commands into robot controls."""
    if not isinstance(actions, np.ndarray) or actions.ndim < 2 or actions.shape[-1] != 10:
        raise ValueError("OFFICIAL_ACTION_DIMENSION_MISMATCH: vehicle width 2 is not DROID width 10")
    raise ValueError("OFFICIAL_ACTION_BINDING_PENDING: width alone does not verify robot semantics")


def load_rollout_inputs(episode):
    """Read only causal payload; no supervision, scene or model runtime access."""
    episode = Path(episode)
    m = read_json(episode / "manifest.json")
    if sha256(episode / "inputs.npz") != m["files"]["inputs.npz"]:
        raise ValueError("INPUT_HASH_MISMATCH")
    inputs = _arrays(episode / "inputs.npz")
    validate_rollout_inputs(inputs)
    return inputs


def convert(dataset, output, *, gaussian_count=2048):
    start = time.perf_counter()
    dataset, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    index = read_json(dataset / "dataset.json")
    if len(index["episodes"]) > 1024:
        raise ValueError("EPISODE_LIMIT_1024")
    directories = [_inside(dataset, relative) for relative in index["episodes"]]
    originals = []
    for directory in directories:
        manifest = read_json(_inside(directory, "manifest.json"))
        _inside(directory, "frames.npz")
        _frame_limits(manifest)
        originals.append((directory, manifest))
    bindings = {}
    for _, manifest in originals:
        scene = read_json(manifest["scene_manifest"])
        _split_binding(bindings, scene, manifest["split"])
    source_summary = validate_dataset(dataset)
    max_frames = max(m["frame_count"] for _, m in originals)
    max_actions = max(m["action_count"] for _, m in originals)
    output.mkdir()
    (output / "scenes").mkdir()
    (output / "episodes").mkdir()
    spec = read_json(SPEC)
    if spec["commit"] != PIN:
        raise ValueError("UPSTREAM_PIN_MISMATCH")
    write_json(output / "upstream_spec.json", spec)
    scene_refs, episode_refs = {}, []
    for i, (directory, source) in enumerate(originals):
        sid = source["scene_id"]
        if sid not in scene_refs:
            scene_path = Path(source["scene_manifest"])
            scene, canonical = load_scene(scene_path.parent)
            ref = output / "scenes" / f"{len(scene_refs):04d}"
            ref.mkdir()
            # Preserve canonical precision for roundtrip. Samples are references only.
            _save(ref / "canonical.npz", canonical)
            sample = sample_gaussians(canonical, gaussian_count, allow_padding=True)
            _save(ref / "sample.npz", {k: sample[k] for k in ("features", "valid_mask")})
            write_json(ref / "manifest.json", dict(schema_version=1, scene_id=sid,
                       source_manifest=str(scene_path), source_manifest_sha256=sha256(scene_path),
                       source_arrays_sha256=scene["array_sha256"], source_manifest_snapshot=scene,
                       canonical_sha256=sha256(ref / "canonical.npz"), sample_sha256=sha256(ref / "sample.npz"),
                       sample_metadata=sample["metadata"], role="PRIVILEGED_REFERENCE_ONLY_NOT_CAUSAL_INPUT"))
            scene_refs[sid] = str(ref.relative_to(output))
        scene_ref = read_json(output / scene_refs[sid] / "manifest.json")
        if scene_ref["source_manifest_sha256"] != source["scene_manifest_sha256"]:
            raise ValueError("SCENE_ID_MULTIPLE_SOURCES")
        data = _arrays(directory / "frames.npz")
        inputs = _inputs(data, source["renderer"]["K"], max_actions)
        validate_rollout_inputs(inputs)
        ep = output / "episodes" / f"{i:04d}"
        ep.mkdir()
        _save(ep / "inputs.npz", inputs)
        _save(ep / "supervision.npz", _padded(data, max_frames, max_actions))
        write_json(ep / "manifest.json", dict(schema_version=1, scene_id=sid,
                   trajectory_id=source["trajectory_id"], split=source["split"],
                   source_episode=str(directory), source_manifest_sha256=sha256(directory / "manifest.json"),
                   source_frames_sha256=sha256(directory / "frames.npz"), source_manifest_snapshot=source,
                   scene_reference=scene_refs[sid], frame_count=len(data["timestamp"]), action_count=len(data["dt"]),
                   files={name: sha256(ep / name) for name in ("inputs.npz", "supervision.npz")},
                   calibration=source["renderer"], source_classification=source["source_classification"],
                   roles=dict(inputs="initial_observation_and_action_plan", supervision="TRAINER_EVALUATOR_ONLY")))
        episode_refs.append(str(ep.relative_to(output)))
    write_json(output / "manifest.json", dict(schema_version=1, format="GWM_ORIENTED_INTERMEDIATE_NOT_OFFICIAL_RUNTIME",
               official_commit=PIN, official_runtime="PENDING", official_action_binding="PENDING",
               causal_gaussian_input="PENDING", gaussian_parameter_binding="PENDING", native_action_dim=2,
               action_units=["signed_speed_command_mps", "steering_rad"], dt_units="seconds",
               image_policy="lossless_native_uint8_RGB_no_resize; optional float32_B_T_C_H_W_div255_reader",
               temporal_policy="native issued transitions; NOT DROID shifted 10/24 windows",
               gaussian_policy="activated RGB/wxyz neutral14; full reference and zero-padded samples are never model input",
               mask_policy="True is valid; prefix masks; zeros are padding not repeated observations",
               source_dataset=str(dataset), source_index_sha256=sha256(dataset / "dataset.json"),
               source_index_snapshot=index, upstream_spec_sha256=sha256(output / "upstream_spec.json"),
               max_frames=max_frames, max_actions=max_actions, scenes=scene_refs, episodes=episode_refs,
               counts=dict(scenes=source_summary["scenes"], trajectories=len(originals), frames=source_summary["frames"],
                           actions=sum(m["action_count"] for _, m in originals)),
               fixture_limitations=index.get("fixture_limitations", "unspecified"), training="NOT_RUN", gpu="NOT_USED"))
    summary = validate_package(output)
    summary["conversion_seconds"] = time.perf_counter() - start
    summary["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(output / "validation.json", summary)
    return summary


def validate_package(root):
    root = Path(root)
    m = read_json(root / "manifest.json")
    if (m.get("schema_version") != 1 or m.get("official_commit") != PIN
            or m.get("format") != "GWM_ORIENTED_INTERMEDIATE_NOT_OFFICIAL_RUNTIME"
            or any(m.get(k) != "PENDING" for k in ("official_runtime", "official_action_binding", "causal_gaussian_input", "gaussian_parameter_binding"))
            or m.get("native_action_dim") != 2 or m.get("training") != "NOT_RUN" or m.get("gpu") != "NOT_USED"):
        raise ValueError("PACKAGE_CONTRACT")
    if sha256(root / "upstream_spec.json") != m["upstream_spec_sha256"]:
        raise ValueError("UPSTREAM_SPEC_HASH_MISMATCH")
    spec = read_json(root / "upstream_spec.json")
    if spec["commit"] != PIN or spec["runtime_verified"] is not False:
        raise ValueError("UPSTREAM_RUNTIME_NOT_VERIFIED")
    source_root = Path(m["source_dataset"])
    if sha256(source_root / "dataset.json") != m["source_index_sha256"]:
        raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    source_index = read_json(source_root / "dataset.json")
    if source_index != m["source_index_snapshot"]:
        raise ValueError("SOURCE_INDEX_SNAPSHOT_MISMATCH")
    if len(source_index["episodes"]) > 1024 or len(m["episodes"]) > 1024:
        raise ValueError("EPISODE_LIMIT_1024")
    _frame_limits(m, "max_frames", "max_actions")
    source_paths = {_inside(source_root, p) for p in source_index["episodes"]}
    episode_paths = [_inside(root, p) for p in m["episodes"]]
    scene_paths = {sid: _inside(root, p) for sid, p in m["scenes"].items()}
    episode_manifests = []
    for ep in episode_paths:
        em = read_json(_inside(ep, "manifest.json"))
        _frame_limits(em)
        _inside(root, em["scene_reference"])
        for name in ("inputs.npz", "supervision.npz"):
            _inside(ep, name)
        episode_manifests.append((ep, em))
    if {Path(em["source_episode"]).resolve() for _, em in episode_manifests} != source_paths or not episode_manifests:
        raise ValueError("SOURCE_EPISODE_COVERAGE_MISMATCH")
    for source in source_paths:
        _inside(source, "manifest.json")
        _inside(source, "frames.npz")
    scenes = {}
    for sid, relative in m["scenes"].items():
        ref = scene_paths[sid]
        sm = read_json(ref / "manifest.json")
        source = Path(sm["source_manifest"])
        if sha256(source) != sm["source_manifest_sha256"] or sm["scene_id"] != sid:
            raise ValueError("SOURCE_SCENE_HASH_OR_ID_MISMATCH")
        scene, canonical = load_scene(source.parent)
        if scene != sm["source_manifest_snapshot"] or scene["array_sha256"] != sm["source_arrays_sha256"]:
            raise ValueError("SOURCE_SCENE_SNAPSHOT_MISMATCH")
        scenes[sid] = scene
        for name, key in (("canonical.npz", "canonical_sha256"), ("sample.npz", "sample_sha256")):
            if sha256(ref / name) != sm[key]:
                raise ValueError("GAUSSIAN_HASH_MISMATCH")
        _equal(_arrays(ref / "canonical.npz"), canonical, "CANONICAL_ROUNDTRIP")
        expected = sample_gaussians(canonical, sm["sample_metadata"]["output_count"], allow_padding=True)
        _equal(_arrays(ref / "sample.npz"), {k: expected[k] for k in ("features", "valid_mask")}, "GAUSSIAN_MASK_OR_SAMPLE")
        if sm["sample_metadata"] != expected["metadata"] or sm["role"] != "PRIVILEGED_REFERENCE_ONLY_NOT_CAUSAL_INPUT":
            raise ValueError("GAUSSIAN_REFERENCE_ROLE")
    identities, scene_splits, seen_sources = set(), {}, set()
    bindings = {}
    frames = actions = 0
    for ep, em in episode_manifests:
        identity = (em["scene_id"], em["trajectory_id"])
        if identity in identities:
            raise ValueError("DUPLICATE_EPISODE")
        identities.add(identity)
        sid, split = em["scene_id"], em["split"]
        if sid in scene_splits and scene_splits[sid] != split:
            raise ValueError("SCENE_SPLIT_LEAKAGE")
        scene_splits[sid] = split
        _split_binding(bindings, scenes[sid], split)
        source = Path(em["source_episode"]).resolve()
        if sha256(source / "manifest.json") != em["source_manifest_sha256"] or sha256(source / "frames.npz") != em["source_frames_sha256"]:
            raise ValueError("SOURCE_EPISODE_HASH_MISMATCH")
        original = read_json(source / "manifest.json")
        if original != em["source_manifest_snapshot"] or any(em[k] != original[k] for k in ("scene_id", "trajectory_id", "split", "frame_count", "action_count", "source_classification")):
            raise ValueError("SOURCE_EPISODE_METADATA_MISMATCH")
        if em["calibration"] != original["renderer"] or em["scene_reference"] != m["scenes"][sid]:
            raise ValueError("CALIBRATION_OR_GAUGE_REFERENCE_MISMATCH")
        if read_json(_inside(root, em["scene_reference"]) / "manifest.json")["source_manifest_sha256"] != original["scene_manifest_sha256"]:
            raise ValueError("SCENE_GAUGE_BINDING_MISMATCH")
        validate_episode(source)
        seen_sources.add(source)
        for name in ("inputs.npz", "supervision.npz"):
            if sha256(ep / name) != em["files"][name]:
                raise ValueError("EPISODE_FILE_HASH_MISMATCH:" + name)
        inputs = load_rollout_inputs(ep)
        data = _arrays(source / "frames.npz")
        _equal(inputs, _inputs(data, original["renderer"]["K"], m["max_actions"]), "CAUSAL_SOURCE_BINDING")
        supervision = _arrays(ep / "supervision.npz")
        _equal(supervision, _padded(data, m["max_frames"], m["max_actions"]), "SUPERVISION_MASK_OR_ROUNDTRIP")
        frames += em["frame_count"]
        actions += em["action_count"]
    if seen_sources != source_paths or not identities:
        raise ValueError("SOURCE_EPISODE_COVERAGE_MISMATCH")
    counts = dict(scenes=len(scene_splits), trajectories=len(identities), frames=frames, actions=actions)
    if counts != m["counts"] or set(scene_splits) != set(m["scenes"]):
        raise ValueError("PACKAGE_COUNTS_MISMATCH")
    return dict(status="PASS", scope="GWM-oriented intermediate integrity; official runtime/action/Gaussian binding PENDING",
                counts=counts, scene_splits=scene_splits, gaussian_causal_input="PENDING", training="NOT_RUN", gpu="NOT_USED")


def roundtrip(package, output):
    package, output = Path(package), Path(output)
    if output.exists():
        raise FileExistsError(output)
    validate_package(package)
    m = read_json(package / "manifest.json")
    output.mkdir()
    entries = []
    for i, relative in enumerate(m["episodes"]):
        source = package / relative
        em = read_json(source / "manifest.json")
        supervision = _arrays(source / "supervision.npz")
        restored = {k: supervision[k][:em["frame_count"] if k in FRAME_KEYS else em["action_count"]].copy()
                    for k in FRAME_KEYS | ACTION_KEYS}
        original_data = _arrays(Path(em["source_episode"]) / "frames.npz")
        _equal(restored, original_data, "ROUNDTRIP_EXACT_ARRAYS")
        destination = output / f"{i:04d}"
        destination.mkdir()
        _save(destination / "frames.npz", restored)
        manifest = dict(em["source_manifest_snapshot"])
        manifest["frames_sha256"] = sha256(destination / "frames.npz")
        write_json(destination / "manifest.json", manifest)
        entries.append(destination.name)
    index = dict(m["source_index_snapshot"])
    index["episodes"] = entries
    write_json(output / "dataset.json", index)
    result = validate_dataset(output)
    summary = dict(status="PASS", exact_arrays_and_dtypes=True, counts=m["counts"],
                   vendor_neutral_validation=result, source_scene_refs="preserved, not relocated",
                   training="NOT_RUN", gpu="NOT_USED")
    write_json(output / "roundtrip.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert_cmd = commands.add_parser("convert")
    convert_cmd.add_argument("--dataset", required=True)
    convert_cmd.add_argument("--output", required=True)
    convert_cmd.add_argument("--gaussian-count", type=int, default=2048)
    validate_cmd = commands.add_parser("validate")
    validate_cmd.add_argument("--package", required=True)
    rt = commands.add_parser("roundtrip")
    rt.add_argument("--package", required=True)
    rt.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        result = convert(args.dataset, args.output, gaussian_count=args.gaussian_count)
    elif args.command == "validate":
        result = validate_package(args.package)
    else:
        result = roundtrip(args.package, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
