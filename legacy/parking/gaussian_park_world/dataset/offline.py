"""Small auditable episode writer and fail-closed validator. No model imports."""

import json
from pathlib import Path

import numpy as np

from gaussian_park_world.scenes.contract import sha256, write_json, validate_scene
from gaussian_park_world.planning.expert import (
    integrate, collision_free, require_planning_geometry, _swept,
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_scene(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if sha256(directory / "scene.npz") != manifest["array_sha256"]:
        raise ValueError("SCENE_HASH_MISMATCH")
    with np.load(directory / "scene.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    validate_scene(manifest, arrays)
    return manifest, arrays


def vehicle_pose(state):
    x, y, yaw = state[:3]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.]])


def render_frames(scene_dir, states, *, backend="cpu_reference"):
    manifest, arrays = load_scene(scene_dir)
    camera = manifest["cameras"][0]
    K = np.asarray(camera["K"], dtype=float)
    size = camera["image_size_wh"]
    # This rig is a synthetic mounting calibration, not a DGGT vehicle calibration.
    rig = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 1.], [0., 0., 0., 1.]])
    if manifest["source_classification"] != "SYNTHETIC_PLANNING_FIXTURE":
        raise ValueError("REAL_SCENE_PLANNING=BLOCKED_GEOMETRY")
    poses = np.array([vehicle_pose(state) for state in states])
    cameras = poses @ rig
    rgb, alpha, coverage = [], [], []
    if backend == "cpu_reference":
        from gaussian_park_world.world_model.gaussian_observation_reference import render_reference_observation
        from gaussian_park_world.world_model.gaussian_state_contract import current_camera_pose, PROVENANCE
        gaussians = dict(provenance=PROVENANCE, ids=np.arange(len(arrays["means"]), dtype=np.int64),
                         centers_world_m=arrays["means"], covariances_world_m2=arrays["covariances"],
                         colors_rgb=arrays["colors"], opacities=arrays["opacities"])
        for pose in poses:
            image = render_reference_observation(gaussians, current_camera_pose(pose, rig), K, size)
            rgb.append(np.rint(np.clip(image["rgb"], 0, 1) * 255).astype(np.uint8))
            alpha.append(image["alpha"].astype(np.float32))
            coverage.append(float(np.mean(image["alpha"] > .01)))
        renderer_file = Path(__file__).parents[1] / "world_model/gaussian_observation_reference.py"
        renderer = "existing_numpy_gaussian_reference_not_production_3dgs"
    elif backend == "gsplat":
        from gaussian_park_world.rendering.renderer import Renderer
        engine = Renderer(Path(scene_dir) / "scene.npz", width=size[0], height=size[1])
        for camera_pose in cameras:
            color, opacity = engine.render(camera_pose, K)
            rgb.append(color)
            alpha.append(opacity)
            coverage.append(float(np.mean(opacity > .01)))
        renderer_file = Path(__file__).parents[1] / "rendering/renderer.py"
        renderer = "existing_cuda_gsplat_RGB_only_plus_alpha"
    else:
        raise ValueError("UNSUPPORTED_RENDERER")
    if not coverage or min(coverage) == 0:
        raise ValueError("EMPTY_VIEW")
    return dict(rgb=np.array(rgb), alpha=np.array(alpha), vehicle_pose=poses, camera_pose=cameras,
                coverage=np.array(coverage)), dict(backend=renderer, source_sha256=sha256(renderer_file),
                source_path=str(renderer_file.relative_to(Path(__file__).parents[2])),
                depth="NOT_EXPOSED", image_size_wh=size, K=K.tolist(),
                T_vehicle_from_optical=rig.tolist(),
                pixel_centers="integer_for_cpu_reference" if backend == "cpu_reference" else "backend_defined_unverified")


def write_episode(output, scene_dir, planning_map, trajectory, trajectory_id, split, goal, *, backend="cpu_reference"):
    output, scene_dir = Path(output), Path(scene_dir)
    if output.exists():
        raise FileExistsError(output)
    scene, _ = load_scene(scene_dir)
    require_planning_geometry(scene, planning_map)
    if trajectory["status"] != "SUCCESS":
        raise ValueError("Cannot publish a failed trajectory")
    states, actions, dt = (trajectory[k] for k in ("states", "actions", "dt"))
    observations, renderer = render_frames(scene_dir, states, backend=backend)
    times = np.r_[0., np.cumsum(dt)]
    arrays = dict(states=states, actions=actions, dt=dt, timestamp=times,
                  frame_id=np.arange(len(states)), command_issue_time=times[:-1],
                  effective_start=times[:-1], effective_end=times[1:],
                  collision=np.zeros(len(states), dtype=bool), **observations)
    output.mkdir(exist_ok=False)
    with (output / "frames.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = dict(schema_version=1, scene_id=scene["scene_id"], trajectory_id=trajectory_id,
                    source_classification=scene["source_classification"], split=split,
                    scene_manifest=str((scene_dir / "manifest.json").resolve()),
                    scene_manifest_sha256=sha256(scene_dir / "manifest.json"),
                    frames_file="frames.npz", frames_sha256=sha256(output / "frames.npz"),
                    frame_count=len(states), action_count=len(actions),
                    planning_map=planning_map, goal=list(goal), goal_reached=True,
                    planner="bounded_Hybrid_A_star", controller="primitive_feedforward_plus_heading_cross_track_feedback",
                    execution=trajectory["metadata"], renderer=renderer,
                    timing=dict(state="rear_axle_x_m_y_m_yaw_rad_signed_speed_mps",
                                action="signed_speed_command_mps_steering_rad", interval="[t,t+dt)",
                                integration="exact_constant_control_bicycle_arc", gear="sign_of_speed; zero_is_stop",
                                command_latency="SYNTHETIC_ZERO_DELAY", nominal_fps=10),
                    observation_gaussian="OBSERVATION_GAUSSIAN_PENDING",
                    privileged_reference="scene_manifest is renderer/evaluator-only; never model input",
                    causal_input="initial RGB/Alpha and proposed issued actions+dt only; no future poses",
                    gwm_adapter="PENDING", training="NOT_RUN")
    write_json(output / "pending_manifest.json", manifest)
    validate_episode(output, _manifest=manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_episode(directory, *, _manifest=None):
    directory = Path(directory)
    m = read_json(directory / "manifest.json") if _manifest is None else _manifest
    if m.get("schema_version") != 1 or m.get("split") not in ("train", "val", "test"):
        raise ValueError("EPISODE_SCHEMA")
    scene_path = Path(m["scene_manifest"])
    if sha256(scene_path) != m["scene_manifest_sha256"]:
        raise ValueError("SCENE_MANIFEST_HASH_MISMATCH")
    scene, _ = load_scene(scene_path.parent)
    require_planning_geometry(scene, m["planning_map"])
    if m["scene_id"] != scene["scene_id"] or m["source_classification"] != scene["source_classification"]:
        raise ValueError("SCENE_ID_OR_SOURCE_MISMATCH")
    calibration = scene["cameras"][0]
    expected_rig = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 1.], [0., 0., 0., 1.]])
    if (m["renderer"]["image_size_wh"] != calibration["image_size_wh"]
            or not np.array_equal(m["renderer"]["K"], calibration["K"])
            or not np.array_equal(m["renderer"]["T_vehicle_from_optical"], expected_rig)):
        raise ValueError("CAMERA_CALIBRATION_MISMATCH")
    execution = m["execution"]
    supported = dict(wheelbase=2.7, length=4.4, width=1.8, rear_axle_to_center=1.2, safety_margin=.2,
                     simulator="SYNTHETIC_ZERO_DELAY", speed_snap=True)
    if (any(execution.get(k) != v for k, v in supported.items())
            or execution.get("speed_snap") is not True
            or execution.get("limits") != dict(max_speed=1., max_steering=.55)
            or execution.get("goal_tolerances") != dict(position=.35, yaw=.12, speed=0.)):
        raise ValueError("UNSUPPORTED_VEHICLE_OR_EXECUTION")
    if m.get("timing") != dict(
            state="rear_axle_x_m_y_m_yaw_rad_signed_speed_mps",
            action="signed_speed_command_mps_steering_rad", interval="[t,t+dt)",
            integration="exact_constant_control_bicycle_arc", gear="sign_of_speed; zero_is_stop",
            command_latency="SYNTHETIC_ZERO_DELAY", nominal_fps=10):
        raise ValueError("UNSUPPORTED_TIMING")
    renderer_files = {
        "existing_numpy_gaussian_reference_not_production_3dgs": "world_model/gaussian_observation_reference.py",
        "existing_cuda_gsplat_RGB_only_plus_alpha": "rendering/renderer.py",
    }
    renderer_path = renderer_files.get(m["renderer"]["backend"])
    if renderer_path is None or sha256(Path(__file__).parents[1] / renderer_path) != m["renderer"]["source_sha256"]:
        raise ValueError("RENDERER_PROVENANCE_MISMATCH")
    if sha256(directory / "frames.npz") != m["frames_sha256"]:
        raise ValueError("FRAMES_HASH_MISMATCH")
    with np.load(directory / "frames.npz", allow_pickle=False) as data:
        a = {k: data[k] for k in data.files}
    n, count = m["frame_count"], m["action_count"]
    w, h = m["renderer"]["image_size_wh"]
    shapes = dict(states=(n, 4), actions=(count, 2), dt=(count,), timestamp=(n,), frame_id=(n,),
                  command_issue_time=(count,), effective_start=(count,), effective_end=(count,),
                  vehicle_pose=(n, 4, 4), camera_pose=(n, 4, 4), collision=(n,),
                  rgb=(n, h, w, 3), alpha=(n, h, w), coverage=(n,))
    if n != count + 1 or count < 1:
        raise ValueError("FRAME_ACTION_LENGTH")
    for name, shape in shapes.items():
        if name not in a or a[name].shape != shape or not np.isfinite(a[name]).all():
            raise ValueError("ARRAY_SCHEMA:" + name)
    if (a["rgb"].dtype != np.uint8 or a["collision"].dtype != bool
            or np.any((a["alpha"] < 0) | (a["alpha"] > 1)) or np.any(a["dt"] <= 0)
            or not np.array_equal(a["frame_id"], np.arange(n))):
        raise ValueError("ARRAY_RANGE_OR_DTYPE")
    times = np.r_[0., np.cumsum(a["dt"])]
    for key, target in (("timestamp", times), ("command_issue_time", times[:-1]),
                        ("effective_start", times[:-1]), ("effective_end", times[1:])):
        if not np.allclose(a[key], target, atol=1e-10, rtol=0):
            raise ValueError("ACTION_STATE_TIME_MISMATCH:" + key)
    limits = m["execution"]["limits"]
    if np.any(abs(a["states"][:, 3]) > limits["max_speed"]):
        raise ValueError("STATE_SPEED_LIMIT")
    if (np.any(abs(a["actions"][:, 0]) > limits["max_speed"])
            or np.any(abs(a["actions"][:, 1]) > limits["max_steering"])):
        raise ValueError("ACTION_LIMIT")
    for i in range(count):
        expected = integrate(a["states"][i], a["actions"][i], a["dt"][i], m["execution"]["wheelbase"])
        if not np.allclose(expected, a["states"][i + 1], atol=1e-10, rtol=0):
            raise ValueError("ACTION_STATE_MISMATCH")
        if not _swept(a["states"][i], a["actions"][i], a["dt"][i], m["planning_map"]):
            raise ValueError("SWEPT_COLLISION")
    if any(not collision_free(state, m["planning_map"]) for state in a["states"]) or np.any(a["collision"]):
        raise ValueError("COLLISION_OR_OUT_OF_BOUNDS")
    try:
        goal = np.asarray(m["goal"], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("INVALID_GOAL") from None
    if (goal.shape not in ((3,), (4,)) or not np.isfinite(goal).all()
            or (goal.shape == (4,) and goal[3] != 0)):
        raise ValueError("INVALID_GOAL")
    last = a["states"][-1]
    tolerances = m["execution"]["goal_tolerances"]
    yaw_error = np.arctan2(np.sin(last[2] - goal[2]), np.cos(last[2] - goal[2]))
    if (m["goal_reached"] is not True or np.linalg.norm(last[:2] - goal[:2]) > tolerances["position"]
            or abs(yaw_error) > tolerances["yaw"] or last[3] != 0):
        raise ValueError("GOAL_NOT_REACHED")
    vehicles = np.array([vehicle_pose(s) for s in a["states"]])
    if (not np.allclose(a["vehicle_pose"], vehicles, atol=1e-10, rtol=0)
            or not np.allclose(a["camera_pose"], vehicles @ m["renderer"]["T_vehicle_from_optical"], atol=1e-10, rtol=0)):
        raise ValueError("CAMERA_POSE_MISMATCH")
    coverage = np.mean(a["alpha"] > .01, axis=(1, 2))
    if np.any(coverage == 0) or not np.allclose(coverage, a["coverage"], atol=1e-10, rtol=0):
        raise ValueError("EMPTY_VIEW_OR_COVERAGE_MISMATCH")
    return dict(scene_id=m["scene_id"], trajectory_id=m["trajectory_id"], split=m["split"], frames=n,
                min_coverage=float(coverage.min()), max_coverage=float(coverage.max()))


def rollout_inputs(directory):
    """Explicit causal allowlist; targets and all privileged poses stay outside."""
    validate_episode(directory)
    with np.load(Path(directory) / "frames.npz", allow_pickle=False) as data:
        return dict(initial_rgb=data["rgb"][:1].copy(), initial_alpha=data["alpha"][:1].copy(),
                    issued_actions=data["actions"].copy(), dt_s=data["dt"].copy())


def validate_dataset(root):
    root = Path(root)
    index = read_json(root / "dataset.json")
    scenes, rows = {}, []
    identities = set()
    for relative in index["episodes"]:
        row = validate_episode(root / relative)
        key = (row["scene_id"], row["trajectory_id"])
        if key in identities:
            raise ValueError("DUPLICATE_EPISODE")
        identities.add(key)
        if row["scene_id"] in scenes and scenes[row["scene_id"]] != row["split"]:
            raise ValueError("SCENE_SPLIT_LEAKAGE")
        scenes[row["scene_id"]] = row["split"]
        rows.append(row)
    if not rows:
        raise ValueError("EMPTY_DATASET")
    if "scene_splits" in index and index["scene_splits"] != scenes:
        raise ValueError("SCENE_SPLIT_INDEX_MISMATCH")
    return dict(status="FIXTURE_ONLY", scenes=len(scenes), trajectories=len(rows),
                frames=sum(row["frames"] for row in rows), scene_splits=scenes, episodes=rows,
                failures=index["failures"], fixture_limitations=index.get("fixture_limitations", "unspecified"),
                training="NOT_RUN")
