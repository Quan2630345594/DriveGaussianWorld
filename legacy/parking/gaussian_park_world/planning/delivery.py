"""Auditable CPU expert delivery. Analytic fixtures are not Gaussian occupancy GT."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..dataset.offline import load_scene, read_json, vehicle_pose
from ..scenes.contract import sha256, write_json
from .expert import (SOURCE, WHEELBASE, MAX_SPEED, MAX_STEERING, GOAL_XY,
                     GOAL_YAW, fixture_map, require_planning_geometry,
                     plan_and_track, integrate, collision_free, _swept, _at_goal)


TIMING = dict(state="rear_axle_x_m_y_m_yaw_rad_signed_speed_mps",
              action="issued_signed_speed_mps_front_steering_rad",
              interval="[effective_start,effective_end)",
              latency="SYNTHETIC_ZERO_DELAY", time_origin="trajectory_start_seconds")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode("ascii")).hexdigest()


def _geometry(map):
    return {k: map[k] for k in ("bounds", "obstacles", "unknown_rectangles",
                               "unknown_occupied", "ground")}


def _map(scene, layout):
    map = fixture_map(scene["scene_id"], 0)
    map["gauge_id"] = scene["world_gauge"]["id"]
    if layout == 1:
        map["obstacles"].append([3., -.8, 4., .8])
    elif layout == 2:
        map["obstacles"] = [[-6., 4.5, -1., 6.5], [6., -6.5, 11., -4.5],
                            [9., 3., 11., 5.]]
    elif layout != 0:
        raise ValueError("UNKNOWN_LAYOUT")
    return map


def _requests(batch):
    requests = []
    starts = [(x, y) for x in (-5., -2.) for y in (-2., 0., 2.)] if batch else [(-2., 0.)]
    for i, (x, y) in enumerate(starts):
        start = np.array([x, y, 0., 0.])
        for name, action, duration in (("forward", [1., 0.], 3.6),
                                       ("reverse", [-1., 0.], 2.4),
                                       ("left", [1., .275], 2.4),
                                       ("right", [1., -.275], 2.4)):
            # Reference goals only. The planner/tracker must issue every control.
            goal = integrate(start, action, duration)
            goal[3] = 0.
            requests.append(dict(trajectory_id=f"start{i}_{name}",
                                 start=start.tolist(), goal=goal.tolist()))
    requests.append(dict(trajectory_id="deliberate_start_blocked",
                         start=[14., 0., 0., 0.], goal=[0., 0., 0., 0.]))
    return requests


def _rig(scene):
    expected = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                         [0., -1., 0., 1.], [0., 0., 0., 1.]])
    if (scene["source_classification"] == SOURCE
            and scene["provenance"].get("reconstruction_version") == "fixture_v1"
            and np.array_equal(scene["cameras"][0]["T_world_from_optical"], expected)):
        return expected
    return None


def _statistics(states, actions):
    forward = int(np.count_nonzero(actions[:, 0] > 0))
    reverse = int(np.count_nonzero(actions[:, 0] < 0))
    curved = int(np.count_nonzero(np.abs(actions[:, 1]) > 1e-8))
    return dict(frame_count=len(states), action_count=len(actions),
                forward_count=forward, reverse_count=reverse, curved_count=curved,
                classification=("mixed" if forward and reverse else
                                "reverse" if reverse else "forward" if forward else "stationary"),
                max_abs_curvature=float(np.max(np.abs(np.tan(actions[:, 1]) / WHEELBASE))))


def _check_arrays(data, entry, scene):
    states, actions, dt = (np.asarray(data[k]) for k in ("states", "actions", "dt"))
    n = len(states)
    shapes = dict(states=(n, 4), actions=(n - 1, 2), dt=(n - 1,), timestamp=(n,),
                  frame_id=(n,), command_issue_time=(n - 1,), effective_start=(n - 1,),
                  effective_end=(n - 1,), vehicle_pose=(n, 4, 4))
    rig = _rig(scene)
    if rig is not None:
        shapes["camera_pose"] = (n, 4, 4)
    elif "camera_pose" in data:
        raise ValueError("UNVERIFIED_CAMERA")
    if n < 2:
        raise ValueError("EMPTY_TRAJECTORY")
    for key, shape in shapes.items():
        a = np.asarray(data[key])
        if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"INVALID_ARRAY:{key}")
    if np.any(dt <= 0):
        raise ValueError("INVALID_DT")
    times = np.r_[0., np.cumsum(dt)]
    for key, expected in (("timestamp", times), ("command_issue_time", times[:-1]),
                          ("effective_start", times[:-1]), ("effective_end", times[1:]),
                          ("frame_id", np.arange(n))):
        if not np.allclose(data[key], expected, atol=1e-10, rtol=0):
            raise ValueError(f"INVALID_TIMING:{key}")
    if np.asarray(data["frame_id"]).dtype.kind not in "iu":
        raise ValueError("INVALID_FRAME_IDS")
    if (np.any(np.abs(actions[:, 0]) > MAX_SPEED + 1e-10)
            or np.any(np.abs(states[:, 3]) > MAX_SPEED + 1e-10)
            or np.any(np.abs(actions[:, 1]) > MAX_STEERING + 1e-10)
            or np.any(np.abs(np.tan(actions[:, 1]) / WHEELBASE)
                      > np.tan(MAX_STEERING) / WHEELBASE + 1e-10)):
        raise ValueError("VEHICLE_LIMITS")
    metadata = entry["metadata"]
    for key, expected in dict(wheelbase=WHEELBASE, length=4.4, width=1.8,
                              rear_axle_to_center=1.2, safety_margin=.2,
                              simulator="SYNTHETIC_ZERO_DELAY", speed_snap=True,
                              limits=dict(max_speed=MAX_SPEED, max_steering=MAX_STEERING),
                              goal_tolerances=dict(position=GOAL_XY, yaw=GOAL_YAW, speed=0.)).items():
        if metadata.get(key) != expected:
            raise ValueError(f"INVALID_VEHICLE_METADATA:{key}")
    start, goal = np.asarray(entry["start"]), np.asarray(entry["goal"])
    if (start.shape != (4,) or goal.shape != (4,) or not np.isfinite(start).all()
            or not np.isfinite(goal).all() or goal[3] != 0
            or not np.allclose(states[0], start, atol=1e-10, rtol=0)
            or not _at_goal(states[-1], goal) or states[-1, 3] != 0
            or not np.array_equal(actions[-1], [0., 0.])):
        raise ValueError("INVALID_START_GOAL_STOP")
    map = entry["planning_map"]
    if not collision_free(start, map) or not collision_free(goal, map):
        raise ValueError("BLOCKED_ENDPOINT")
    for i, (action, duration) in enumerate(zip(actions, dt)):
        if not np.allclose(integrate(states[i], action, duration), states[i + 1],
                           atol=1e-9, rtol=0):
            raise ValueError("ACTION_REPLAY_MISMATCH")
        if not _swept(states[i], action, duration, map):
            raise ValueError("SWEPT_COLLISION")
    if not all(collision_free(state, map) for state in states):
        raise ValueError("STATIC_COLLISION")
    poses = np.array([vehicle_pose(state) for state in states])
    if not np.allclose(data["vehicle_pose"], poses, atol=1e-10, rtol=0):
        raise ValueError("VEHICLE_POSE_MISMATCH")
    if rig is not None and not np.allclose(data["camera_pose"], poses @ rig, atol=1e-10, rtol=0):
        raise ValueError("CAMERA_POSE_MISMATCH")
    if entry["camera_calibration"] != ("fixture_v1_identity_vehicle_camera_1m" if rig is not None else "unavailable"):
        raise ValueError("CAMERA_CALIBRATION_MISMATCH")
    if entry["statistics"] != _statistics(states, actions):
        raise ValueError("STATISTICS_MISMATCH")


def generate(scene_root, output, batch=False, budget=6000, retries=2, requests=None):
    """Exclusively create a delivery; optional request dictionaries keep CPU tests small.

    Batch cycles three analytic layouts across all eligible sources, repeating
    sources if needed to cover all three layouts. These are not real-scene GT.
    A request has trajectory_id, start[4], goal[4]; goals are never issued controls.
    """
    if type(budget) is not int or not 0 <= budget <= 20000 or type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("Invalid planner budget/retries")
    started = time.monotonic()
    root, output = Path(scene_root).resolve(), Path(output).resolve()
    directories = [root] if (root / "manifest.json").is_file() else sorted(
        p for p in root.iterdir() if (p / "manifest.json").is_file())
    if not directories:
        raise ValueError("NO_SOURCE_SCENES")
    scenes = [(p, load_scene(p)[0]) for p in directories]
    if output == root or root in output.parents or any(output in p.parents for p in directories):
        raise ValueError("SOURCE_OUTPUT_OVERLAP")
    output.mkdir(exist_ok=False)
    source_classes = sorted({scene["source_classification"] for _, scene in scenes})
    index = dict(schema_version=1,
                 source_classification=source_classes[0] if len(source_classes) == 1 else "MIXED",
                 source_classifications=source_classes, trajectory_source_classifications=[], timing=TIMING,
                 source_scenes=[], maps=[], requests=[], trajectories=[], failures=[],
                 batch=bool(batch), budget=budget, retries=retries,
                 limitations="Analytic synthetic planning maps are separate from visual Gaussians; no occupancy GT, real-scene GT, rendering, GWM or benchmark claim.")
    for directory, scene in scenes:
        index["source_scenes"].append(dict(scene_id=scene["scene_id"], scene_dir=str(directory),
             source_classification=scene["source_classification"],
             scene_manifest_sha256=sha256(directory / "manifest.json"),
             scene_array_sha256=scene["array_sha256"], world_gauge=scene["world_gauge"]))
    eligible = []
    for directory, scene in scenes:
        try:
            require_planning_geometry(scene, _map(scene, 0))
        except (ValueError, KeyError, TypeError):
            index["failures"].append(dict(scene_id=scene["scene_id"], reason="BLOCKED_GEOMETRY",
                                           attempts=0, elapsed_seconds=0.))
        else:
            eligible.append((directory, scene))
    assignments = [(i % 3, *eligible[i % len(eligible)])
                   for i in range(max(3, len(eligible)))] if batch and eligible else [
        (0, directory, scene) for directory, scene in eligible]
    for layout, directory, scene in assignments:
        map = _map(scene, layout)
        require_planning_geometry(scene, map)
        map_id = f"map{len(index['maps'])}"
        descriptor = dict(map_id=map_id, name=("fixture_aisle", "center_obstacle", "off_aisle_geometry")[layout],
                          layout=layout, scene_id=scene["scene_id"], planning_map=map,
                          geometry_sha256=_digest(_geometry(map)))
        write_json(output / f"{map_id}.json", descriptor)
        descriptor = dict(descriptor, file=f"{map_id}.json", sha256=sha256(output / f"{map_id}.json"))
        index["maps"].append(descriptor)
        for request in (_requests(batch) if requests is None else requests):
            request = dict(request)
            name = request["trajectory_id"]
            if not isinstance(name, str) or not name or not all(c.isascii() and (c.isalnum() or c in "_-") for c in name):
                raise ValueError("INVALID_REQUEST_ID")
            request["start"] = np.asarray(request["start"], dtype=float).tolist()
            request["goal"] = np.asarray(request["goal"], dtype=float).tolist()
            request_id = f"{map_id}_{request['trajectory_id']}"
            if any(r["request_id"] == request_id for r in index["requests"]):
                raise ValueError("DUPLICATE_REQUEST")
            record = dict(request, request_id=request_id, map_id=map_id, scene_id=scene["scene_id"])
            index["requests"].append(record)
            tick = time.monotonic()
            result = plan_and_track(map, request["start"], request["goal"],
                                    max_expansions=budget, max_retries=retries)
            common = dict(record, trajectory_id=request_id, attempts=result["metadata"]["attempts"],
                          elapsed_seconds=time.monotonic() - tick, metadata=result["metadata"])
            if result["status"] != "SUCCESS":
                index["failures"].append(dict(common, reason=result["reason"]))
                continue
            states, actions, dt = (result[k] for k in ("states", "actions", "dt"))
            times = np.r_[0., np.cumsum(dt)]
            arrays = dict(states=states, actions=actions, dt=dt, timestamp=times,
                          frame_id=np.arange(len(states)), command_issue_time=times[:-1],
                          effective_start=times[:-1], effective_end=times[1:],
                          vehicle_pose=np.array([vehicle_pose(s) for s in states]))
            rig = _rig(scene)
            if rig is not None:
                arrays["camera_pose"] = arrays["vehicle_pose"] @ rig
            entry = dict(common, scene_dir=str(directory), planning_map=map,
                         source_classification=scene["source_classification"],
                         scene_manifest_sha256=sha256(directory / "manifest.json"),
                         geometry_sha256=descriptor["geometry_sha256"],
                         camera_calibration="fixture_v1_identity_vehicle_camera_1m" if rig is not None else "unavailable",
                         statistics=_statistics(states, actions), file=f"{request_id}.npz")
            _check_arrays(arrays, entry, scene)
            with (output / entry["file"]).open("xb") as stream:
                np.savez(stream, **arrays)
            entry["sha256"] = sha256(output / entry["file"])
            index["trajectories"].append(entry)
    index["trajectory_source_classifications"] = sorted({
        entry["source_classification"] for entry in index["trajectories"]})
    index["counts"] = _counts(index)
    index["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "experts.json", index)
    return validate_experts(output)


def _counts(index):
    return dict(trajectories=len(index["trajectories"]), failures=len(index["failures"]),
                requests=len(index["requests"]), maps=len(index["maps"]),
                frames=sum(e["statistics"]["frame_count"] for e in index["trajectories"]),
                actions=sum(e["statistics"]["action_count"] for e in index["trajectories"]),
                geometry_layouts=len({m["geometry_sha256"] for m in index["maps"]}),
                failure_reasons=dict(Counter(e["reason"] for e in index["failures"])))


def validate_experts(directory):
    """Recheck source bytes/gauges, map snapshots, issued-control replay and counts.

    Raises ValueError (never a success report) for invalid or stale deliveries.
    Hashes detect stale files, not adversarial rewriting of the entire index.
    """
    directory = Path(directory).resolve()
    try:
        index = read_json(directory / "experts.json")
        if index["schema_version"] != 1 or index["timing"] != TIMING:
            raise ValueError("INVALID_DELIVERY_SCHEMA")
        scenes = {}
        for source in index["source_scenes"]:
            sid = source["scene_id"]
            path = Path(source["scene_dir"])
            if sid in scenes or not path.is_absolute() or path == directory or directory in path.parents:
                raise ValueError("DUPLICATE_OR_OUTPUT_SOURCE")
            if sha256(path / "manifest.json") != source["scene_manifest_sha256"]:
                raise ValueError("STALE_SCENE_MANIFEST_HASH")
            scene, _ = load_scene(path)
            if source["source_classification"] != scene["source_classification"]:
                raise ValueError("SOURCE_CLASSIFICATION_MISMATCH")
            if (scene["scene_id"] != sid or scene["world_gauge"] != source["world_gauge"]
                    or scene["array_sha256"] != source["scene_array_sha256"]):
                raise ValueError("SOURCE_SCENE_GAUGE_MISMATCH")
            scenes[sid] = (source, scene)
        source_classes = sorted({scene["source_classification"] for _, scene in scenes.values()})
        if (not source_classes or index["source_classifications"] != source_classes
                or index["source_classification"] != (source_classes[0] if len(source_classes) == 1 else "MIXED")):
            raise ValueError("SOURCE_CLASSIFICATION_MISMATCH")
        files, maps = {"experts.json"}, {}

        def local_file(name):
            if not isinstance(name, str) or Path(name).name != name or name in files:
                raise ValueError("DUPLICATE_OR_CROSS_OUTPUT_REFERENCE")
            path = directory / name
            if path.is_symlink() or path.resolve().parent != directory:
                raise ValueError("CROSS_OUTPUT_REFERENCE")
            files.add(name)
            return path

        for descriptor in index["maps"]:
            mid = descriptor["map_id"]
            if mid in maps:
                raise ValueError("DUPLICATE_MAP")
            path = local_file(descriptor["file"])
            if sha256(path) != descriptor["sha256"]:
                raise ValueError("MAP_HASH_MISMATCH")
            if read_json(path) != {k: v for k, v in descriptor.items() if k not in ("file", "sha256")}:
                raise ValueError("MAP_SNAPSHOT_MISMATCH")
            scene = scenes[descriptor["scene_id"]][1]
            map = descriptor["planning_map"]
            require_planning_geometry(scene, map)
            if map != _map(scene, descriptor["layout"]) or descriptor["geometry_sha256"] != _digest(_geometry(map)):
                raise ValueError("MAP_GEOMETRY_MISMATCH")
            maps[mid] = descriptor
        if index["batch"] and maps and len({m["geometry_sha256"] for m in maps.values()}) != 3:
            raise ValueError("BATCH_LAYOUTS_NOT_UNIQUE")
        requests = {}
        for request in index["requests"]:
            rid = request["request_id"]
            if rid in requests or request["map_id"] not in maps or request["scene_id"] != maps[request["map_id"]]["scene_id"]:
                raise ValueError("INVALID_OR_DUPLICATE_REQUEST")
            requests[rid] = request
        seen, gated = set(), set()
        for entry in index["trajectories"] + index["failures"]:
            if "request_id" not in entry:
                if (entry["reason"] != "BLOCKED_GEOMETRY" or entry["scene_id"] not in scenes
                        or entry["scene_id"] in gated or entry["attempts"] != 0):
                    raise ValueError("INVALID_GEOMETRY_GATE_FAILURE")
                gated.add(entry["scene_id"])
                scene = scenes[entry["scene_id"]][1]
                try:
                    require_planning_geometry(scene, _map(scene, 0))
                except (ValueError, KeyError, TypeError):
                    continue
                raise ValueError("FALSE_GEOMETRY_GATE_FAILURE")
            rid = entry["request_id"]
            if (rid in seen or rid not in requests or entry["trajectory_id"] != rid
                    or any(entry[k] != v for k, v in requests[rid].items() if k != "trajectory_id")):
                raise ValueError("REQUEST_OUTCOME_MISMATCH")
            seen.add(rid)
            if (not np.isfinite(entry["elapsed_seconds"]) or entry["elapsed_seconds"] < 0
                    or entry["attempts"] != entry["metadata"]["attempts"]
                    or type(entry["attempts"]) is not int
                    or not 0 <= entry["attempts"] <= index["retries"] + 1
                    or entry["metadata"]["max_expansions"] != index["budget"]
                    or entry["metadata"]["max_retries"] != index["retries"]
                    or type(entry["metadata"]["expansions"]) is not int
                    or not 0 <= entry["metadata"]["expansions"] <= index["budget"]):
                raise ValueError("INVALID_ATTEMPTS")
            if "reason" in entry:
                map = maps[entry["map_id"]]["planning_map"]
                reason = entry["reason"]
                if reason not in {"INVALID_INPUT", "START_BLOCKED", "GOAL_BLOCKED", "SEARCH_EXHAUSTED", "TRACKING_FAILED"}:
                    raise ValueError("INVALID_FAILURE_REASON")
                if reason in {"START_BLOCKED", "GOAL_BLOCKED"}:
                    endpoint = entry["start"] if reason == "START_BLOCKED" else entry["goal"]
                    if collision_free(endpoint, map) or entry["attempts"] != 0:
                        raise ValueError("FALSE_BLOCKED_ENDPOINT")
        if seen != set(requests):
            raise ValueError("MISSING_REQUEST_OUTCOME")
        if set(scenes) != gated | {m["scene_id"] for m in maps.values()}:
            raise ValueError("UNACCOUNTED_SOURCE_SCENE")
        for entry in index["trajectories"]:
            source, scene = scenes[entry["scene_id"]]
            if entry["source_classification"] != scene["source_classification"]:
                raise ValueError("TRAJECTORY_SOURCE_CLASSIFICATION_MISMATCH")
            descriptor = maps[entry["map_id"]]
            if (entry["scene_dir"] != source["scene_dir"]
                    or entry["scene_manifest_sha256"] != source["scene_manifest_sha256"]
                    or entry["planning_map"] != descriptor["planning_map"]
                    or entry["geometry_sha256"] != descriptor["geometry_sha256"]):
                raise ValueError("TRAJECTORY_SOURCE_MAP_MISMATCH")
            require_planning_geometry(scene, entry["planning_map"])
            path = local_file(entry["file"])
            if sha256(path) != entry["sha256"]:
                raise ValueError("TRAJECTORY_HASH_MISMATCH")
            with np.load(path, allow_pickle=False) as data:
                _check_arrays(data, entry, scene)
        trajectory_classes = sorted({entry["source_classification"] for entry in index["trajectories"]})
        if index["trajectory_source_classifications"] != trajectory_classes:
            raise ValueError("TRAJECTORY_SOURCE_CLASSIFICATION_MISMATCH")
        if {p.name for p in directory.iterdir()} != files:
            raise ValueError("UNINDEXED_OUTPUT_FILES")
        counts = _counts(index)
        if counts != index["counts"]:
            raise ValueError("INDEX_COUNT_MISMATCH")
        return dict(valid=True, **counts, elapsed_seconds=index["elapsed_seconds"],
                    source_classification=index["source_classification"],
                    source_classifications=source_classes,
                    trajectory_source_classifications=trajectory_classes)
    except (KeyError, TypeError, OSError, IndexError) as error:
        raise ValueError(f"INVALID_EXPERT_DELIVERY:{error}") from error
