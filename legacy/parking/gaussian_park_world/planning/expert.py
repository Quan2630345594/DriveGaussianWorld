"""Small CPU Hybrid A* expert, not a planner for reconstructed Gaussian scenes.

States are rear-axle [x_m, y_m, yaw_rad, signed_speed_m_s]; controls are
[signed_speed_m_s, front_steering_rad]. Rectangles use [xmin, ymin, xmax, ymax].
The zero-delay simulator snaps speed to the command, including from rest; it
does not model acceleration, steering rate, actuation delay, or real dynamics.
"""

from enum import Enum
import heapq
import math

import numpy as np


SOURCE = "SYNTHETIC_PLANNING_FIXTURE"
WHEELBASE = 2.7
MAX_SPEED = 1.0
MAX_STEERING = .55
GOAL_XY = .35
GOAL_YAW = .12


class FailureReason(str, Enum):
    NO_MAP = "NO_MAP"
    UNTRUSTED_GEOMETRY = "UNTRUSTED_GEOMETRY"
    INVALID_INPUT = "INVALID_INPUT"
    START_BLOCKED = "START_BLOCKED"
    GOAL_BLOCKED = "GOAL_BLOCKED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    TRACKING_FAILED = "TRACKING_FAILED"


def fixture_map(scene_id, variant=0):
    """Return fresh, fully known static fixture geometry, never Gaussian alpha.

    Variant 0 has two parked rectangles off the central aisle; variant 1 adds
    an aisle obstacle. Unknown rectangles, if supplied, are also occupied.
    This descriptor can serve as the scene in a synthetic-only gate test.
    """
    if (not isinstance(scene_id, str) or not scene_id or type(variant) is not int
            or variant < 0):
        raise ValueError("Need a nonempty scene_id and nonnegative fixture variant")
    obstacles = [[-5., 4., 0., 6.], [5., -6., 10., -4.]]
    if variant % 2:
        obstacles.append([3., -.8, 4., .8])
    return dict(scene_id=scene_id, source_type=SOURCE,
                metric_scale_trusted=True,
                metric_scale_evidence="declared synthetic SI fixture coordinates",
                gauge_id=f"synthetic:{scene_id}:{variant}",
                bounds=[-10., -8., 15., 8.],
                ground=dict(trusted=True, evidence="known planar synthetic ground"),
                obstacles=obstacles, unknown_rectangles=[], unknown_occupied=True)


def _rectangles(value):
    array = np.asarray(value, dtype=float)
    if array.shape == (0,):
        return array.reshape(0, 4)
    if (array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all()
            or np.any(array[:, 2:] <= array[:, :2])):
        raise ValueError("Expected finite positive-area rectangles")
    return array


def require_planning_geometry(scene, map):
    """Raise ValueError unless both descriptors attest the same metric gauge.

    Maps need source_type, scene_id, gauge_id, metric_scale_trusted=True and
    metric_scale_evidence. Scenes may use this descriptor or the scene manifest
    contract's verified world_gauge with synthetic-by-construction evidence.
    Only the explicit synthetic source is supported. Gaussian opacity/alpha is
    never inspected or used as geometry.
    Returns the supplied map unchanged; it does not certify arbitrary real data.
    """
    if isinstance(scene, dict) and "world_gauge" in scene:
        gauge = scene["world_gauge"]
        if (not isinstance(gauge, dict) or gauge.get("metric_scale") != "verified"
                or gauge.get("units") != "meters"
                or gauge.get("axes") != "x_forward_y_left_z_up"
                or gauge.get("confidence") != "synthetic_by_construction"
                or scene.get("source_classification") != SOURCE
                or scene.get("planning_status") != "SYNTHETIC_FIXTURE_ONLY"):
            raise ValueError("Scene manifest lacks supported metric fixture evidence")
        scene = dict(source_type=SOURCE, scene_id=scene.get("scene_id"),
                     gauge_id=gauge.get("id"), metric_scale_trusted=True,
                     metric_scale_evidence=gauge["confidence"])
    for descriptor in (scene, map):
        if not isinstance(descriptor, dict):
            raise ValueError("Missing scene or planning map")
        if (descriptor.get("source_type") != SOURCE
                or descriptor.get("metric_scale_trusted") is not True
                or not all(isinstance(descriptor.get(k), str) and descriptor[k].strip()
                           for k in ("scene_id", "gauge_id", "metric_scale_evidence"))):
            raise ValueError("Need explicit synthetic metric evidence and gauge")
    if any(scene[k] != map[k] for k in ("scene_id", "gauge_id")):
        raise ValueError("Scene and map must share scene_id and metric gauge")
    ground = map.get("ground")
    if (not isinstance(ground, dict) or ground.get("trusted") is not True
            or not isinstance(ground.get("evidence"), str) or not ground["evidence"].strip()
            or map.get("unknown_occupied") is not True):
        raise ValueError("Need evidenced ground and occupied unknown space")
    bounds = np.asarray(map.get("bounds"), dtype=float)
    if bounds.shape != (4,):
        raise ValueError("Need finite metric bounds")
    _rectangles([bounds])
    _rectangles(map.get("obstacles"))
    _rectangles(map.get("unknown_rectangles", []))
    return map


def _angle(yaw):
    return math.atan2(math.sin(yaw), math.cos(yaw))


def integrate(state, action, dt, wheelbase=WHEELBASE):
    """Exact constant-control bicycle arc; successor.v is commanded speed.

    Yaw is left unwrapped. Positive steering turns left when driving forward
    and right in reverse. This general integrator does not clip controls.
    """
    state, action = np.asarray(state, dtype=float), np.asarray(action, dtype=float)
    if (state.shape != (4,) or action.shape != (2,) or not np.isfinite(state).all()
            or not np.isfinite(action).all() or not np.isfinite(dt) or dt <= 0
            or not np.isfinite(wheelbase) or wheelbase <= 0 or abs(action[1]) >= math.pi / 2):
        raise ValueError("Invalid bicycle state, control, duration or wheelbase")
    distance = action[0] * dt
    turn = distance * math.tan(action[1]) / wheelbase
    # sinc form avoids cancellation for nearly straight arcs.
    travel = distance * np.sinc(turn / (2 * math.pi))
    heading = state[2] + turn / 2
    return np.array([state[0] + travel * math.cos(heading),
                     state[1] + travel * math.sin(heading),
                     state[2] + turn, action[0]], dtype=float)


def _footprint_free(state, map, extra=0.):
    c, s = math.cos(state[2]), math.sin(state[2])
    axes = np.array([[c, s], [-s, c]])
    center = state[:2] + 1.2 * axes[0]
    half = np.array([4.4 / 2 + .2 + extra, 1.8 / 2 + .2 + extra])
    corners = center + np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * half @ axes
    bounds = np.asarray(map["bounds"])
    if np.any(corners <= bounds[:2]) or np.any(corners >= bounds[2:]):
        return False
    for rect in list(map["obstacles"]) + list(map.get("unknown_rectangles", [])):
        rect = np.asarray(rect)
        other_center = (rect[:2] + rect[2:]) / 2
        other_half = (rect[2:] - rect[:2]) / 2
        delta = other_center - center
        # Separating axes for an oriented vehicle rectangle and world AABB.
        if np.any(np.abs(delta) > half @ np.abs(axes) + other_half):
            continue
        if np.any(np.abs(axes @ delta) > half + np.abs(axes) @ other_half):
            continue
        return False
    return True


def collision_free(state, map):
    """Fail closed on missing/untrusted maps; check the entire safety footprint."""
    try:
        require_planning_geometry(map, map)
        state = np.asarray(state, dtype=float)
        return bool(state.shape == (4,) and np.isfinite(state).all()
                    and _footprint_free(state, map))
    except (ValueError, TypeError, KeyError):
        return False


def _swept(state, action, dt, map):
    # Bound travel of every footprint point to <= .08 m between checks.
    distance = abs(action[0] * dt)
    reach = math.hypot(3.4 + .2, .9 + .2)
    motion = distance * (1 + reach * abs(math.tan(action[1])) / WHEELBASE)
    steps = max(1, math.ceil(motion / .08))
    # Inflate by the inter-sample displacement bound: thin walls cannot tunnel.
    for i in range(steps + 1):
        pose = state if i == 0 else integrate(state, action, dt * i / steps)
        if not _footprint_free(pose, map, motion / steps / 2):
            return False
    return True


def _at_goal(state, goal):
    return (np.linalg.norm(state[:2] - goal[:2]) <= GOAL_XY
            and abs(_angle(state[2] - goal[2])) <= GOAL_YAW)


def plan_and_track(map, start, goal, *, max_expansions=3000, max_retries=1):
    """Return status/reason, float64 states[T+1,4], actions[T,2], dt[T].

    status is SUCCESS or FAILURE; reason is None or a FailureReason string.
    Goal may be [x,y,yaw] or [x,y,yaw,0]; only stopped goals are supported.
    max_expansions is a TOTAL budget across initial search and retries (0..2).
    Failures return only the original state and no controls, never a partial
    episode. Call require_planning_geometry(scene, map) first for scene binding.
    """
    metadata = dict(simulator="SYNTHETIC_ZERO_DELAY", speed_snap=True,
                    state_fields=["x", "y", "yaw", "signed_speed"],
                    action_fields=["signed_speed", "steering"],
                    wheelbase=WHEELBASE, length=4.4, width=1.8,
                    rear_axle_to_center=1.2, safety_margin=.2,
                    limits=dict(max_speed=MAX_SPEED, max_steering=MAX_STEERING),
                    goal_tolerances=dict(position=GOAL_XY, yaw=GOAL_YAW, speed=0.),
                    expansions=0, attempts=0, max_expansions=max_expansions,
                    max_retries=max_retries)
    initial = np.empty((0, 4), dtype=float)

    def result(reason=None, states=None, actions=None, durations=None):
        return dict(status="SUCCESS" if reason is None else "FAILURE",
                    reason=None if reason is None else reason.value,
                    states=initial.copy() if states is None else np.asarray(states, dtype=float),
                    actions=np.empty((0, 2)) if actions is None else np.asarray(actions, dtype=float),
                    dt=np.empty(0) if durations is None else np.asarray(durations, dtype=float),
                    metadata=metadata)

    try:
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if (start.shape != (4,) or goal.shape not in ((3,), (4,))
                or not np.isfinite(start).all() or not np.isfinite(goal).all()
                or abs(start[3]) > MAX_SPEED or (len(goal) == 4 and goal[3] != 0)
                or type(max_expansions) is not int or not 0 <= max_expansions <= 20000
                or type(max_retries) is not int or not 0 <= max_retries <= 2):
            return result(FailureReason.INVALID_INPUT)
        initial = start[None].copy()
    except (TypeError, ValueError):
        return result(FailureReason.INVALID_INPUT)
    if map is None:
        return result(FailureReason.NO_MAP)
    try:
        require_planning_geometry(map, map)
    except (ValueError, TypeError, KeyError):
        return result(FailureReason.UNTRUSTED_GEOMETRY)
    if not _footprint_free(start, map):
        return result(FailureReason.START_BLOCKED)
    if not _footprint_free(goal, map):
        return result(FailureReason.GOAL_BLOCKED)

    path = None
    for attempt in range(max_retries + 1):
        metadata["attempts"] += 1
        resolution = (.4, .25, .15)[attempt]
        yaw_resolution = (math.pi / 18, math.pi / 36, math.pi / 48)[attempt]

        def key(state):
            bins = round(2 * math.pi / yaw_resolution)
            return (round(state[0] / resolution), round(state[1] / resolution),
                    round(_angle(state[2]) / yaw_resolution) % bins, int(np.sign(state[3])))

        def heuristic(state):
            return np.linalg.norm(state[:2] - goal[:2]) + .5 * abs(_angle(state[2] - goal[2]))

        # Node records remain immutable so a replaced grid key cannot corrupt parents.
        nodes = [(start.copy(), None, None, 0.)]
        queue = [(heuristic(start), 0)]
        costs = {key(start): 0.}
        # Reserve some of the total budget for finer-resolution retries.
        allowance = (max_expansions - metadata["expansions"]) // (max_retries + 1 - attempt)
        used = 0
        while queue:
            _, index = heapq.heappop(queue)
            state, parent, control, cost = nodes[index]
            if cost > costs.get(key(state), math.inf) + 1e-9:
                continue
            if _at_goal(state, goal):
                path = []
                while parent is not None:
                    path.append((state, control))
                    index = parent
                    state, parent, control, cost = nodes[index]
                path.reverse()
                break
            if used >= allowance:
                break
            used += 1
            metadata["expansions"] += 1
            for speed in (MAX_SPEED, -MAX_SPEED):
                for steering in (0., -.55, .55, -.275, .275):
                    action = np.array([speed, steering])
                    successor = integrate(state, action, .6)
                    new_cost = cost + .6 * (1. if speed > 0 else 1.15) + .03 * abs(steering)
                    if state[3] * speed < 0:
                        new_cost += .3
                    successor_key = key(successor)
                    if new_cost >= costs.get(successor_key, math.inf):
                        continue
                    if not _swept(state, action, .6, map):
                        continue
                    costs[successor_key] = new_cost
                    nodes.append((successor, index, action, new_cost))
                    heapq.heappush(queue, (new_cost + 1.4 * heuristic(successor), len(nodes) - 1))
        if path is not None:
            break
    if path is None:
        return result(FailureReason.SEARCH_EXHAUSTED)

    states, actions, durations = [start.copy()], [], []
    reference_start = start.copy()
    for reference_end, primitive in path:
        for substep in range(1, 7):
            reference = (reference_start if substep == 1 else
                         integrate(reference_start, primitive, .1 * (substep - 1)))
            actual = states[-1]
            error = actual[:2] - reference[:2]
            cross_track = -math.sin(reference[2]) * error[0] + math.cos(reference[2]) * error[1]
            heading_error = _angle(actual[2] - reference[2])
            steering = np.clip(primitive[1] - np.sign(primitive[0]) * .8 * heading_error
                               - .4 * cross_track, -MAX_STEERING, MAX_STEERING)
            action = np.array([primitive[0], steering])
            if not _swept(actual, action, .1, map):
                return result(FailureReason.TRACKING_FAILED)
            actions.append(action)
            durations.append(.1)
            states.append(integrate(actual, action, .1))
        reference_start = reference_end
    if not _at_goal(states[-1], goal):
        return result(FailureReason.TRACKING_FAILED)
    actions.append(np.array([0., 0.]))
    durations.append(.1)
    states.append(integrate(states[-1], actions[-1], durations[-1]))
    return result(states=states, actions=actions, durations=durations)
