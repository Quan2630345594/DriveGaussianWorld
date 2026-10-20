"""NumPy-only midpoint bicycle reference for hypothetical test camera motion."""

import numpy as np


def _checked_inputs(initial_pose, commands, dt_seconds):
    pose = np.asarray(initial_pose, dtype=np.float64)
    controls = np.asarray(commands, dtype=np.float64)
    dt = np.asarray(dt_seconds, dtype=np.float64)
    if (pose.shape != (3,) or controls.ndim != 2 or controls.shape[1] != 2
            or not len(controls) or dt.shape != (len(controls),)
            or not all(np.isfinite(a).all() for a in (pose, controls, dt)) or np.any(dt <= 0)):
        raise ValueError("Need finite current pose[3], controls[H,2] and positive dt[H]")
    if (len(controls) > 48 or np.any(np.abs(controls) > [1.5, .45])
            or np.any(dt < .05) or np.any(dt > .2)):
        raise ValueError("Outside declared speed/steering-radian/dt domain or horizon budget")
    return pose.copy(), controls, dt


def midpoint_oracle_step(state, action_speed_steering_rad, dt_seconds, wheelbase_m):
    """Exact discrete midpoint equation extracted from the retired toy oracle."""
    action, dt = np.asarray(action_speed_steering_rad), np.asarray(dt_seconds)
    if action.shape != (2,) or dt.shape != () or not np.isscalar(wheelbase_m):
        raise ValueError("Need action[2], scalar dt and scalar wheelbase")
    if not np.isfinite(wheelbase_m) or wheelbase_m <= 0:
        raise ValueError("Wheelbase must be finite and positive")
    state, actions, durations = _checked_inputs(state, action[None], dt[None])
    v, steering = actions[0]
    yaw_delta = v / wheelbase_m * np.tan(steering) * durations[0]
    yaw_mid = state[2] + .5 * yaw_delta
    state[:2] += v * durations[0] * np.array([np.cos(yaw_mid), np.sin(yaw_mid)])
    state[2] += yaw_delta
    if not np.isfinite(state).all():
        raise ValueError("Oracle produced nonfinite state")
    return state


def oracle_rollout(initial_state, actions_speed_steering_rad, dt_seconds, wheelbase_m):
    state, actions, dt = _checked_inputs(initial_state, actions_speed_steering_rad, dt_seconds)
    predicted = np.empty((len(actions), 3))
    for i in range(len(actions)):
        state = midpoint_oracle_step(state, actions[i], dt[i], wheelbase_m)
        predicted[i] = state
    return predicted
