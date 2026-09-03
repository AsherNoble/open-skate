"""One gesture episode as a single `lax.scan`, batched with `vmap`.

This is the piece that turns the CPU simulator into a training environment:
the whole episode — 949 substeps of gesture plus settle — becomes one compiled
scan, and a batch of thousands of environments becomes one vmap over it.

Shapes follow the device's gesture schema, which is also the capture format, so
an action here is executable on a phone with no translation:

    points   (S, 3, 2)   S slots, 3 normalised screen waypoints each
    seg_t    (S, 3)      cumulative segment boundary times, millisecond-quantised
    t0       (S,)        slot start times from the delay chain

A slot's finger is live for `t0 <= t <= t0 + seg_t[-1]`. Outside that window it
contributes no force, which is expressed as a multiply rather than a branch so
the whole thing stays traceable.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import gesture as g
from .touch import FingerState, body_wrench, finger_force, initial_finger


class Rollout(NamedTuple):
    """Board trajectory over an episode, one row per substep."""
    pos: np.ndarray    # (T, 3)
    quat: np.ndarray   # (T, 4)


def gesture_arrays(recipe: dict, n_slots: int = 2, xp=np):
    """Recipe dict -> (points, seg_t, t0), padded to exactly `n_slots`.

    Padding repeats the last slot with a zero-length window rather than
    inserting zeros, so a padded slot is simply never live and needs no special
    case in the scan.
    """
    pts, segs, starts = [], [], []
    t = 0.0
    for i in range(n_slots):
        gi = recipe["gestures"][min(i, len(recipe["gestures"]) - 1)]
        p = np.asarray(gi["points"], dtype=float)
        P, seg_t, dur = g.schedule(p, float(gi["duration"]),
                                   float(gi["easing_power"]), xp=np)
        real = i < len(recipe["gestures"])
        pts.append(P)
        segs.append(seg_t)
        starts.append(t if real else 1e6)   # padded slots never become live
        if real and i < len(recipe["gestures"]) - 1:
            delays = recipe.get("delays") or [0.0]
            t += float(dur) + float(delays[min(i, len(delays) - 1)])
    return (xp.asarray(np.stack(pts)), xp.asarray(np.stack(segs)),
            xp.asarray(np.asarray(starts)))


# The capture window is pre_s 0.5 + window_s 1.8 = 2.3 s, which is what gives
# the real samples their 68-69 frames at 30 fps. Episodes use the same span so a
# simulated rollout and an expert demonstration are the same shaped object.
EPISODE_SECONDS = 2.3


def episode_length(params, seconds: float = EPISODE_SECONDS) -> int:
    """Substeps in one episode: 1150 at the defaults, 69 frames at 30 fps."""
    return int(round(seconds / params.timestep))


def frame_indices(n_steps: int, params, fps: float = 30.0) -> np.ndarray:
    """Substep indices to sample as frames, matching the capture's rate.

    The real captures are 68-69 frames over 2.3 s; keeping the same cadence
    means a simulated rollout and an expert demonstration are the same object.
    """
    stride = max(1, int(round(1.0 / (fps * params.timestep))))
    return np.arange(0, n_steps, stride)


def make_step_fn(mx, model, params, deck_bid, deck_gids, n_slots: int = 2):
    """Build the per-substep function for `lax.scan`.

    Returns `step(carry, t) -> (carry, board_pose)` where carry is
    `(mjx_data, FingerState-per-slot)`. Closed over the model so the scan body
    takes only traced values.
    """
    import jax.numpy as jnp
    from mujoco import mjx

    dt = params.timestep

    def step(carry, t):
        data, fingers, cam, points, seg_t, t0 = carry
        # The camera is updated BEFORE the fingers act, matching the reference
        # loop's order. It lags the board by `cam_follow_tau`, which changes
        # where a mid-gesture screen point lands and so changes the force.
        cam_target, cam_yaw = g.camera_update(
            cam[0], cam[1], data.xpos[deck_bid],
            g.board_yaw(data.qpos[3:7], xp=jnp), params, dt, xp=jnp)
        total_force = jnp.zeros(3)
        total_torque = jnp.zeros(3)
        new_fingers = []
        for s in range(n_slots):
            live = (t >= t0[s]) & (t <= t0[s] + seg_t[s][-1])
            nxny = g.path_position(points[s], seg_t[s], t - t0[s], xp=jnp)
            f, p, st = finger_force(fingers[s], nxny[0], nxny[1], model, data,
                                    deck_bid, deck_gids, params, dt,
                                    cam_target=cam_target, cam_yaw=cam_yaw,
                                    xp=jnp)
            force, torque = body_wrench(f, p, data.xipos[deck_bid], xp=jnp)
            gate = jnp.where(live, 1.0, 0.0)
            total_force = total_force + gate * force
            total_torque = total_torque + gate * torque
            # A slot that is not live keeps its state frozen, so a finger does
            # not "acquire" the board before its gesture has begun.
            new_fingers.append(
                FingerState(*[jnp.where(live, a, b)
                              for a, b in zip(st, fingers[s])]))

        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[deck_bid, :3].set(total_force)
        xfrc = xfrc.at[deck_bid, 3:].set(total_torque)
        data = mjx.step(mx, data.replace(xfrc_applied=xfrc))
        return (data, tuple(new_fingers), (cam_target, cam_yaw),
                points, seg_t, t0), (data.qpos[:3], data.qpos[3:7])

    return step


def rollout(mx, model, params, deck_bid, deck_gids, init_data,
            points, seg_t, t0, n_steps: int, n_slots: int = 2) -> Rollout:
    """Run one episode. `vmap` this over environments to batch it."""
    import jax
    import jax.numpy as jnp

    step = make_step_fn(mx, model, params, deck_bid, deck_gids, n_slots)
    fingers = tuple(initial_finger(xp=jnp) for _ in range(n_slots))
    cam = g.camera_reset(init_data.xpos[deck_bid],
                         g.board_yaw(init_data.qpos[3:7], xp=jnp), params, xp=jnp)
    ts = jnp.arange(n_steps, dtype=float) * params.timestep
    carry, (pos, quat) = jax.lax.scan(
        step, (init_data, fingers, cam, points, seg_t, t0), ts)
    return Rollout(pos=pos, quat=quat)
