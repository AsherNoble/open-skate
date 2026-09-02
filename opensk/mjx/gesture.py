"""Gesture sampling and the chase camera, in a form that survives jit and vmap.

Both are straight ports of `sim/gesture_spec.py` and `sim/camera.py`, written
against an `xp` namespace so the same code runs under numpy (for the parity
tests) and jax.numpy (batched on GPU).

THE MILLISECOND QUANTISATION IS NOT AN IMPLEMENTATION DETAIL. The device
schedules a gesture as whole-millisecond per-segment durations, and easing_power
redistributes time BETWEEN segments rather than shaping velocity within one.
Get that wrong and the environment's action space stops matching the phone's,
which is the one property that makes anything learned here transferable. See
`sim/gesture_spec.py` and `tests/test_gesture_parity.py`, which checks it
against the rig's own code.
"""
from __future__ import annotations

import numpy as np


def segment_durations_ms(n_segments: int, total_ms, easing_power, xp=np):
    """Per-segment durations in whole milliseconds.

    Mirrors `easing_to_segment_durations` in the rig's touch_actions.py, plus
    curved_drag's even split when easing_power == 1.0. Uses floor rather than
    Python's int() so it stays traceable; both truncate toward zero here
    because the values are positive.
    """
    i = xp.arange(n_segments + 1, dtype=float)
    b = (i / n_segments) ** easing_power
    raw = b[1:] - b[:-1]
    eased = xp.maximum(1.0, xp.floor(raw / xp.sum(raw) * total_ms))
    even = xp.maximum(1.0, xp.floor(total_ms / n_segments))
    return xp.where(xp.abs(easing_power - 1.0) < 1e-9, even, eased)


def path_position(points, seg_t, t, xp=np):
    """Screen position along a gesture path at time `t` seconds.

    `points` is (n, 2); `seg_t` the cumulative segment boundary times from
    `segment_durations_ms`. Each segment is traversed at constant speed --
    easing redistributes time between segments, it does not shape velocity
    within one.

    Branchless: clamps before the first waypoint and after the last, and picks
    the active segment by summing masked contributions rather than indexing,
    so it vectorises.
    """
    total = seg_t[-1]
    tc = xp.clip(t, 0.0, total)
    n_seg = points.shape[0] - 1
    out = xp.zeros(2)
    for i in range(n_seg):
        t0, t1 = seg_t[i], seg_t[i + 1]
        span = xp.maximum(t1 - t0, 1e-9)
        u = xp.clip((tc - t0) / span, 0.0, 1.0)
        seg = points[i] + u * (points[i + 1] - points[i])
        active = ((tc >= t0) & (tc < t1)) | ((i == n_seg - 1) & (tc >= t1))
        out = out + xp.where(active, 1.0, 0.0) * seg
    return out


def schedule(points, duration, easing_power, xp=np):
    """(points, seg_t, executed_duration) for one gesture.

    `executed_duration` is the millisecond-truncated total, which is what the
    device actually runs -- generally NOT the requested duration.
    """
    n_seg = points.shape[0] - 1
    ms = segment_durations_ms(n_seg, xp.floor(duration * 1000.0), easing_power, xp=xp)
    seg_t = xp.concatenate([xp.zeros(1), xp.cumsum(ms) / 1000.0])
    return points, seg_t, seg_t[-1]


# --- camera ---------------------------------------------------------------

def camera_basis(yaw, pitch_deg, xp=np):
    """(right, up, forward) unit vectors, world frame.

    up = forward x right. The reverse flips the screen vertically while leaving
    projection round-trips self-consistent, so it survives the obvious test --
    it cost a debugging session on the CPU side.
    """
    pitch = xp.radians(pitch_deg)
    cy, sy = xp.cos(yaw), xp.sin(yaw)
    cp, sp = xp.cos(pitch), xp.sin(pitch)
    forward = xp.stack([cy * cp, sy * cp, sp])
    right = xp.stack([-sy, cy, xp.zeros_like(sy)])
    up = xp.cross(forward, right)
    return right, up, forward / xp.linalg.norm(forward)


def camera_target(board_pos, yaw, lead_m, xp=np):
    """The point the camera aims at: ahead of the board, not at it."""
    return board_pos + xp.stack([lead_m * xp.cos(yaw), lead_m * xp.sin(yaw),
                                 xp.zeros_like(yaw)])


def camera_ray(nx, ny, board_pos, yaw, params, xp=np):
    """Normalised screen point -> (origin, unit direction) world ray."""
    right, up, forward = camera_basis(yaw, params.cam_pitch_deg, xp=xp)
    target = camera_target(board_pos, yaw, params.cam_lead_m, xp=xp)
    origin = target - params.cam_distance * forward
    tan_v = xp.tan(0.5 * xp.radians(params.cam_fov_deg))
    tan_h = tan_v * params.screen_aspect
    sx = (nx - 0.5) * 2.0 * tan_h
    sy = (0.5 - ny) * 2.0 * tan_v      # screen y runs downward
    d = forward + sx * right + sy * up
    return origin, d / xp.linalg.norm(d)


def board_yaw(quat, xp=np):
    """Heading of the deck's long axis, from a wxyz quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return xp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
