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


def camera_reset(board_pos, board_yaw_, params, xp=np):
    """Initial (target, yaw) camera state: aimed straight at the board."""
    return camera_target(board_pos, board_yaw_, params.cam_lead_m, xp=xp), board_yaw_


def camera_update(target, yaw, board_pos, board_yaw_, params, dt, xp=np):
    """One substep of the chase camera's first-order lag -> (target, yaw).

    THE LAG IS PHYSICS, NOT PRESENTATION. During a trick the board rotates far
    faster than the camera follows, so the lag changes where a mid-gesture
    screen point projects and therefore changes the force the finger applies.
    `cam_follow_tau` is a fitted parameter for exactly that reason. An
    unlagged port silently simulates a different game.
    """
    tau = params.cam_follow_tau
    a = 1.0 if tau <= 1e-6 else 1.0 - np.exp(-dt / tau)   # both plain floats
    target = target + a * (camera_target(board_pos, board_yaw_,
                                         params.cam_lead_m, xp=xp) - target)
    # Shortest-arc yaw blend, so passing through +/-pi does not whip round.
    yaw = yaw + a * xp.arctan2(xp.sin(board_yaw_ - yaw), xp.cos(board_yaw_ - yaw))
    return target, yaw


def camera_ray_at(nx, ny, target, yaw, params, xp=np):
    """Screen point -> world ray, from an explicit (lagged) camera state."""
    right, up, forward = camera_basis(yaw, params.cam_pitch_deg, xp=xp)
    origin = target - params.cam_distance * forward
    tan_v = xp.tan(0.5 * xp.radians(params.cam_fov_deg))
    tan_h = tan_v * params.screen_aspect
    sx = (nx - 0.5) * 2.0 * tan_h
    sy = (0.5 - ny) * 2.0 * tan_v      # screen y runs downward
    d = forward + sx * right + sy * up
    return origin, d / xp.linalg.norm(d)


def camera_ray(nx, ny, board_pos, yaw, params, xp=np):
    """Screen ray for a camera aimed exactly at the board -- no lag.

    Only correct at the start of an episode, where the camera has just been
    reset. Everything inside a rollout must use `camera_ray_at` with the
    carried state.
    """
    target, _ = camera_reset(board_pos, yaw, params, xp=xp)
    return camera_ray_at(nx, ny, target, yaw, params, xp=xp)


def board_yaw(quat, xp=np):
    """Heading of the deck's long axis, from a wxyz quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return xp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def camera_pose(target, yaw, params, xp=np):
    """(position, 3x3 rotation) for MuJoCo's camera, from the chase state.

    MuJoCo cameras look down their own **-z** axis, with +x right and +y up, so
    the rotation's columns are (right, up, -forward). Getting that sign wrong
    renders the scene from behind the camera, which shows up as an empty frame
    rather than an error.

    This is what makes rendered frames the GAME's frames: the same fitted
    `FollowCamera` that decides where a screen touch lands also decides what
    the pixels show. A renderer with its own camera would produce images the
    touch model does not agree with.
    """
    right, up, forward = camera_basis(yaw, params.cam_pitch_deg, xp=xp)
    position = target - params.cam_distance * forward
    mat = xp.stack([right, up, -forward], axis=-1)
    return position, mat


def _camera_base_quat(pitch_deg: float) -> np.ndarray:
    """Constant part of the camera's orientation: everything but the yaw.

    The full camera rotation factorises as `Rz(yaw) @ M`, where M depends only
    on the (fixed) pitch. Computing M's quaternion once in numpy and composing
    a pure yaw rotation onto it at run time avoids ever converting a general
    rotation matrix back to a quaternion -- the trace form is singular exactly
    where a flipping board puts it, which has already cost this project one bug.
    """
    from ..sim.state import quat_from_mat_safe

    p = np.radians(pitch_deg)
    cp, sp = np.cos(p), np.sin(p)
    # Columns are (-right, up, -forward), NOT (right, up, -forward). MuJoCo
    # cameras follow the OpenGL convention: view along -z, +y up, and the frame
    # right-handed. Since right x up = +forward, the frame (right, up,
    # -forward) has determinant -1 -- a reflection, not a rotation, which
    # `quat_from_mat_safe` cannot represent and which silently produced a
    # camera pointing somewhere else entirely.
    M = np.array([[-0.0, -sp, -cp],
                  [-1.0, 0.0, 0.0],
                  [-0.0, cp, -sp]])
    assert abs(np.linalg.det(M) - 1.0) < 1e-9, "camera frame must be a rotation"
    return quat_from_mat_safe(M)


def camera_quat(yaw, pitch_deg: float, base_quat, xp=np):
    """wxyz orientation of the chase camera: a yaw applied to the fixed base."""
    half = 0.5 * yaw
    qz = xp.stack([xp.cos(half), xp.zeros_like(half), xp.zeros_like(half),
                   xp.sin(half)])
    w0, x0, y0, z0 = qz[0], qz[1], qz[2], qz[3]
    w1, x1, y1, z1 = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
    return xp.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
