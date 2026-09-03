"""The touch model as a pure function, ready for `scan` over substeps.

A direct port of `sim/touch.py`, restructured so nothing mutates: finger state
is carried in and out, and the only output is the force/torque to write into
`xfrc_applied`. That is what lets the whole episode become one `jax.lax.scan`
and the whole batch one `vmap`.

The behaviour is the CPU model's, and each piece of it was expensive to learn
(see `sim/touch.py` for the measurements):

  * the finger holds the MATERIAL point it touched, and does not let go at a
    fixed distance -- a hard slip release left it disengaged for 76% of the
    median real gesture;
  * it sticks or slips by a COULOMB limit -- pure sliding gets flips but kills
    the ollie, pure sticking cannot roll the board at all;
  * a finger that starts on the ground can still CATCH the deck mid-gesture --
    without that, 51% of captured trick recipes did nothing in simulation.

Ray casts go through `geom.cast_deck_or_ground`, not `mj_ray`, which is
validated equivalent in `tests/test_mjx_geom.py`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import gesture as g
from .geom import cast_deck_or_ground

# Finger contact state, as small integers so the pytree stays numeric.
KIND_NONE, KIND_DECK, KIND_GROUND = 0, 1, 2


class FingerState(NamedTuple):
    """Carried across substeps. All arrays so this vmaps as a pytree."""
    kind: np.ndarray        # () int: NONE / DECK / GROUND
    local: np.ndarray       # (3,) grab point in the deck's frame
    depth: np.ndarray       # () view depth at touch-down
    prev_screen: np.ndarray  # (2,) previous screen position


def initial_finger(xp=np) -> FingerState:
    return FingerState(kind=xp.array(KIND_NONE),
                       local=xp.zeros(3),
                       depth=xp.array(0.0),
                       prev_screen=xp.zeros(2))


def deck_boxes_world(model, data, deck_gids, xp=np):
    """(pos, mat, half) per deck collision box, from live geom poses.

    `reshape(3, 3)` is a no-op under MJX and the fix under MuJoCo C, which
    stores rotation matrices flat -- so one implementation drives both and the
    parity test can compare them directly.
    """
    return [(data.geom_xpos[i], data.geom_xmat[i].reshape(3, 3), model.geom_size[i])
            for i in deck_gids]


def _point_at_depth(nx, ny, depth, board_pos, yaw, params, xp=np):
    """Where a screen point sits on the view-parallel plane at `depth`.

    This is what makes dragging feel like dragging: the fingertip stays at the
    depth of whatever it grabbed, so lateral screen motion becomes lateral
    world motion instead of motion toward the camera.
    """
    origin, d = g.camera_ray(nx, ny, board_pos, yaw, params, xp=xp)
    _, _, forward = g.camera_basis(yaw, params.cam_pitch_deg, xp=xp)
    denom = xp.dot(d, forward)
    safe = xp.where(xp.abs(denom) < 1e-9, 1e-9, denom)
    return origin + d * (depth / safe)


def _depth_of(point, board_pos, yaw, params, xp=np):
    origin, _ = g.camera_ray(0.5, 0.5, board_pos, yaw, params, xp=xp)
    _, _, forward = g.camera_basis(yaw, params.cam_pitch_deg, xp=xp)
    return xp.dot(point - origin, forward)


def finger_force(state: FingerState, nx, ny, model, data, deck_bid, deck_gids,
                 params, dt, xp=np):
    """One substep for one finger -> (force, point, torque_extra, new state).

    Branchless: every case is computed and selected with `where`, because a
    Python `if` on a traced value cannot be jitted. That costs a little work
    per step and buys the whole batch.
    """
    board_pos = data.xpos[deck_bid]
    # MJX stores xmat as (nbody, 3, 3); MuJoCo C stores it flat. reshape is a
    # no-op on the former and the fix for the latter, so the same code drives
    # both and the parity test can compare them directly.
    R = data.xmat[deck_bid].reshape(3, 3)
    yaw = g.board_yaw(_quat_from_mat(R, xp=xp), xp=xp)
    origin, ray_d = g.camera_ray(nx, ny, board_pos, yaw, params, xp=xp)
    boxes = deck_boxes_world(model, data, deck_gids, xp=xp)
    hit_deck, t_hit = cast_deck_or_ground(origin, ray_d, boxes, xp=xp)
    hit_point = origin + ray_d * t_hit

    # --- acquisition: at touch-down, and again for a ground finger that has
    # since crossed the deck (the re-acquisition that rescued 51% of recipes).
    acquiring = (state.kind == KIND_NONE) | \
                ((state.kind == KIND_GROUND) & hit_deck)
    new_local = R.T @ (hit_point - board_pos)
    new_depth = _depth_of(hit_point, board_pos, yaw, params, xp=xp)
    kind_after_acq = xp.where(hit_deck, KIND_DECK, KIND_GROUND)

    kind = xp.where(acquiring, kind_after_acq, state.kind)
    local = xp.where(acquiring, new_local, state.local)
    depth = xp.where(acquiring, new_depth, state.depth)

    # --- deck contact: stick or slip by a Coulomb limit ---------------------
    contact = board_pos + R @ local
    tip = _point_at_depth(nx, ny, _depth_of(contact, board_pos, yaw, params, xp=xp),
                          board_pos, yaw, params, xp=xp)
    pull = tip - contact
    normal = R[:, 2]
    f_n = xp.dot(pull, normal)
    tangent = pull - f_n * normal
    slipping = xp.linalg.norm(tangent) > params.touch_friction * xp.abs(f_n)

    rel = R.T @ (tip - board_pos)
    half_l, half_w = 0.5 * params.deck_length, 0.5 * params.deck_width
    slid = xp.stack([xp.clip(rel[0], -half_l, half_l),
                     xp.clip(rel[1], -half_w, half_w),
                     xp.array(0.5 * params.deck_thickness)])
    local = xp.where(slipping & (kind == KIND_DECK), slid, local)

    contact = board_pos + R @ local
    err = tip - contact
    r = contact - data.xipos[deck_bid]
    # Free-joint velocities, not cvel: the CPU model uses qvel[0:3] linear and
    # qvel[3:6] angular in the world frame, and cvel is a different quantity
    # (COM-based spatial velocity). Matching the reference exactly matters more
    # than elegance here.
    vel = data.qvel[0:3] + xp.cross(data.qvel[3:6], r)
    force = params.touch_gain * err - params.touch_damping * vel
    mag = xp.linalg.norm(force)
    force = force * xp.minimum(1.0, params.touch_force_max / xp.maximum(mag, 1e-9))

    # --- ground contact: a push, as an impulse proportional to screen travel -
    step_screen = xp.stack([nx, ny]) - state.prev_screen
    travel = xp.linalg.norm(step_screen)
    sign = xp.where(step_screen[1] >= 0.0, 1.0, -1.0)
    thrust = sign * params.ground_shove_gain * travel / dt
    shove = xp.stack([thrust * xp.cos(yaw), thrust * xp.sin(yaw),
                      xp.zeros_like(thrust)])

    on_deck = (kind == KIND_DECK)
    out_force = xp.where(on_deck, force, shove)
    out_point = xp.where(on_deck, contact, data.xipos[deck_bid])

    return out_force, out_point, FingerState(
        kind=kind, local=local, depth=depth, prev_screen=xp.stack([nx, ny]))


def _quat_from_mat(R, xp=np):
    """wxyz quaternion from a rotation matrix, branchlessly.

    Uses the trace form guarded away from its singularity rather than the
    usual four-case branch, which cannot be traced.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    w = xp.sqrt(xp.maximum(0.0, 1.0 + tr)) / 2.0
    w = xp.maximum(w, 1e-8)
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return xp.stack([w, x, y, z])


def body_wrench(force, point, com, xp=np):
    """(force, torque about the body COM) for writing into xfrc_applied."""
    return force, xp.cross(point - com, force)
