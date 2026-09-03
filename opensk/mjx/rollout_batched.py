"""Batch-major rollout: one wide `Data`, stepped and rendered as a whole.

The env-major rollout in `rollout.py` is `vmap` over independent episodes, each
its own `scan`. That is the fastest shape for physics and it is what
`results/THROUGHPUT.md` measures. It cannot render: MJX's batch render context
hardcodes `nworld` because Warp allocates buffers JAX cannot see, so `render`
must not appear under a `vmap`.

So the pixel path inverts the nesting. The batch axis lives INSIDE `Data`, the
physics steps with `vmap(mjx.step)`, and rendering happens outside any `vmap`
on the whole batched `Data` at once.

The loop is two levels, which is not an implementation detail:

    scan over 68 FRAMES
      scan over the ~17 substeps between frames   <- physics
      render once                                 <- one call, whole batch

Rendering every substep would cost 17x for frames nobody trains on, and making
the render conditional inside a single flat scan would put a Warp FFI call
under `lax.cond`. Splitting the loop avoids both.

Everything else is shared with the env-major path -- the same `finger_force`,
the same gesture schedule, the same camera lag -- because those were written
against an `xp` namespace and are branchless. Only the nesting changes, and
`tests/test_mjx_rollout_batched.py` checks the two agree.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import gesture as g
from .rollout import EPISODE_SECONDS, episode_length
from .touch import FingerState, body_wrench, finger_force, initial_finger


class BatchRollout(NamedTuple):
    pos: np.ndarray     # (F, B, 3) sampled at frame times
    quat: np.ndarray    # (F, B, 4)
    rgb: np.ndarray | None    # (F, B, H, W, 3) when rendering, else None


def frames_and_substeps(params, seconds: float = EPISODE_SECONDS,
                        fps: float = 30.0) -> tuple[int, int]:
    """(frames, substeps per frame) for the two-level loop.

    Rounded UP, so the frame count matches the env-major path's 68 exactly.
    Frame k is sampled at substep `k * per_frame` in both paths, which is what
    makes their outputs comparable; the batch-major loop then runs 6 substeps
    (12 ms) past the last sampled frame, which nothing observes.
    """
    per_frame = max(1, int(round(1.0 / (fps * params.timestep))))
    n = episode_length(params, seconds)
    return -(-n // per_frame), per_frame


def initial_batch(mx, d0, batch: int):
    """`d0` broadcast to a batch axis, which is what `vmap(mjx.step)` consumes."""
    import jax

    return jax.tree.map(lambda x: np.broadcast_to(x, (batch,) + np.shape(x)).copy()
                        if np.ndim(x) >= 0 else x, d0)


def make_frame_fn(mx, model, params, deck_bid, deck_gids, n_slots: int,
                  substeps: int, render=None, cam_id: int = 0,
                  mocap_id: int = 0):
    """Build the per-FRAME function: `substeps` of physics, then one render."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    dt = params.timestep
    base_quat = jnp.asarray(g._camera_base_quat(params.cam_pitch_deg))

    def wrench(data, fingers, cam, points, seg_t, t0, t):
        """Forces for ONE environment. vmapped over the batch by the caller."""
        cam_target, cam_yaw = g.camera_update(
            cam[0], cam[1], data.xpos[deck_bid],
            g.board_yaw(data.qpos[3:7], xp=jnp), params, dt, xp=jnp)
        force = jnp.zeros(3)
        torque = jnp.zeros(3)
        new_fingers = []
        for s in range(n_slots):
            live = (t >= t0[s]) & (t <= t0[s] + seg_t[s][-1])
            nxny = g.path_position(points[s], seg_t[s], t - t0[s], xp=jnp)
            f, p, st = finger_force(fingers[s], nxny[0], nxny[1], model, data,
                                    deck_bid, deck_gids, params, dt,
                                    cam_target=cam_target, cam_yaw=cam_yaw,
                                    xp=jnp)
            fo, to = body_wrench(f, p, data.xipos[deck_bid], xp=jnp)
            gate = jnp.where(live, 1.0, 0.0)
            force = force + gate * fo
            torque = torque + gate * to
            new_fingers.append(FingerState(*[jnp.where(live, a, b)
                                             for a, b in zip(st, fingers[s])]))
        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[deck_bid, :3].set(force)
        xfrc = xfrc.at[deck_bid, 3:].set(torque)
        return xfrc, tuple(new_fingers), (cam_target, cam_yaw)

    batched_wrench = jax.vmap(wrench, in_axes=(0, 0, 0, 0, 0, 0, None))
    batched_step = jax.vmap(mjx.step, in_axes=(None, 0))

    def substep(carry, t):
        data, fingers, cam, points, seg_t, t0 = carry
        xfrc, fingers, cam = batched_wrench(data, fingers, cam, points,
                                            seg_t, t0, t)
        data = batched_step(mx, data.replace(xfrc_applied=xfrc))
        return (data, fingers, cam, points, seg_t, t0), None

    def frame(carry, t0_frame):
        # ONE substep, then sample, then the rest. That looks arbitrary and is
        # not: the env-major scan emits state AFTER `mjx.step`, so its frame k
        # is the state at substep 17k + 1, and the CPU generator yields after
        # `sim.step()` for the same reason. Sampling at 17k instead leaves the
        # two paths one substep apart -- which measured as 7.6e-3 m of
        # "divergence" at frame 5 in float64, an off-by-one wearing the costume
        # of a numerical difference.
        carry, _ = jax.lax.scan(substep, carry,
                                t0_frame + jnp.zeros((1,), dtype=float))
        data = carry[0]
        rgb = None
        if render is not None:
            # Point the FITTED chase camera, per world, before rendering.
            #
            # Through MOCAP, not cam_xpos/cam_xmat: those are outputs,
            # recomputed from the model whenever kinematics runs, so a camera
            # written that way silently reverts to its MJCF pose. Measured
            # before the fix: the board was in frame 0 of every episode and in
            # the last frame of 3%, including episodes that never moved 2 m.
            pos, quat = jax.vmap(lambda c: (
                g.camera_pose(c[0], c[1], params, xp=jnp)[0],
                g.camera_quat(c[1], params.cam_pitch_deg, base_quat, xp=jnp),
            ))(carry[2])
            data = data.replace(
                mocap_pos=data.mocap_pos.at[:, mocap_id].set(pos),
                mocap_quat=data.mocap_quat.at[:, mocap_id].set(quat))
            # Outside every vmap, by necessity: the render context owns Warp
            # buffers of a fixed nworld and cannot be traced through one.
            rgb, data = render(data)
            carry = (data,) + carry[1:]
        out = (data.qpos[:, :3], data.qpos[:, 3:7])
        ts = t0_frame + jnp.arange(1, substeps, dtype=float) * dt
        carry, _ = jax.lax.scan(substep, carry, ts)
        return carry, out if rgb is None else out + (rgb,)

    return frame


def rollout_batched(mx, model, params, deck_bid, deck_gids, init_data,
                    points, seg_t, t0, *, n_slots: int = 2, render=None,
                    cam_id: int = 0, mocap_id: int = 0,
                    seconds: float = EPISODE_SECONDS) -> BatchRollout:
    """Run a whole batch of episodes together. `init_data` carries the batch axis.

    `points`, `seg_t` and `t0` are the env-major gesture arrays with a leading
    batch axis, exactly as `rollout.gesture_arrays` produces them per episode.
    """
    import jax
    import jax.numpy as jnp

    n_frames, substeps = frames_and_substeps(params, seconds)
    batch = points.shape[0]
    frame = make_frame_fn(mx, model, params, deck_bid, deck_gids, n_slots,
                          substeps, render, cam_id, mocap_id)

    fingers = tuple(jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape),
                                 initial_finger(xp=jnp)) for _ in range(n_slots))
    cam = jax.vmap(lambda p, q: g.camera_reset(p, g.board_yaw(q, xp=jnp),
                                               params, xp=jnp))(
        init_data.xpos[:, deck_bid], init_data.qpos[:, 3:7])

    frame_starts = jnp.arange(n_frames, dtype=float) * substeps * params.timestep
    _, out = jax.lax.scan(frame, (init_data, fingers, cam, points, seg_t, t0),
                          frame_starts)
    return BatchRollout(pos=out[0], quat=out[1],
                        rgb=out[2] if len(out) > 2 else None)
