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
    cam_pos: np.ndarray | None  # (F, B, 3) camera actually used for the render
    roll_deg: np.ndarray
    yaw_deg: np.ndarray
    peak_height: np.ndarray
    air_s: np.ndarray
    displacement: np.ndarray


class _OutcomeState(NamedTuple):
    previous_quat: np.ndarray
    first_pos: np.ndarray
    roll: np.ndarray
    yaw: np.ndarray
    peak: np.ndarray
    air: np.ndarray
    displacement: np.ndarray
    seen: np.ndarray


def _quat_mul(a, b, xp):
    w0, x0, y0, z0 = (a[..., i] for i in range(4))
    w1, x1, y1, z1 = (b[..., i] for i in range(4))
    return xp.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], axis=-1)


def _quat_to_mat(q, xp):
    w, x, y, z = (q[..., i] for i in range(4))
    return xp.stack([
        xp.stack([1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)], -1),
        xp.stack([2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)], -1),
        xp.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)], -1),
    ], axis=-2)


def _update_outcome(state, data, active, rest_z, dt, xp):
    """Accumulate the same summary as the env-major physics-rate rollout."""
    pos, quat = data.qpos[:, :3], data.qpos[:, 3:7]
    conj = xp.stack([state.previous_quat[..., 0],
                     -state.previous_quat[..., 1],
                     -state.previous_quat[..., 2],
                     -state.previous_quat[..., 3]], axis=-1)
    delta = _quat_mul(quat, conj, xp)
    norm = xp.linalg.norm(delta[..., 1:], axis=-1)
    angle = 2.0 * xp.arctan2(norm, delta[..., 0])
    axis = delta[..., 1:] / xp.maximum(norm, 1e-12)[..., None]
    local = xp.einsum("...ji,...j->...i", _quat_to_mat(quat, xp), axis)
    pair = active & state.seen
    live = xp.where(pair & (norm > 1e-12), 1.0, 0.0)

    first = active & ~state.seen
    first_pos = xp.where(first[..., None], pos, state.first_pos)
    height = pos[:, 2] - rest_z
    peak = xp.where(active, xp.maximum(state.peak, height), state.peak)
    air = state.air + xp.where(active & (height > 0.01), dt, 0.0)
    travel = xp.linalg.norm(pos[:, :2] - first_pos[:, :2], axis=-1)
    displacement = xp.where(
        active, xp.maximum(state.displacement, travel), state.displacement)
    return _OutcomeState(
        previous_quat=xp.where(active, quat, state.previous_quat),
        first_pos=first_pos,
        roll=state.roll + live * angle * local[..., 0],
        yaw=state.yaw + live * angle * local[..., 2],
        peak=peak,
        air=air,
        displacement=displacement,
        seen=state.seen | active,
    )


def frames_and_substeps(params, seconds: float = EPISODE_SECONDS,
                        fps: float = 30.0) -> tuple[int, int]:
    """(frames, substeps per frame) for the two-level loop.

    Rounded UP, so the frame count matches the env-major path's 68 exactly.
    Frame k is sampled at substep `k * per_frame` in both paths, which is what
    makes their outputs comparable; the batch-major loop then runs 6 substeps
    (12 ms) past the last sampled frame. Frames do not observe it and outcome
    accumulation explicitly gates it off at the requested physics step count.
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
                  substeps: int, n_steps: int, rest_z, render=None,
                  cam_id: int = 0, mocap_id: int = 0):
    """Build the per-FRAME function: `substeps` of physics, then one render."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    dt = params.timestep
    base_quat = jnp.asarray(g._camera_base_quat(params.cam_pitch_deg))

    def wrench(data, fingers, cam, points, seg_t, t0, spin, t):
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
        spin_live = ((spin[0] > 0.5) & (t >= spin[1]) & (t <= spin[2]))
        deck_normal = data.xmat[deck_bid].reshape(3, 3)[:, 2]
        torque = torque + jnp.where(
            spin_live, params.spin_torque, 0.0) * deck_normal
        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[deck_bid, :3].set(force)
        xfrc = xfrc.at[deck_bid, 3:].set(torque)
        return xfrc, tuple(new_fingers), (cam_target, cam_yaw)

    batched_wrench = jax.vmap(wrench, in_axes=(0, 0, 0, 0, 0, 0, 0, None))
    batched_step = jax.vmap(mjx.step, in_axes=(None, 0))

    def camera_mocap(data, cam):
        """Write the chase camera's pose into mocap, per world.

        This must happen BEFORE `mjx.step`, not after. `cam_xpos`/`cam_xmat`
        are computed by kinematics DURING the step, so a mocap pose written
        afterwards is not seen until the next one -- and a render placed
        between the two uses the previous step's camera. That is why writing
        the camera after stepping changed the rendered frames not at all,
        to the last decimal place.
        """
        pos, quat = jax.vmap(lambda c: (
            g.camera_pose(c[0], c[1], params, xp=jnp)[0],
            g.camera_quat(c[1], params.cam_pitch_deg, base_quat, xp=jnp),
        ))(cam)
        return data.replace(
            mocap_pos=data.mocap_pos.at[:, mocap_id].set(pos),
            mocap_quat=data.mocap_quat.at[:, mocap_id].set(quat))

    def substep(carry, t):
        data, fingers, cam, points, seg_t, t0, spin, outcome = carry
        xfrc, fingers, cam = batched_wrench(data, fingers, cam, points,
                                            seg_t, t0, spin, t)
        data = camera_mocap(data.replace(xfrc_applied=xfrc), cam)
        data = batched_step(mx, data)
        step_index = jnp.rint(t / dt).astype(jnp.int32)
        outcome = _update_outcome(
            outcome, data, step_index < n_steps, rest_z, dt, jnp)
        return (data, fingers, cam, points, seg_t, t0, spin, outcome), None

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
            # The camera pose is already correct here: `substep` writes mocap
            # before every `mjx.step`, so the kinematics that produced this
            # state also placed the camera. Nothing to set.
            #
            # Outside every vmap, by necessity: the render context owns Warp
            # buffers of a fixed nworld and cannot be traced through one.
            rgb, data = render(data)
            carry = (data,) + carry[1:]
        out = (data.qpos[:, :3], data.qpos[:, 3:7])
        ts = t0_frame + jnp.arange(1, substeps, dtype=float) * dt
        carry, _ = jax.lax.scan(substep, carry, ts)
        # The camera pose the renderer actually saw, read back rather than
        # assumed. Whether a mocap write reaches the render is exactly the
        # thing that cannot be settled by reading the plumbing.
        return carry, (out if rgb is None
                       else out + (rgb, data.cam_xpos[:, cam_id]))

    return frame


def rollout_batched(mx, model, params, deck_bid, deck_gids, init_data,
                    points, seg_t, t0, *, n_slots: int = 2, render=None,
                    cam_id: int = 0, mocap_id: int = 0,
                    seconds: float = EPISODE_SECONDS, spin=None,
                    rest_z=None) -> BatchRollout:
    """Run a whole batch of episodes together. `init_data` carries the batch axis.

    `points`, `seg_t` and `t0` are the env-major gesture arrays with a leading
    batch axis, exactly as `rollout.gesture_arrays` produces them per episode.
    """
    import jax
    import jax.numpy as jnp

    n_frames, substeps = frames_and_substeps(params, seconds)
    n_steps = episode_length(params, seconds)
    batch = points.shape[0]
    if spin is None:
        spin = jnp.zeros((batch, 3))
    if rest_z is None:
        rest_z = init_data.qpos[:, 2]
    else:
        rest_z = jnp.broadcast_to(jnp.asarray(rest_z), (batch,))
    frame = make_frame_fn(mx, model, params, deck_bid, deck_gids, n_slots,
                          substeps, n_steps, rest_z, render, cam_id, mocap_id)

    fingers = tuple(jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape),
                                 initial_finger(xp=jnp)) for _ in range(n_slots))
    cam = jax.vmap(lambda p, q: g.camera_reset(p, g.board_yaw(q, xp=jnp),
                                               params, xp=jnp))(
        init_data.xpos[:, deck_bid], init_data.qpos[:, 3:7])
    outcome = _OutcomeState(
        previous_quat=init_data.qpos[:, 3:7],
        first_pos=init_data.qpos[:, :3],
        roll=jnp.zeros(batch), yaw=jnp.zeros(batch),
        peak=jnp.full((batch,), -jnp.inf), air=jnp.zeros(batch),
        displacement=jnp.zeros(batch), seen=jnp.zeros(batch, dtype=bool))

    frame_starts = jnp.arange(n_frames, dtype=float) * substeps * params.timestep
    carry, out = jax.lax.scan(
        frame, (init_data, fingers, cam, points, seg_t, t0, spin, outcome),
        frame_starts)
    outcome = carry[-1]
    return BatchRollout(pos=out[0], quat=out[1],
                        rgb=out[2] if len(out) > 2 else None,
                        cam_pos=out[3] if len(out) > 3 else None,
                        roll_deg=jnp.degrees(outcome.roll),
                        yaw_deg=jnp.degrees(outcome.yaw),
                        peak_height=outcome.peak, air_s=outcome.air,
                        displacement=outcome.displacement)
