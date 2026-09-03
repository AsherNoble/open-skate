"""The batched gesture environment: thousands of episodes in one compiled call.

One action is one whole gesture; one episode is the 2.3 s window a capture
covers. `step` returns the ROLLOUT -- 68 frames of board pose -- not a single
final observation, because a world model consumes sequences and because a
simulated rollout and a real expert demonstration then have the same shape.

Deliberately not a Gym `Env`. Gym's interface is one environment stepping one
timestep, and everything here is thousands of environments running a whole
episode at once; wrapping that in a per-step API would force the batch back
through Python and give away the entire reason for the port. A single-env
adapter can sit on top if some tool needs one.

Measured on an A10G (`results/THROUGHPUT.md`): the batch is what buys the
speed -- at B=1 the GPU is slower than the phone rig, at B=1024 it is ~64x
faster.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..mjx import gesture as g
from ..mjx.rollout import EPISODE_SECONDS, episode_length, frame_indices, rollout
from ..sim.params import SkateParams
from .action import action_dim, decode


class Episodes(NamedTuple):
    """A batch of rollouts. Leading axis is the environment, always."""
    pos: np.ndarray        # (B, F, 3) board position at each sampled frame
    quat: np.ndarray       # (B, F, 4) wxyz orientation
    roll_deg: np.ndarray   # (B,) accumulated turn about the deck's long axis
    yaw_deg: np.ndarray    # (B,) accumulated turn about the deck's normal
    peak_height: np.ndarray    # (B,) metres above the resting ride height
    air_s: np.ndarray          # (B,) time clear of the ground
    displacement: np.ndarray   # (B,) furthest travelled, in plan
    valid: np.ndarray          # (B,) bool: the episode stayed physical


# An episode past these bounds did not happen: a finger on a 0.9 kg deck cannot
# put the board 3 m up or 40 m away inside 2.3 s.
#
# It has to be checked because the model ALLOWS it. The deck force saturates at
# `touch_force_max` -- a finger pulls only as hard as a finger can -- but the
# ground shove has no such cap: `thrust = ground_shove_gain * travel / dt`, so
# a fast full-screen drag asks for kilonewtons and MuJoCo's solver goes
# unstable. This is the REFERENCE model's behaviour, not a port artefact: the
# same actions blow up on CPU, where MuJoCo prints "Nan, Inf or huge value in
# QACC". Capping the shove the way the deck force is capped is the obvious fix
# and is a change to the fitted physics, so it belongs with the fidelity work
# rather than here. Until then the environment reports which episodes are real
# instead of quietly handing a trainer a NaN.
MAX_PLAUSIBLE_HEIGHT_M = 3.0
MAX_PLAUSIBLE_TRAVEL_M = 40.0


def _quat_mul(a, b, xp):
    w0, x0, y0, z0 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w1, x1, y1, z1 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return xp.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], axis=-1)


def _quat_to_mat(q, xp):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


# Clearance above the resting height that counts as airborne. Same value as
# `mjx/outcomes.py`, and it has to stay the same or the two disagree by
# definition rather than by physics.
AIR_CLEARANCE_M = 0.01


def summarise(pos, quat, timestep, rest_z, xp):
    """Trajectory -> what the board did. Vectorised over time, no Python loop.

    Rotation is accumulated per step and projected onto the deck's own axes, so
    a full turn reads as 360 degrees rather than zero and a board that rolls
    and yaws at once is decomposed correctly.
    """
    q1, q0 = quat[1:], quat[:-1]
    conj = xp.stack([q0[..., 0], -q0[..., 1], -q0[..., 2], -q0[..., 3]], axis=-1)
    d = _quat_mul(q1, conj, xp)
    n = xp.linalg.norm(d[..., 1:], axis=-1)
    safe = xp.maximum(n, 1e-12)
    ang = 2.0 * xp.arctan2(n, d[..., 0])
    axis = d[..., 1:] / safe[..., None]
    # into the deck's own frame at each instant
    local = xp.einsum("tji,tj->ti", _quat_to_mat(q1, xp), axis)
    live = xp.where(n > 1e-12, 1.0, 0.0)
    roll = xp.sum(live * ang * local[..., 0])
    yaw = xp.sum(live * ang * local[..., 2])

    height = pos[:, 2] - rest_z
    return (xp.degrees(roll), xp.degrees(yaw), xp.max(height),
            xp.sum(xp.where(height > AIR_CLEARANCE_M, 1.0, 0.0)) * timestep,
            xp.max(xp.linalg.norm(pos[:, :2] - pos[0, :2], axis=-1)))


class GestureEnv:
    """Batched, episodic, gesture-level. Build once, `step` many times.

    Construction compiles the rollout for a given batch size, which costs
    10-25 s and is paid once. `step` after that is a single accelerator call.
    """

    def __init__(self, params: SkateParams | None = None, *, n_slots: int = 2,
                 seconds: float = EPISODE_SECONDS, settle_steps: int = 200):
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        from ..mjx.parity import make_mjx

        self.params = params or SkateParams()
        self.n_slots = n_slots
        self.action_dim = action_dim(n_slots)
        self.n_steps = episode_length(self.params, seconds)
        self.frames = frame_indices(self.n_steps, self.params)

        mx, d0, cpu = make_mjx(self.params)
        step = jax.jit(lambda dd: mjx.step(mx, dd))
        for _ in range(settle_steps):     # the board settles before every episode
            d0 = step(d0)
        self._mx, self._d0, self._cpu = mx, d0, cpu
        self.rest_z = float(np.asarray(d0.qpos)[2])
        self._one = None

    # -- the batched call --------------------------------------------------

    def _build(self):
        import jax
        import jax.numpy as jnp

        p, mx, d0, cpu = self.params, self._mx, self._d0, self._cpu
        n_slots, n_steps = self.n_slots, self.n_steps
        frames = jnp.asarray(self.frames)
        deck_gids = sorted(cpu._deck_gids)

        def one(vec):
            points, durations, easings, delays = decode(vec, n_slots, xp=jnp)
            segs, starts, t = [], [], 0.0
            for i in range(n_slots):
                _, seg_t, dur = g.schedule(points[i], durations[i], easings[i], xp=jnp)
                segs.append(seg_t)
                starts.append(t)
                if i < n_slots - 1:
                    t = t + dur + delays[i]
            seg_t = jnp.stack(segs)
            t0 = jnp.stack([jnp.asarray(s, dtype=float) for s in starts])

            r = rollout(mx, cpu.model, p, cpu.deck_bid, deck_gids, d0,
                        points, seg_t, t0, n_steps, n_slots)
            roll, yaw, peak, air, disp = summarise(
                r.pos, r.quat, p.timestep, self.rest_z, jnp)
            valid = (jnp.isfinite(peak) & jnp.isfinite(disp) & jnp.isfinite(roll)
                     & (peak < MAX_PLAUSIBLE_HEIGHT_M)
                     & (disp < MAX_PLAUSIBLE_TRAVEL_M))
            return (r.pos[frames], r.quat[frames], roll, yaw, peak, air, disp, valid)

        return jax.jit(jax.vmap(one))

    def step(self, actions) -> Episodes:
        """(B, action_dim) unbounded reals -> a batch of rollouts.

        There is no `reset`: every episode starts from the same settled board,
        so a batch of actions is fully described by the actions themselves.
        That is what makes the whole batch one call.
        """
        import jax.numpy as jnp

        if self._one is None:
            self._one = self._build()
        a = jnp.asarray(np.asarray(actions, dtype=float))
        if a.ndim == 1:
            a = a[None, :]
        pos, quat, roll, yaw, peak, air, disp, valid = self._one(a)
        return Episodes(pos=pos, quat=quat, roll_deg=roll, yaw_deg=yaw,
                        peak_height=peak, air_s=air, displacement=disp,
                        valid=valid)

    def sample_actions(self, batch: int, seed: int = 0) -> np.ndarray:
        """A batch of actions from the prior the squashing implies.

        Standard normals: `action.decode` squashes through tanh, so this covers
        the device's whole legal gesture space without piling up on the bounds.
        """
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, (batch, self.action_dim))
