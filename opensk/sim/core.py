"""SkateSim — the physics core.

Pure in the sense that matters for this project: no wall-clock, no rendering,
no file or network I/O, fixed timestep, and every random draw comes from a
seeded generator owned by the instance. Two SkateSims built from the same
params, reset with the same seed and fed the same forces produce bitwise
identical trajectories. `tests/test_determinism.py` enforces that, because
the MJX port and every batched world-model rollout depend on it.
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("Open Skate needs mujoco: pip install mujoco") from exc

from .model.build import FLAT_PARK, build_scene, ride_height
from .params import SkateParams
from .state import WHEELS, State

# Static contact penetration at rest, metres. Small enough to be physically
# meaningless, large enough that the contact solver sees the wheels.
_SPAWN_SAG = 5e-4

_STEER_JOINTS = ("front_steer", "rear_steer")
_SPIN_JOINTS = tuple(f"{w}_spin" for w in WHEELS)


class SkateSim:
    def __init__(self, params: SkateParams | None = None, park: str = FLAT_PARK):
        self.params = params or SkateParams()
        self.park = park
        self.model = mujoco.MjModel.from_xml_string(build_scene(self.params, park))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(0)

        n = mujoco.mj_name2id
        M, O = self.model, mujoco.mjtObj
        self.deck_bid = n(M, O.mjOBJ_BODY, "deck")
        self._steer_q = [M.jnt_qposadr[n(M, O.mjOBJ_JOINT, j)] for j in _STEER_JOINTS]
        self._spin_v = [M.jnt_dofadr[n(M, O.mjOBJ_JOINT, j)] for j in _SPIN_JOINTS]
        self._wheel_gids = {n(M, O.mjOBJ_GEOM, w): i for i, w in enumerate(WHEELS)}
        def _named(prefix):
            return {i for i in range(M.ngeom)
                    if (mujoco.mj_id2name(M, O.mjOBJ_GEOM, i) or "").startswith(prefix)}

        # Collision geoms of the deck: what contact and ray casts see.
        self._deck_gids = _named("deck_")
        # What the silhouette comparison sees: the deck outline ONLY.
        #
        # Trucks and wheels were briefly included, to break the flip symmetry
        # that made per-frame pose fitting ambiguous. That cost more than it
        # bought: the wheels sit at the deck's half-width plus their own
        # radius, so they widen the silhouette by ~26% and broke the width
        # match the camera had been calibrated on (measured +22.3% against
        # real frames, where length and framing were within 1%). The real
        # board at rest shows a clean deck outline with the hardware hidden
        # underneath -- real width 0.1104 matches deck-only 0.1116, not
        # deck-plus-wheels 0.1350. Pose fitting has since been abandoned, so
        # the reason for including them is gone.
        self._deck_visual_gids = _named("vis_")
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: int | None = None, speed: float = 0.0,
              pos: tuple[float, float] = (0.0, 0.0), heading: float = 0.0) -> State:
        """Place the board at rest (or rolling) at a known anchor.

        `speed` spins the wheels to match, so a rolling start begins rolling
        rather than skidding — the difference is a large transient in the
        first tenth of a second and would otherwise pollute every fit.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        q[0], q[1] = pos
        # Spawn a hair below the geometric ride height: at exactly zero gap
        # MuJoCo registers no contact, so the board would report itself
        # airborne on the first frame and a rolling start would begin with a
        # free-fall transient.
        q[2] = ride_height(self.params) - _SPAWN_SAG
        h = 0.5 * heading
        q[3], q[4], q[5], q[6] = np.cos(h), 0.0, 0.0, np.sin(h)
        if speed:
            self.data.qvel[0] = speed * np.cos(heading)
            self.data.qvel[1] = speed * np.sin(heading)
            for a in self._spin_v:
                self.data.qvel[a] = speed / self.params.wheel_radius
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    # -- stepping ----------------------------------------------------------

    def apply_force(self, force_world: np.ndarray, point_world: np.ndarray) -> None:
        """Accumulate an external force applied at a world point on the deck.

        Cleared by `step()`, so it must be re-applied every substep — which is
        what the touch model does.

        Writes into xfrc_applied, which takes force and torque about the body
        COM, so the off-centre lever arm is formed explicitly here. That field
        exists in MJX, so the whole touch pathway ports to GPU unchanged --
        mj_applyFT would have been equivalent on CPU but targets qfrc, which
        does not.
        """
        f = np.asarray(force_world, dtype=np.float64)
        r = np.asarray(point_world, dtype=np.float64) - self.data.xipos[self.deck_bid]
        self.data.xfrc_applied[self.deck_bid, :3] += f
        self.data.xfrc_applied[self.deck_bid, 3:] += np.cross(r, f)

    def step(self, n: int = 1) -> State:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[:] = 0.0
        return self.state()

    def advance(self, duration: float) -> int:
        """Step for `duration` seconds. Returns the number of steps taken."""
        n = int(round(duration / self.params.timestep))
        self.step(n)
        return n

    # -- observation -------------------------------------------------------

    def state(self) -> State:
        d, m = self.data, self.model
        wheel_contact = np.zeros(4, dtype=bool)
        deck_contact = False
        for c in d.contact[: d.ncon]:
            for g in (c.geom1, c.geom2):
                if g in self._wheel_gids:
                    wheel_contact[self._wheel_gids[g]] = True
                elif g in self._deck_gids:
                    deck_contact = True
        return State(
            t=float(d.time),
            pos=d.qpos[0:3].copy(),
            quat=d.qpos[3:7].copy(),
            linvel=d.qvel[0:3].copy(),
            angvel=d.qvel[3:6].copy(),
            steer=np.array([d.qpos[a] for a in self._steer_q]),
            wheel_spin=np.array([d.qvel[a] for a in self._spin_v]),
            wheel_contact=wheel_contact,
            deck_contact=deck_contact,
        )

    # -- geometry helpers used by the touch model --------------------------

    def deck_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """(origin, 3x3 basis) of the deck body in world coordinates."""
        return (self.data.xpos[self.deck_bid].copy(),
                self.data.xmat[self.deck_bid].reshape(3, 3).copy())

    def body_point(self, local: np.ndarray) -> np.ndarray:
        """A point in the deck's local frame -> world coordinates."""
        o, R = self.deck_frame()
        return o + R @ np.asarray(local, dtype=np.float64)

    def local_point(self, world: np.ndarray) -> np.ndarray:
        """World point -> the deck's local frame."""
        o, R = self.deck_frame()
        return R.T @ (np.asarray(world, dtype=np.float64) - o)
