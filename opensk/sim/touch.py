"""Screen gestures -> forces on the board.

The one place where True Skate's input model meets MuJoCo. A finger presses on
the deck, grabs the material point under it, and drags that point toward where
the fingertip now is — a capped spring-damper, which is why hard flicks slip
off instead of teleporting the board.

Nothing here knows about tricks. An ollie is what happens when a finger drags
the tail down hard enough for it to strike the ground; a kickflip is what
happens when the drag also runs off the deck's edge. That emergence is the
point: it is what makes the physics parameters worth fitting.

Forces go through `SkateSim.apply_force` into `xfrc_applied`, which exists in
MJX, so this whole pathway ports to GPU unchanged.
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("Open Skate needs mujoco") from exc

from .camera import FollowCamera, board_yaw
from .core import SkateSim
from .gesture_spec import (GesturePath, PUSH_PRE_DELAY, schedule_recipe,
                           spin_window)

# Where a touch landed, decided once at touch-down and then held.
ON_DECK, ON_GROUND, MISSED = "deck", "ground", "missed"


class Finger:
    """One touch, from touch-down to lift.

    The grab point is recorded in the deck's LOCAL frame, so it stays with the
    material point through arbitrary rotation — that is what lets a finger
    ride the deck through a flip instead of being left behind in world space.
    """

    __slots__ = ("path", "t0", "kind", "local", "depth", "released", "_prev_screen")

    def __init__(self, path: GesturePath, t0: float):
        self.path = path
        self.t0 = t0
        self.kind = None
        self.local = None
        self.depth = 0.0
        self.released = False
        self._prev_screen = None

    def active(self, t: float) -> bool:
        return not self.released and self.t0 <= t <= self.t0 + self.path.duration


class TouchModel:
    """Applies a gesture recipe to a `SkateSim`, one physics substep at a time."""

    def __init__(self, sim: SkateSim, camera: FollowCamera | None = None):
        self.sim = sim
        self.p = sim.params
        self.camera = camera or FollowCamera(sim.params)
        st = sim.state()
        self.camera.reset(st.pos, board_yaw(st.quat))
        self._deck_gids = sim._deck_gids
        self._geomid = np.zeros(1, dtype=np.int32)

    # -- ray casting -------------------------------------------------------

    def cast(self, nx: float, ny: float) -> tuple[str, np.ndarray | None]:
        """Screen point -> (what it hit, world hit point)."""
        origin, direction = self.camera.ray(nx, ny)
        dist = mujoco.mj_ray(self.sim.model, self.sim.data, origin, direction,
                             None, 1, -1, self._geomid)
        if dist < 0:
            return MISSED, None
        hit = origin + direction * dist
        gid = int(self._geomid[0])
        return (ON_DECK if gid in self._deck_gids else ON_GROUND), hit

    # -- per-substep force -------------------------------------------------

    def _apply_finger(self, f: Finger, t: float, dt: float) -> None:
        local_t = t - f.t0
        nx, ny = f.path.position_at(local_t)

        if f.kind is None:  # touch-down: decide what was grabbed, once
            kind, hit = self.cast(nx, ny)
            f.kind = kind
            f._prev_screen = np.array([nx, ny])
            if kind == ON_DECK:
                f.local = self.sim.local_point(hit)
                f.depth = self.camera.depth_of(hit)
            return

        if f.kind == ON_DECK:
            grab_world = self.sim.body_point(f.local)
            # Track the fingertip at the depth of the grabbed point, so lateral
            # screen motion is lateral world motion rather than motion toward
            # the camera.
            depth = self.camera.depth_of(grab_world)
            target = self.camera.point_at_depth(nx, ny, depth)
            err = target - grab_world
            if np.linalg.norm(err) > self.p.touch_slip_distance:
                f.released = True  # dragged off the deck: the grip lets go
                return
            vel = self._point_velocity(grab_world)
            force = self.p.touch_gain * err - self.p.touch_damping * vel
            mag = float(np.linalg.norm(force))
            if mag > self.p.touch_force_max:
                force *= self.p.touch_force_max / mag
            self.sim.apply_force(force, grab_world)

        elif f.kind == ON_GROUND:
            # A push. Each substep contributes an impulse proportional to the
            # screen distance covered in it, so the total impulse over the
            # gesture depends on how far the finger travelled and not on how
            # quickly the device happened to execute it.
            cur = np.array([nx, ny])
            step = cur - f._prev_screen
            f._prev_screen = cur
            travel = float(np.linalg.norm(step))
            if travel <= 0.0:
                return
            # Dragging down the screen pushes forward, and up pushes backward,
            # which is what a foot does against the ground.
            sign = 1.0 if step[1] >= 0.0 else -1.0
            yaw = board_yaw(self.sim.state().quat)
            thrust = sign * self.p.push_impulse_gain * travel / dt
            self.sim.apply_force(
                np.array([thrust * np.cos(yaw), thrust * np.sin(yaw), 0.0]),
                self.sim.data.xipos[self.sim.deck_bid].copy())

    def _point_velocity(self, world_point: np.ndarray) -> np.ndarray:
        """Velocity of the deck's material point at `world_point`."""
        st = self.sim.state()
        r = world_point - self.sim.data.xipos[self.sim.deck_bid]
        return st.linvel + np.cross(st.angvel, r)

    # -- execution ---------------------------------------------------------

    def run(self, recipe: dict, *, push: bool = True, settle: float = 0.6,
            record: bool = True) -> list:
        """Execute a full recipe and return the state trajectory.

        Mirrors the device execution flow in GESTURES.md: push, wait out
        PUSH_PRE_DELAY, fire the scheduled gestures, then let the board settle
        so the landing is part of the trajectory.
        """
        dt = self.p.timestep
        schedule = schedule_recipe(recipe)
        total = max((t0 + g.duration for t0, g in schedule), default=0.0)
        spin = spin_window(recipe, total)
        fingers = [Finger(g, t0) for t0, g in schedule]

        if push:
            self._run_push()

        traj = []
        t, end = 0.0, total + settle
        while t < end:
            self.camera.update(self.sim.state().pos,
                               board_yaw(self.sim.state().quat), dt)
            for f in fingers:
                if f.active(t):
                    self._apply_finger(f, t, dt)
            if spin and spin[0] <= t <= spin[1]:
                # The rotate button: a yaw torque about the deck's own normal,
                # so it spins the board in its own frame even mid-flip.
                n = self.sim.deck_frame()[1][:, 2]
                self.sim.data.xfrc_applied[self.sim.deck_bid, 3:] += \
                    self.p.spin_torque * n
            st = self.sim.step()
            if record:
                traj.append(st)
            t += dt
        return traj

    def _run_push(self) -> None:
        """The standard push, then the remainder of PUSH_PRE_DELAY."""
        from .gesture_spec import push_path
        dt = self.p.timestep
        path = push_path()
        f = Finger(path, 0.0)
        t = 0.0
        while t <= path.duration:
            self.camera.update(self.sim.state().pos,
                               board_yaw(self.sim.state().quat), dt)
            self._apply_finger(f, t, dt)
            self.sim.step()
            t += dt
        remaining = PUSH_PRE_DELAY - path.duration
        for _ in range(max(0, int(remaining / dt))):
            self.camera.update(self.sim.state().pos,
                               board_yaw(self.sim.state().quat), dt)
            self.sim.step()
