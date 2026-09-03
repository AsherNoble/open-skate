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

    __slots__ = ("path", "t0", "kind", "local", "depth", "released",
                 "_prev_screen", "is_push")

    def __init__(self, path: GesturePath, t0: float, is_push: bool = False):
        self.path = path
        self.t0 = t0
        self.kind = None
        self.local = None   # deck-local (x, y) the finger is currently over
        self.depth = 0.0    # view depth the fingertip is held at, from touch-down
        self.released = False
        self._prev_screen = None
        # The deliberate push gesture and an incidental ground touch during a
        # trick are different mechanics; they take different gains.
        self.is_push = is_push

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
        # Ray casts see collision geometry only: group 0 (ground, trucks,
        # wheels) and group 3 (the deck's collision boxes). Group 2 is the
        # visual popsicle outline, which has no contact at all and must not
        # intercept touches.
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[0] = 1
        self._geomgroup[3] = 1

    # -- ray casting -------------------------------------------------------

    def cast(self, nx: float, ny: float) -> tuple[str, np.ndarray | None]:
        """Screen point -> (what it hit, world hit point)."""
        origin, direction = self.camera.ray(nx, ny)
        dist = mujoco.mj_ray(self.sim.model, self.sim.data, origin, direction,
                             self._geomgroup, 1, -1, self._geomid)
        if dist < 0:
            return MISSED, None
        hit = origin + direction * dist
        gid = int(self._geomid[0])
        return (ON_DECK if gid in self._deck_gids else ON_GROUND), hit

    # -- per-substep force -------------------------------------------------

    def _apply_finger(self, f: Finger, t: float, dt: float) -> None:
        local_t = t - f.t0
        nx, ny = f.path.position_at(local_t)

        if f.kind is None:      # touch-down
            kind, hit = self.cast(nx, ny)
            f.kind = kind
            f._prev_screen = np.array([nx, ny])
            if kind == ON_DECK:
                f.local = self.sim.local_point(hit)
                f.depth = self.camera.depth_of(hit)
            return

        if f.kind == ON_GROUND:
            # A finger that started off the board can still catch it: real
            # trick gestures commonly begin on the ground behind the tail and
            # flick up through the deck. Deciding contact once at touch-down
            # made 51% of captured trick recipes do NOTHING in simulation --
            # their start points sit at screen y 0.70-0.88, below a deck
            # spanning 0.45-0.73 -- while the real board plainly moved. So
            # contact is re-tested every substep until the finger grabs.
            kind, hit = self.cast(nx, ny)
            if kind == ON_DECK:
                f.kind = ON_DECK
                f.local = self.sim.local_point(hit)
                f.depth = self.camera.depth_of(hit)

        if f.kind == ON_DECK:
            # The finger holds the MATERIAL point it first touched, and pulls
            # it toward the fingertip. The grab is not released early: a
            # sliding contact was tried and falsified immediately (from the
            # tail tip, any down-screen drag carries the contact point off the
            # end of the deck within 2-3 substeps), and a hard release past a
            # fixed distance left the finger disengaged for 76% of the median
            # real gesture. Instead the force simply saturates at
            # touch_force_max, which reads as "you are pulling the board that
            # way as hard as a finger can".
            # The contact STICKS or SLIPS, by a Coulomb limit.
            #
            # This is what makes a flip possible at all. Roll torque is
            # r_y * F_z: a downward force at a laterally offset contact. With
            # the contact pinned where the finger first landed -- near the
            # centreline for these recipes -- r_y is about zero and no pull can
            # roll the board, which is why only 2% of captured trick recipes
            # reached a full flip and the median roll was 1 degree. Letting go
            # at the edge is worse still (0% full flips): it cuts the force off
            # before it does any work.
            #
            # Sliding all the way, every step, is not right either: it costs
            # the ollie, because a tail press only pops while the contact
            # STAYS on the tail. A Coulomb limit gives both -- a press is
            # mostly normal force so it sticks, a flick is mostly tangential
            # so it slips out to the rail and rolls the board.
            origin, R = self.sim.deck_frame()
            contact = self.sim.body_point(f.local)
            tip_now = self.camera.point_at_depth(
                nx, ny, self.camera.depth_of(contact))
            pull = tip_now - contact
            normal = R[:, 2]
            f_n = float(pull @ normal)
            tangent = pull - f_n * normal
            if np.linalg.norm(tangent) > self.p.touch_friction * abs(f_n):
                rel = R.T @ (tip_now - origin)
                half_l = 0.5 * self.p.deck_length
                half_w = 0.5 * self.p.deck_width
                f.local = np.array([
                    float(np.clip(rel[0], -half_l, half_l)),
                    float(np.clip(rel[1], -half_w, half_w)),
                    0.5 * self.p.deck_thickness])
                contact = self.sim.body_point(f.local)
            tip = self.camera.point_at_depth(nx, ny,
                                             self.camera.depth_of(contact))
            err = tip - contact
            vel = self._point_velocity(contact)
            force = self.p.touch_gain * err - self.p.touch_damping * vel
            mag = float(np.linalg.norm(force))
            if mag > self.p.touch_force_max:
                force *= self.p.touch_force_max / mag
            self.sim.apply_force(force, contact)

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
            gain = (self.p.push_impulse_gain if f.is_push
                    else self.p.ground_shove_gain)
            thrust = sign * gain * travel / dt
            # Capped like the deck force, and for the same reason: a finger
            # can only push as hard as a finger can. Uncapped, a fast
            # full-screen drag asks for kilonewtons and the solver goes
            # unstable -- of 128 gestures drawn from the environment's prior
            # only 26 stayed physical, and 26 went non-finite. The cap never
            # binds on the captured corpus, so the fitted parameters are
            # unaffected (see results/THROUGHPUT.md).
            thrust = float(np.clip(thrust, -self.p.touch_force_max,
                                   self.p.touch_force_max))
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
        """Execute a full recipe and return the state trajectory."""
        return list(self.run_iter(recipe, push=push, settle=settle))

    def run_iter(self, recipe: dict, *, push: bool = True,
                 settle: float = 0.6):
        """Execute a recipe, yielding the state after every physics substep.

        Generator form so callers can sample at arbitrary timestamps without
        materialising the whole trajectory — the fitting objective needs the
        state at ~11 real frame times out of several hundred substeps.

        Mirrors the device execution flow in GESTURES.md: optional push, wait
        out PUSH_PRE_DELAY, fire the scheduled gestures, then settle so the
        landing is part of the trajectory. Note the capture this is fitted
        against does NOT push.
        """
        dt = self.p.timestep
        schedule = schedule_recipe(recipe)
        total = max((t0 + g.duration for t0, g in schedule), default=0.0)
        spin = spin_window(recipe, total)
        fingers = [Finger(g, t0) for t0, g in schedule]

        if push:
            self._run_push()

        t, end = 0.0, total + settle
        while t < end:
            st = self.sim.state()
            self.camera.update(st.pos, board_yaw(st.quat), dt)
            for f in fingers:
                if f.active(t):
                    self._apply_finger(f, t, dt)
            if spin and spin[0] <= t <= spin[1]:
                # The rotate button: a yaw torque about the deck's own normal,
                # so it spins the board in its own frame even mid-flip.
                n = self.sim.deck_frame()[1][:, 2]
                self.sim.data.xfrc_applied[self.sim.deck_bid, 3:] += \
                    self.p.spin_torque * n
            yield self.sim.step()
            t += dt

    def _run_push(self) -> None:
        """The standard push, then the remainder of PUSH_PRE_DELAY."""
        from .gesture_spec import push_path
        dt = self.p.timestep
        path = push_path()
        f = Finger(path, 0.0, is_push=True)
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
