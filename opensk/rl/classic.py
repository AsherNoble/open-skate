"""Small, deterministic local experiments using the fitted CPU touch model."""
from dataclasses import replace
import numpy as np
import mujoco

from .action import action_dim, to_recipe
from .env import Episodes, summarise, MAX_PLAUSIBLE_HEIGHT_M, MAX_PLAUSIBLE_TRAVEL_M
from ..sim.appearance import resolve_appearance
from ..sim.core import SkateSim
from ..sim.model.parks import PARKS
from ..sim.touch import TouchModel, Finger
from ..sim.gesture_spec import schedule_recipe, spin_window
from ..sim.camera import board_yaw
from ..pose.render import SceneRenderer


def variation(appearance, seed):
    """Only appearance fields vary; never fitted physical or camera parameters."""
    rng = np.random.default_rng(seed)
    preset = resolve_appearance(appearance)
    fields = {}
    for names, scale in ((('ground_light', 'ground_dark', 'concrete_light', 'concrete_dark'), .035),
                         (('grip_light', 'grip_dark', 'underside'), .025),
                         (('key_diffuse', 'fill_diffuse', 'ambient'), .045)):
        offset = rng.uniform(-scale, scale, 3)
        for name in names:
            fields[name] = tuple(np.clip(np.asarray(getattr(preset, name)) + offset, .01, .98))
    return replace(preset, **fields)


# Different views of the EXISTING obstacle composition. Collision layout stays
# authoritative; never draw fake obstacles without corresponding contacts.
ANCHORS = {"flat": ((0., 0.), (2., .3)),
           "plaza": ((0., 0.), (-.25, .25)),
           "sls": ((-9., 2.5), (-4.3, -3.5))}


class ClassicEnv:
    pixels = True

    def __init__(self, *, park="flat", appearance="day", batch=1,
                 seconds=1.2, settle_steps=200, seed=0, vary=False):
        if park not in PARKS:
            raise ValueError(f"unknown park {park}")
        if seconds <= 0 or settle_steps < 0 or batch < 1:
            raise ValueError("invalid duration/settle/batch")
        self.park_name, self.appearance, self.batch = park, appearance, batch
        self.seconds, self.settle_steps = seconds, settle_steps
        self.seed, self.vary = seed, vary
        self.action_dim = action_dim()
        self.last_extras = {}

    def sample_actions(self, batch, seed=0):
        return np.random.default_rng(seed).normal(size=(batch, self.action_dim))

    def step(self, actions):
        actions = np.atleast_2d(actions)
        results, extras = [], []
        for i, action in enumerate(actions):
            result, extra = self.run_recipe(to_recipe(action), self.seed + i)
            results.append(result)
            extras.append(extra)
        self.last_extras = {k: np.stack([e[k] for e in extras]) for k in extras[0]}
        return Episodes(*(np.concatenate([getattr(r, k) for r in results])
                          for k in Episodes._fields))

    def run_recipe(self, recipe, seed):
        rng = np.random.default_rng(seed)
        appearance = variation(self.appearance, seed) if self.vary else self.appearance
        sim = SkateSim(park=PARKS[self.park_name], appearance=appearance)
        anchor = ANCHORS[self.park_name][seed % 2] if self.vary else (0., 0.)
        pos = np.asarray(anchor) + (rng.uniform(-.08, .08, 2) if self.vary else 0.)
        sim.reset(seed=seed, pos=tuple(pos))
        sim.step(self.settle_steps)
        rest_z = sim.state().pos[2]
        touch = TouchModel(sim)
        schedule = schedule_recipe(recipe)
        spin = spin_window(recipe, max((s+p.duration for s,p in schedule), default=0.))
        fingers = [Finger(p, s) for s, p in schedule]
        dt = sim.params.timestep
        n = int(round(self.seconds / dt))
        # Initial frame at t=0 followed by exact integer physics steps.
        indices = set(np.rint(np.arange(1, int(self.seconds*30)+1) / 30 / dt).astype(int))
        states = [sim.state()]
        rendered, masks, times, sampled = [], [], [], []
        obstacle_pixels = []
        renderer = SceneRenderer.for_mode(sim)
        obstacle_ids = [j for j in range(sim.model.ngeom)
                        if (sim.model.geom(j).name or '').startswith('parkvis_')
                        and sim.model.geom(j).name not in ('parkvis_ground', 'parkvis_contest_flat')]
        def capture(t):
            rendered.append(renderer.render(touch.camera).copy())
            masks.append(renderer.board_pixels(touch.camera).copy())
            sampled.append(sim.state())
            times.append(t)
            renderer._renderer.enable_segmentation_rendering()
            try:
                seg = renderer.render(touch.camera)[:, :, 0]
                obstacle_pixels.append(int(np.isin(seg, obstacle_ids).sum()))
            finally:
                renderer._renderer.disable_segmentation_rendering()
        try:
            capture(0.)
            for step in range(n):
                t = step * dt
                state = sim.state()
                touch.camera.update(state.pos, board_yaw(state.quat), dt)
                for finger in fingers:
                    if finger.active(t):
                        touch._apply_finger(finger, t, dt)
                if spin and spin[0] <= t <= spin[1]:
                    sim.data.xfrc_applied[sim.deck_bid, 3:] += sim.params.spin_torque * sim.deck_frame()[1][:, 2]
                states.append(sim.step())
                if step + 1 in indices:
                    capture((step+1)*dt)
        finally:
            renderer._renderer.close()
        positions = np.array([s.pos for s in states])
        quats = np.array([s.quat for s in states])
        roll, yaw, peak, air, disp = summarise(positions, quats, dt, rest_z, np)
        valid = bool(np.isfinite([roll,yaw,peak,air,disp]).all()
                     and peak < MAX_PLAUSIBLE_HEIGHT_M and disp < MAX_PLAUSIBLE_TRAVEL_M)
        ep = Episodes(np.array([[s.pos for s in sampled]]),
                      np.array([[s.quat for s in sampled]]),
                      *[np.array([v]) for v in (roll,yaw,peak,air,disp,valid)],
                      np.array([rendered], dtype=np.uint8))
        return ep, {"frame_times": np.array(times), "deck_mask": np.array(masks),
                    "obstacle_pixels": np.array(obstacle_pixels),
                    "initial_xy": pos, "variation_seed": np.asarray(seed)}
