"""The analytic ray cast must match `mj_ray`, because it replaces it.

`sim/touch.py` calls mj_ray every substep; MJX-JAX's general ray support is
documented slow, and a BVH query is over-powered for "does this ray hit one of
three known boxes or the ground". The closed-form version has to be exactly
equivalent or the ported touch model is a different model.
"""
import mujoco
import numpy as np

from opensk.mjx.geom import cast_deck_or_ground, deck_boxes_from
from opensk.sim.core import SkateSim
from opensk.sim.params import SkateParams
from opensk.sim.touch import ON_DECK, TouchModel


def _cases(n_poses: int = 5, n_rays: int = 250):
    """Random screen rays over several board poses, not just the resting one."""
    sim = SkateSim(SkateParams())
    sim.reset(seed=0)
    sim.step(200)
    tm = TouchModel(sim)
    rng = np.random.default_rng(0)
    for trial in range(n_poses):
        if trial:
            sim.data.qpos[0] += rng.uniform(-0.3, 0.3)
            sim.data.qpos[2] += rng.uniform(0.0, 0.25)
            ang = rng.uniform(-0.6, 0.6)
            sim.data.qpos[3] = np.cos(ang / 2)
            sim.data.qpos[4] = np.sin(ang / 2)
            mujoco.mj_forward(sim.model, sim.data)
        boxes = deck_boxes_from(sim)
        for _ in range(n_rays):
            nx, ny = rng.uniform(0.15, 0.95), rng.uniform(0.30, 0.88)
            kind, hit = tm.cast(nx, ny)
            o, d = tm.camera.ray(nx, ny)
            hd, t = cast_deck_or_ground(o, d, boxes)
            yield kind, hit, o, d, bool(hd), float(t)


def test_deck_vs_ground_classification_matches_mj_ray():
    agree = total = 0
    for kind, _, _, _, hd, _ in _cases():
        total += 1
        agree += int(hd == (kind == ON_DECK))
    assert total > 500
    assert agree == total, f"classification differs on {total - agree}/{total} rays"


def test_deck_hit_point_matches_to_machine_precision():
    """Only deck hits matter: that is the only case touch.py uses the point.

    Non-deck rays legitimately differ -- mj_ray can strike a wheel or truck
    where the analytic cast reports the ground behind it -- but both classify
    the finger as not-on-deck, so the touch model behaves identically.
    """
    worst = 0.0
    n = 0
    for kind, hit, o, d, _, t in _cases():
        if kind != ON_DECK or hit is None:
            continue
        n += 1
        worst = max(worst, float(np.linalg.norm(o + d * t - hit)))
    assert n > 50, f"too few deck hits to be meaningful ({n})"
    assert worst < 1e-9, f"deck hit point differs by {worst:.2e} m"
