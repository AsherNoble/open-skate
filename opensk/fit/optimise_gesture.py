"""Search for a gesture that produces a target board rotation, in simulation.

This is the front half of the round-trip transfer test, which is the only
experiment that validates the project's premise: find a gesture HERE, execute
it on a real phone, and see whether the real board does the same thing.

The target is stated as ROTATION, never as a trick name. Open Skate has no
notion of a kickflip; it has a board that rolls about its long axis and yaws
about its normal, and the rig's OCR is what turns an outcome into a name. That
separation is what makes the test meaningful — if this module knew what a
kickflip was, matching one would prove nothing.

The gesture is emitted in the schema `GESTURES.md` defines, so it can be
executed on the device with no translation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..sim.core import SkateSim
from ..sim.gesture_spec import X_BOUND_MIN, Y_BOUND_MAX, Y_BOUND_MIN
from ..sim.params import SkateParams
from ..sim.state import quat_delta
from ..sim.touch import TouchModel

# One gesture slot: 3 waypoints (x, y), duration, easing power.
PARAMS_PER_SLOT = 8


@dataclass(frozen=True)
class Outcome:
    roll_deg: float          # about the deck's long axis (flips)
    yaw_deg: float           # about the deck's normal (shuvits)
    peak_height: float       # metres above the resting ride height
    landed: bool             # all four wheels back down, board upright
    airborne_s: float
    displacement: float = 0.0   # furthest the board travelled from its start


def decode(vec: np.ndarray, n_slots: int = 2) -> dict:
    """Flat vector -> gesture recipe, in GESTURES.md's schema.

    Layout mirrors the rig's: per slot 3 waypoints, duration, easing power,
    then n-1 inter-slot delays.
    """
    gestures = []
    for i in range(n_slots):
        b = i * PARAMS_PER_SLOT
        pts = [[float(np.clip(vec[b + 2 * j], X_BOUND_MIN, 1.0)),
                float(np.clip(vec[b + 2 * j + 1], Y_BOUND_MIN, Y_BOUND_MAX))]
               for j in range(3)]
        gestures.append({
            "points": pts,
            "duration": float(np.clip(vec[b + 6], 0.05, 0.80)),
            "easing_power": float(np.clip(vec[b + 7], 0.3, 3.0)),
        })
    delays = [float(np.clip(v, -0.25, 0.60))
              for v in vec[n_slots * PARAMS_PER_SLOT:
                           n_slots * PARAMS_PER_SLOT + n_slots - 1]]
    return {"gestures": gestures, "delays": delays}


def simulate(recipe: dict, params: SkateParams | None = None,
             settle: float = 1.6) -> Outcome:
    """Run a recipe and report what the board actually did."""
    params = params or SkateParams()
    sim = SkateSim(params)
    sim.reset(seed=0)
    sim.step(200)
    touch = TouchModel(sim)

    prev = sim.state().quat.copy()
    roll = yaw = 0.0
    peak = 0.0
    airborne = 0
    rest = sim.state().pos[2]
    start_xy = sim.state().pos[:2].copy()
    displacement = 0.0
    last = None
    for st in touch.run_iter(recipe, push=False, settle=settle):
        d = quat_delta(prev, st.quat)
        prev = st.quat.copy()
        n = float(np.linalg.norm(d[1:]))
        if n > 1e-12:
            ang = 2.0 * np.arctan2(n, d[0])
            axis = d[1:] / n
            local = sim.data.xmat[sim.deck_bid].reshape(3, 3).T @ axis
            roll += ang * local[0]
            yaw += ang * local[2]
        peak = max(peak, st.pos[2])
        if st.airborne:
            airborne += 1
        displacement = max(displacement,
                           float(np.linalg.norm(st.pos[:2] - start_xy)))
        last = st
    upright = last is not None and abs(last.quat[0]) > 0.85
    landed = bool(last is not None and last.wheel_contact.all() and upright)
    return Outcome(float(np.degrees(roll)), float(np.degrees(yaw)),
                   float(peak - rest), landed, airborne * params.timestep,
                   displacement)


def loss(out: Outcome, target_roll: float = 360.0,
         target_yaw: float = 0.0, max_displacement: float = 0.6) -> float:
    """Distance from the requested rotation, with landing preferred.

    Rotation error dominates; height and landing are shaping terms. A gesture
    that rotates correctly but never leaves the ground is not the same trick,
    so a minimum air time is required before the rotation counts for much.
    """
    r_err = abs(abs(out.roll_deg) - abs(target_roll)) / 360.0
    y_err = abs(abs(out.yaw_deg) - abs(target_yaw)) / 360.0
    air = 0.0 if out.airborne_s > 0.12 else (0.12 - out.airborne_s) * 4.0
    # Penalise travelling far. A gesture that rotates correctly but carries the
    # board 1.4 m ends up on a rail in the real park -- which is what the first
    # two round-trip tests produced: six of ten trials came back as slides. The
    # simulation replays on flat ground and cannot see the obstacle, so the
    # constraint has to be imposed here rather than discovered.
    travel = max(0.0, out.displacement - max_displacement)
    return (r_err + 0.5 * y_err + air + travel
            + (0.0 if out.landed else 0.35))


def search(target_roll: float = 360.0, target_yaw: float = 0.0, *,
           n_slots: int = 2, evals: int = 600, seed: int = 0,
           max_displacement: float = 0.6,
           params: SkateParams | None = None, verbose: bool = True):
    """CMA-ES over gesture parameters. Returns (recipe, outcome, loss)."""
    import cma

    params = params or SkateParams()
    dim = n_slots * PARAMS_PER_SLOT + (n_slots - 1)
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([
        np.tile(np.array([0.5, 0.55, 0.5, 0.65, 0.5, 0.75, 0.20, 1.5]), n_slots),
        np.full(n_slots - 1, 0.05)])
    x0 += rng.normal(0, 0.02, dim)

    es = cma.CMAEvolutionStrategy(list(x0), 0.12,
                                  {"seed": seed + 1, "maxfevals": evals,
                                   "verbose": -9})
    best = (1e9, None, None)
    while not es.stop():
        xs = es.ask()
        ls = []
        for x in xs:
            rec = decode(np.asarray(x), n_slots)
            out = simulate(rec, params)
            l = loss(out, target_roll, target_yaw, max_displacement)
            ls.append(l)
            if l < best[0]:
                best = (l, rec, out)
        es.tell(xs, ls)
        if verbose and best[2] is not None:
            print(f"  loss {best[0]:.3f}  roll {best[2].roll_deg:+.0f}deg "
                  f"air {best[2].airborne_s:.2f}s travel {best[2].displacement:.2f}m "
                  f"landed {best[2].landed}", flush=True)
    return best[1], best[2], best[0]


def save(recipe: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(recipe, fh, indent=2)
