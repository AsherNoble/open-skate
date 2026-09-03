"""The action space: one flat vector, one whole gesture.

An episode is a single gesture, because that is the unit the device executes
and the unit the capture rig records. The layout is exactly the one
`fit/optimise_gesture.decode` already defines, so a vector produced here is
executable on a phone with no translation -- which is the property that makes
anything learned in simulation worth learning.

    per slot:  x0 y0 x1 y1 x2 y2 duration easing_power     (8)
    then:      n_slots - 1 inter-slot delays

Bounds are the device's, not ours: `X_BOUND_MIN` and the y bounds come from the
rig's own touch code, and a point outside them is not a gesture the phone can
perform. Squashing rather than clipping, so a policy always gets a gradient and
never has to learn to avoid a flat region.
"""
from __future__ import annotations

import numpy as np

from ..sim.gesture_spec import X_BOUND_MIN, Y_BOUND_MAX, Y_BOUND_MIN

PARAMS_PER_SLOT = 8

# Matching `fit/optimise_gesture.decode` exactly.
DURATION_MIN, DURATION_MAX = 0.05, 0.80
EASING_MIN, EASING_MAX = 0.3, 3.0
DELAY_MIN, DELAY_MAX = -0.25, 0.60


def action_dim(n_slots: int = 2) -> int:
    return n_slots * PARAMS_PER_SLOT + (n_slots - 1)


def _squash(v, lo, hi, xp=np):
    """Unbounded real -> (lo, hi), smoothly.

    tanh rather than clip: a clipped policy that wanders out of bounds gets no
    signal telling it which way to come back.
    """
    return lo + 0.5 * (hi - lo) * (xp.tanh(v) + 1.0)


def decode(vec, n_slots: int = 2, xp=np):
    """Flat action -> (points (S,3,2), durations (S,), easings (S,), delays (S-1,)).

    Pure and traceable, so a policy's output can go straight into a batched
    rollout without leaving the accelerator.
    """
    body = vec[:n_slots * PARAMS_PER_SLOT].reshape(n_slots, PARAMS_PER_SLOT)
    xs = _squash(body[:, 0:6:2], X_BOUND_MIN, 1.0, xp=xp)
    ys = _squash(body[:, 1:6:2], Y_BOUND_MIN, Y_BOUND_MAX, xp=xp)
    points = xp.stack([xs, ys], axis=-1)                    # (S, 3, 2)
    durations = _squash(body[:, 6], DURATION_MIN, DURATION_MAX, xp=xp)
    easings = _squash(body[:, 7], EASING_MIN, EASING_MAX, xp=xp)
    delays = _squash(vec[n_slots * PARAMS_PER_SLOT:], DELAY_MIN, DELAY_MAX, xp=xp)
    return points, durations, easings, delays


def to_recipe(vec, n_slots: int = 2) -> dict:
    """Flat action -> the JSON recipe the device executes, for round-tripping.

    This is the whole point of matching the device's schema: an action found by
    a policy here is a file the rig can run on a phone unchanged.
    """
    points, durations, easings, delays = decode(np.asarray(vec, dtype=float), n_slots)
    return {"gestures": [{"points": points[i].tolist(),
                          "duration": float(durations[i]),
                          "easing_power": float(easings[i])}
                         for i in range(n_slots)],
            "delays": [float(d) for d in delays]}
