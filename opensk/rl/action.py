"""The action space: one flat vector, one whole gesture.

An episode is a single gesture, because that is the unit the device executes
and the unit the capture rig records. The layout mirrors the rig's
`rl.cmaes.action_param` contract, including its optional rotate-button hold, so
a vector produced here is executable on a phone with no translation -- which
is the property that makes anything learned in simulation worth learning.

    per slot:  x0 y0 x1 y1 x2 y2 duration easing_power     (8)
    then:      n_slots - 1 inter-slot delays
    then:      spin gate, spin start fraction, spin end fraction       (3)

Bounds are the device's, not ours: `X_BOUND_MIN` and the y bounds come from the
rig's own touch code, and a point outside them is not a gesture the phone can
perform. Squashing rather than clipping, so a policy always gets a gradient and
never has to learn to avoid a flat region.
"""
from __future__ import annotations

import numpy as np

from ..sim.gesture_spec import X_BOUND_MIN, Y_BOUND_MAX, Y_BOUND_MIN

PARAMS_PER_SLOT = 8
SPIN_PARAMS = 3

# Matching TrueSkate-AI's `rl.cmaes.action_param` bounds.
DURATION_MIN, DURATION_MAX = 0.05, 0.80
EASING_MIN, EASING_MAX = 0.3, 3.0
DELAY_MIN, DELAY_MAX = -0.25, 0.60


def action_dim(n_slots: int = 2) -> int:
    return n_slots * PARAMS_PER_SLOT + (n_slots - 1) + SPIN_PARAMS


def _squash(v, lo, hi, xp=np):
    """Unbounded real -> (lo, hi), smoothly.

    tanh rather than clip: a clipped policy that wanders out of bounds gets no
    signal telling it which way to come back.
    """
    return lo + 0.5 * (hi - lo) * (xp.tanh(v) + 1.0)


def decode(vec, n_slots: int = 2, xp=np):
    """Flat action -> points, durations, easings, delays and spin controls.

    Pure and traceable, so a policy's output can go straight into a batched
    rollout without leaving the accelerator.
    """
    expected = action_dim(n_slots)
    if vec.shape[-1] != expected:
        raise ValueError(f"expected {expected} action values for {n_slots} slots, "
                         f"got {vec.shape[-1]}")
    body = vec[:n_slots * PARAMS_PER_SLOT].reshape(n_slots, PARAMS_PER_SLOT)
    xs = _squash(body[:, 0:6:2], X_BOUND_MIN, 1.0, xp=xp)
    ys = _squash(body[:, 1:6:2], Y_BOUND_MIN, Y_BOUND_MAX, xp=xp)
    points = xp.stack([xs, ys], axis=-1)                    # (S, 3, 2)
    durations = _squash(body[:, 6], DURATION_MIN, DURATION_MAX, xp=xp)
    easings = _squash(body[:, 7], EASING_MIN, EASING_MAX, xp=xp)
    delay_start = n_slots * PARAMS_PER_SLOT
    spin_start = delay_start + n_slots - 1
    delays = _squash(vec[delay_start:spin_start], DELAY_MIN, DELAY_MAX, xp=xp)
    spin_raw = vec[spin_start:spin_start + SPIN_PARAMS]
    a = _squash(spin_raw[1], 0.0, 1.0, xp=xp)
    b = _squash(spin_raw[2], 0.0, 1.0, xp=xp)
    # Shape: [enabled-as-0/1, start fraction, end fraction].  The gate uses
    # the rig's exact >= 0 contract rather than another arbitrary threshold.
    spin = xp.stack([xp.where(spin_raw[0] >= 0.0, 1.0, 0.0),
                     xp.minimum(a, b), xp.maximum(a, b)])
    return points, durations, easings, delays, spin


def to_recipe(vec, n_slots: int = 2) -> dict:
    """Flat action -> the JSON recipe the device executes, for round-tripping.

    This is the whole point of matching the device's schema: an action found by
    a policy here is a file the rig can run on a phone unchanged.
    """
    points, durations, easings, delays, spin = decode(
        np.asarray(vec, dtype=float), n_slots)
    return {"gestures": [{"points": points[i].tolist(),
                          "duration": float(durations[i]),
                          "easing_power": float(easings[i])}
                         for i in range(n_slots)],
            "delays": [float(d) for d in delays],
            "spin": {"enabled": bool(spin[0]),
                     "t_start": float(spin[1]),
                     "t_end": float(spin[2])}}
