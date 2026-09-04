"""The deck's shape, measured from True Skate's own geometry.

Why this file exists
--------------------
The deck used to be described by four hand-chosen scalars -- `flat_fraction`,
`kick_angle_deg`, `deck_taper_power`, `deck_tip_width_frac` -- fitted against
the ONE projection the frame corpus contains, the chase camera's silhouette.
Rendered from any other angle the result is a banana: a straight ramp starting
6 cm too early at each end and finishing 17 mm too high, with a plan view that
steps rather than curves.

The game ships the geometry, so it is copied rather than inferred. These are
DERIVED NUMBERS -- two normalised profiles and three ratios, extracted by
`opensk/assets/tsmesh.py` from the app bundle the user owns. No mesh, texture
or other asset from the game is stored in this repository.

Provenance
----------
`res/edge_top.bin` and `res/edge_bottom.bin` (the perimeter band, 1664
vertices each) give the plan outline; `res/grip_tape.bin` (the top surface,
840 vertices) gives the centreline. All are `OMSH`, which the decoder
supports; `deck.bin` is `SKDE` and is still refused.

Scale, and why it is believable
-------------------------------
One game unit is **39.85 mm**. The anchor is the WHEEL, not the deck -- using
the deck to set the scale and then reporting the deck's size would be
circular. `res/wheel.bin` is a surface of revolution 1.2546 u across; taken as
a 50 mm street wheel that fixes the unit, and everything else then lands on a
standard part without being asked to:

    deck width      5.2496 u -> 209.2 mm = 8.24 in   (a standard 8.25 deck)
    deck length    20.2953 u -> 808.8 mm = 31.84 in  (its standard pairing)
    deck thickness  0.3014 u ->  12.0 mm             (a 7-ply deck)
    oldschool wheel 1.4447 u ->  57.6 mm             (a cruiser wheel)

Four quantities, one assumed number. That is what makes 39.85 mm evidence
rather than a restatement.

A fifth, from inside our own model: `axle_halfwidth = 0.105` was measured with
a ruler and implies a 0.21 m deck. The game's 0.2092 m agrees; the previously
fitted 0.196 m did not.

What the tables are
-------------------
`STATIONS` runs 0 (deck centre) to 1 (tip), unevenly spaced because the tip
carries all the curvature. The deck is treated as symmetric: real decks have a
slightly longer nose, but the truck positions are in `SKTR`, which the decoder
still refuses, so there is no way to place the origin correctly and a
fabricated asymmetry would be worse than none.

  `HALF_WIDTH[i]`  half-width at that station / the deck's maximum half-width
  `RISE[i]`        centreline height above the flat / the deck's HALF-length

Both are dimensionless, so `SkateParams.deck_length` and `deck_width` still
set the size and this file only sets the shape.

Two honest caveats:
  * the top surface's vertices stop at t = 0.9834, where the deck rolls over
    its tip. `RISE` beyond that is a linear extrapolation of the local slope,
    covering the last 1.7% of the length and adding about 1.4 mm of rise;
  * `HALF_WIDTH[-1]` is set to exactly 0. The measured outline reaches 0.001
    of the maximum at t = 1, i.e. the cap closes on the centreline; zero is
    the closure, not a guess.
"""
from __future__ import annotations

import bisect
import math

# t along the deck, 0 = middle, 1 = tip.
STATIONS: tuple[float, ...] = (
    0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.550, 0.600,
    0.650, 0.700, 0.750, 0.800, 0.850, 0.875, 0.900, 0.925,
    0.950, 0.965, 0.980, 0.990, 0.995, 1.000,
)

# Half-width at each station, as a fraction of the deck's maximum half-width.
# Flat to within 0.8% out to t = 0.7 -- the taper the old model started at
# t = 0 does not begin until three quarters of the way to the tip.
HALF_WIDTH: tuple[float, ...] = (
    0.9921, 0.9921, 0.9921, 0.9921, 0.9925, 0.9953, 0.9984, 1.0000,
    0.9998, 0.9970, 0.9676, 0.9028, 0.8109, 0.7814, 0.7283, 0.6369,
    0.5687, 0.4961, 0.4184, 0.3383, 0.2575, 0.0000,
)

# Centreline rise, as a fraction of the deck's HALF-length. Note the 1.2 mm
# dip at t = 0.65: the deck bends slightly downward before it kicks up, which
# no straight-ramp model can produce. The kick is progressive, reaching about
# 21 degrees around t = 0.93 -- it is a curve, not a hinge.
RISE: tuple[float, ...] = (
    0.00000, 0.00003, 0.00006, 0.00012, 0.00024, -0.00001, -0.00069, -0.00215,
    -0.00308, -0.00154, 0.00555, 0.01571, 0.03270, 0.04098, 0.04866, 0.05898,
    0.06821, 0.07596, 0.08187, 0.08439, 0.08629, 0.08819,
)

# length / width, from the raw extents of the perimeter band.
ASPECT = 3.8661
# thickness / length, from the top and bottom surfaces at the centreline,
# constant to 4 decimal places across the whole flat.
THICKNESS_FRAC = 0.014851
# Concave depth / width: the top surface at mid-deck sits 5.6 mm lower at the
# centreline than at the rails. Visual only -- contact is boxes.
CONCAVE_FRAC = 0.0268

# Metres per game unit, from the wheel anchor above. Only used to derive the
# defaults in `SkateParams`; nothing at runtime reads it.
UNIT_M = 0.039852

assert len(STATIONS) == len(HALF_WIDTH) == len(RISE)


def _interp(table: tuple[float, ...], t: float) -> float:
    """Linear interpolation on `STATIONS`, clamped at both ends."""
    t = min(1.0, abs(t))
    i = bisect.bisect_left(STATIONS, t)
    if i == 0:
        return table[0]
    t0, t1 = STATIONS[i - 1], STATIONS[i]
    f = (t - t0) / (t1 - t0)
    return table[i - 1] + f * (table[i] - table[i - 1])


def half_width_frac(t: float) -> float:
    """Half-width at station `t`, as a fraction of the maximum half-width."""
    return _interp(HALF_WIDTH, t)


def rise_frac(t: float) -> float:
    """Centreline rise at `t`, as a fraction of the deck's half-length."""
    return _interp(RISE, t)


def station(t: float, half_length: float) -> tuple[float, float]:
    """(x, z) on the deck's centreline at signed station `t` in [-1, 1]."""
    return t * half_length, rise_frac(t) * half_length


def pitch_rad(t: float, eps: float = 0.004) -> float:
    """Local slope of the centreline at signed station `t`, in radians.

    The profile is a table, so this is differenced rather than
    differentiated, and one-sided at the ends so the clamp in `_interp`
    cannot flatten the tip. `rise_frac` folds to |t|, so differencing the
    SIGNED stations is what makes the tail's slope come out negative.
    """
    a, b = max(-1.0, t - eps), min(1.0, t + eps)
    if b == a:
        a, b = t - 2 * eps, t
    return math.atan2(rise_frac(b) - rise_frac(a), b - a)
