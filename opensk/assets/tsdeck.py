"""Measure the game deck's shape from its own vertices.

This needs NO topology, which is why it works: the index buffer's meaning is
still unknown (see `tsmesh.triangles`), but a point cloud is enough to recover
a plan-view outline and a side profile, and those are what the silhouette --
the thing Open Skate is actually fitted against -- is made of.

Scale is recovered rather than assumed, and it checks out from two independent
directions: taking the deck to be a standard 8.0 in wide gives a length of
31.84 in, and taking the length to be the already-fitted 0.813 m gives a width
of 8.04 in. Two anchors agreeing to 0.5% is what makes the unit believable --
1 game unit is ~40 mm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tsmesh import TSMesh, read_from_ipa

INCH = 0.0254
STANDARD_DECK_WIDTH_M = 8.0 * INCH


@dataclass(frozen=True)
class DeckShape:
    """The game deck's geometry, in metres."""

    length: float
    width: float
    rise: float                 # kicktail height above the flat
    flat_fraction: float        # fraction of the length that is flat
    kick_angle_deg: float
    tip_width_frac: float       # tip half-width / waist half-width
    unit_m: float               # metres per game unit
    profile: np.ndarray         # (n, 3): t along deck, half-width, height


def _axes(p: np.ndarray) -> tuple[int, int, int]:
    """Length, width, height axes, by extent. The mesh has no stated frame."""
    return tuple(np.argsort(-(p.max(0) - p.min(0))))


def measure(mesh: TSMesh, *, bins: int = 48) -> DeckShape:
    p = mesh.positions.astype(float)
    li, wi, hi = _axes(p)
    ext = p.max(0) - p.min(0)
    unit = STANDARD_DECK_WIDTH_M / ext[wi]

    t = (p[:, li] - p[:, li].min()) / ext[li]
    half = np.abs(p[:, wi] - np.median(p[:, wi]))
    z = p[:, hi] - p[:, hi].min()

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        s = (t >= lo) & (t < hi_)
        if s.sum() >= 3:
            rows.append((0.5 * (lo + hi_), half[s].max(), z[s].max()))
    prof = np.array(rows)

    waist = np.median(prof[(prof[:, 0] > 0.3) & (prof[:, 0] < 0.7), 1])
    # Flat = where the deck is not yet rising. Use the height profile, not the
    # width one: the kick is a bend, the taper is a separate plan-view feature.
    floor = np.median(prof[(prof[:, 0] > 0.3) & (prof[:, 0] < 0.7), 2])
    rising = prof[:, 2] > floor + 0.15 * (prof[:, 2].max() - floor)
    flat_frac = 1.0 - rising.mean()

    # Kick angle from a line through the rising nose section.
    nose = prof[rising & (prof[:, 0] < 0.5)]
    if len(nose) >= 2:
        slope = np.polyfit(nose[:, 0] * ext[li], nose[:, 2], 1)[0]
        kick = float(np.degrees(np.arctan(abs(slope))))
    else:
        kick = float("nan")

    return DeckShape(
        length=float(ext[li] * unit),
        width=float(ext[wi] * unit),
        rise=float((prof[:, 2].max() - floor) * unit),
        flat_fraction=float(flat_frac),
        kick_angle_deg=kick,
        tip_width_frac=float(prof[:, 1].min() / waist),
        unit_m=float(unit),
        profile=prof,
    )


def from_ipa(ipa: Path, name: str = "deck_bottom.bin") -> DeckShape:
    return measure(read_from_ipa(ipa, name))
