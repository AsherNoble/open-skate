"""The built deck must have True Skate's shape, not merely be built from it.

The profile tables are the INPUT. What renders, and what the fitting
silhouette is taken from, is the compiled mesh -- so that is what these
compare, vertex cloud against vertex cloud, in metres.

Half the file runs without the game bundle: the tables themselves, and the
properties of the compiled mesh that follow from them. Only the direct
comparison needs the user's own copy.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from opensk.sim.core import SkateSim
from opensk.sim.model import deck_profile as dp
from opensk.sim.params import SkateParams

IPA = Path.home() / "Projects/Robotics & hardware/TrueSkate.ipa"


def _frame(P):
    """(length, width, height) columns, centred, longest axis first.

    MuJoCo re-expresses a mesh in its own principal frame, so the axes are
    identified by extent rather than assumed. For this deck the rotation is a
    pure 90-degree axis swap, which a column permutation undoes exactly.
    """
    li, wi, hi = np.argsort(-np.ptp(P, axis=0))
    Q = np.c_[P[:, li], P[:, wi], P[:, hi]].astype(float)
    Q[:, 0] -= 0.5 * (Q[:, 0].max() + Q[:, 0].min())
    Q[:, 1] -= np.median(Q[:, 1])
    return Q


def _profile(P, stations, band=0.014):
    """(half-width, mid-surface height) at each station.

    Windows widen until they hold enough points: the two clouds are sampled
    differently, and a fixed bin narrower than the sim mesh's ring spacing
    aliases onto whichever neighbouring ring leaks in. Height is the MID
    surface -- half-way between the top and bottom skins -- because a median
    over the band mixes the two populations and jumps by the plate's whole
    thickness depending on how many of each land in the bin.
    """
    ax, ay = np.abs(P[:, 0]), np.abs(P[:, 1])
    w, z = [], []
    for t in stations:
        h = 0.011
        while (np.abs(ax - t) <= h).sum() < 8 and h <= 0.05:
            h += 0.004
        s = np.abs(ax - t) <= h
        w.append(ay[s].max() if s.sum() else np.nan)
        h = 0.011
        while ((np.abs(ax - t) <= h) & (ay < band)).sum() < 4 and h <= 0.05:
            h += 0.004
        sc = (np.abs(ax - t) <= h) & (ay < band)
        z.append(0.5 * (P[sc, 2].max() + P[sc, 2].min()) if sc.sum() >= 2
                 else np.nan)
    return np.array(w), np.array(z)


def _shell_vertices(model):
    gid = model.geom("vis_deck").id
    mid = model.geom_dataid[gid]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    return model.mesh_vert[a:a + n].astype(float)


# --- the tables, and what the mesh inherits from them ----------------------

def test_the_profile_tables_are_a_deck_and_not_a_plank():
    """A popsicle: parallel through the middle, tapering only near the tips."""
    assert dp.HALF_WIDTH[0] > 0.98, "the waist must be near the full width"
    mid = [w for t, w in zip(dp.STATIONS, dp.HALF_WIDTH) if t <= 0.7]
    assert max(mid) - min(mid) < 0.01, "the middle two thirds must be parallel"
    assert dp.HALF_WIDTH[-1] == 0.0, "the tip cap closes on the centreline"
    assert all(a >= b - 1e-9 for a, b in zip(dp.HALF_WIDTH[10:],
                                             dp.HALF_WIDTH[11:])), \
        "the taper must not reverse once it starts"


def test_the_kick_is_a_progressive_curve_not_a_hinge():
    """The old model was one straight ramp, which is what made it a banana."""
    slopes = [math.degrees(dp.pitch_rad(t)) for t in (0.72, 0.80, 0.88, 0.96)]
    assert all(b > a for a, b in zip(slopes, slopes[1:])), \
        f"kick slope must increase along the tail, got {slopes}"
    assert slopes[0] < 12.0 and slopes[-1] > 18.0


def test_the_compiled_mesh_is_the_size_the_parameters_ask_for():
    p = SkateParams()
    V = _frame(_shell_vertices(SkateSim(p).model))
    assert np.ptp(V[:, 0]) == pytest.approx(p.deck_length, abs=5e-4)
    assert np.ptp(V[:, 1]) == pytest.approx(p.deck_width, abs=5e-4)


def test_the_deck_is_flat_where_the_game_deck_is_flat():
    """Two thirds of the deck carries no rise at all. The old one kicked from
    60% of the half-length; this asserts the flat that replaced it."""
    p = SkateParams()
    hl = 0.5 * p.deck_length
    rises = [abs(dp.rise_frac(t)) * hl for t in np.linspace(0.0, 0.6, 13)]
    assert max(rises) < 0.002, f"the flat is not flat: {max(rises):.4f} m"


def test_the_collision_boxes_track_the_visual_shell():
    """Both are built from one `deck_station`, so they cannot drift apart --
    but a future edit could still move one and not the other.

    Each box centre is a CHORD midpoint, so it need not lie exactly on the
    curve; what must hold is that it stays inside the plate. Not asserted as
    "the chord is below the arc": the profile dips 1 mm before it kicks, so
    that inequality is false at the first nose box and would be a wrong test
    that happened to pass on a convex deck.
    """
    p = SkateParams()
    sim = SkateSim(p)
    half_t = 0.5 * p.deck_thickness
    seen = 0
    for gid in sorted(sim._deck_gids):
        name = sim.model.geom(gid).name
        pos = sim.model.geom_pos[gid]
        _, z = dp.station(pos[0] / (0.5 * p.deck_length), 0.5 * p.deck_length)
        assert abs(pos[2] - z) < half_t, (
            f"{name} is {abs(pos[2] - z) * 1000:.1f} mm off the centreline")
        seen += 1
    assert seen == 7, f"expected 7 deck collision boxes, found {seen}"


# --- against the game's own geometry --------------------------------------

@pytest.mark.skipif(not IPA.exists(),
                    reason="needs the user's own copy of the game bundle")
class TestAgainstTheGame:

    @staticmethod
    def _clouds():
        from opensk.assets.tsmesh import read_from_ipa
        p = SkateParams()
        S = _frame(_shell_vertices(SkateSim(p).model))
        rim = _frame(np.vstack([read_from_ipa(IPA, f).positions.astype(float)
                                for f in ("edge_top.bin", "edge_bottom.bin")]))
        # The perimeter band carries the outline but has no centreline
        # vertices, so the side profile comes from the two surfaces instead.
        skin = _frame(np.vstack([read_from_ipa(IPA, f).positions.astype(float)
                                 for f in ("grip_tape.bin", "deck_bottom.bin")]))
        u = np.ptp(S[:, 0]) / np.ptp(rim[:, 0])
        rim *= u
        skin *= u
        for Q in (S, skin):
            b = (np.abs(Q[:, 0]) < 0.10) & (np.abs(Q[:, 1]) < 0.014)
            Q[:, 2] -= 0.5 * (Q[b, 2].max() + Q[b, 2].min())
        return p, S, rim, skin

    def test_the_plan_outline_matches(self):
        p, S, rim, _ = self._clouds()
        st = np.linspace(0.01, 0.5 * p.deck_length - 0.01, 20)
        gw, _ = _profile(rim, st)
        sw, _ = _profile(S, st)
        d = (sw - gw) * 1000.0
        assert np.isfinite(d).all()
        assert np.sqrt((d ** 2).mean()) < 2.5, f"outline rms {d} mm"
        assert np.abs(d).max() < 6.0, f"outline max {np.abs(d).max():.2f} mm"

    def test_the_side_profile_matches(self):
        p, S, _, skin = self._clouds()
        st = np.linspace(0.01, 0.5 * p.deck_length - 0.01, 20)
        _, gz = _profile(skin, st)
        _, sz = _profile(S, st)
        d = (sz - gz) * 1000.0
        assert np.isfinite(d).all()
        assert np.sqrt((d ** 2).mean()) < 2.0, f"profile rms {d} mm"
        assert np.abs(d).max() < 5.0, f"profile max {np.abs(d).max():.2f} mm"

    def test_the_tip_rise_matches(self):
        """The number that governs pop: how far the tail must travel to hit
        the ground. The old deck's tips sat 52 mm up against the game's 35."""
        p, S, _, skin = self._clouds()
        st = np.array([0.5 * p.deck_length - 0.01])
        assert _profile(S, st)[1][0] == pytest.approx(
            _profile(skin, st)[1][0], abs=0.004)
