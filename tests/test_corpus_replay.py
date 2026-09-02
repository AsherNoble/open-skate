"""Replaying the real corpus must actually manipulate the board.

These tests tie the sim to the data it will be fitted against. They caught two
wrong assumptions that no self-contained test could have:

  * The capture does NOT push before a gesture. `collect_self_labeled_traces.py`
    imports only `curved_drag` and `reset_position`, and resets every gesture
    (`--reset-every` defaults to 1). Replaying with a push carried the board out
    from under the gesture and dropped median finger-on-deck time to zero.
  * `sample_flick` spreads start points near-uniformly over the safe screen
    bounds, deliberately, so the corpus could teach a frame->gesture model to
    read touch traces anywhere. A large minority of gestures therefore never
    touch the board and carry no information about action -> Delta-environment.

Skipped when the corpus is absent so Open Skate stays testable standalone.
"""
import random

import numpy as np
import pytest

from opensk.pose.frames import CORPUS, iter_samples
from opensk.sim.core import SkateSim
from opensk.sim.touch import ON_DECK, TouchModel

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")


@pytest.fixture(scope="module")
def samples():
    s = list(iter_samples())
    assert s, "corpus present but no samples loaded"
    return s


def _at_anchor():
    """The board as the capture had it: reset, settled, never pushed."""
    sim = SkateSim()
    sim.reset(seed=0)
    sim.step(200)
    return sim


def test_most_gestures_start_on_the_deck(samples):
    """Sizes the usable subset -- the rest cannot constrain the physics.

    If this drops sharply, the camera calibration has moved and the board is
    no longer where the real gestures were aimed.
    """
    tm = TouchModel(_at_anchor())
    on = sum(1 for s in samples
             if tm.cast(float(s.waypoints[0][0]), float(s.waypoints[0][1]))[0] == ON_DECK)
    frac = on / len(samples)
    assert frac > 0.55, f"only {frac:.1%} of real gestures start on the deck"


def test_replaying_real_gestures_moves_the_board(samples):
    """A gesture that lands on the deck must do something to it."""
    random.seed(0)
    moved, engaged = [], []
    for s in random.sample(samples, 20):
        sim = _at_anchor()
        start = sim.state().pos.copy()
        tm = TouchModel(sim)
        counts = [0, 0]
        inner = tm._apply_finger

        def spy(f, t, dt, inner=inner, counts=counts):
            inner(f, t, dt)
            counts[1] += 1
            if f.kind == ON_DECK:
                counts[0] += 1

        tm._apply_finger = spy
        traj = tm.run(s.recipe(), push=False, settle=0.5)
        moved.append(float(np.linalg.norm(traj[-1].pos - start)))
        if counts[1]:
            engaged.append(counts[0] / counts[1])

    assert np.median(engaged) > 0.5, (
        f"finger on the deck for only {np.median(engaged):.0%} of substeps -- "
        "is the replay pushing when the capture did not?")
    assert (np.array(moved) > 0.02).mean() > 0.6, "gestures barely move the board"
