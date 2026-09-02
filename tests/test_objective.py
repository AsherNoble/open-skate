"""The fitting objective must actually respond to physics.

An objective that scores every parameter set alike is worse than useless: CMA-ES
would wander and report convergence. These check it runs end to end on real data
and that it discriminates.
"""
import random

import numpy as np
import pytest

from opensk.fit.objective import real_masks, score_sample
from opensk.pose.frames import CORPUS, iter_samples
from opensk.sim.core import SkateSim
from opensk.sim.params import SkateParams
from opensk.sim.touch import ON_DECK, TouchModel

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")


@pytest.fixture(scope="module")
def usable():
    """Samples whose gesture starts on the deck -- the only ones that inform."""
    sim = SkateSim()
    sim.reset(seed=0)
    sim.step(200)
    tm = TouchModel(sim)
    out = [s for s in iter_samples()
           if tm.cast(float(s.waypoints[0][0]), float(s.waypoints[0][1]))[0] == ON_DECK]
    assert len(out) > 500, f"only {len(out)} usable samples"
    return out


def test_scores_real_samples(usable):
    random.seed(0)
    scored = [score_sample(s) for s in random.sample(usable, 6)]
    assert any(s.n_scored > 0 for s in scored), "no frame survived filtering"
    for s in scored:
        assert 0.0 <= s.iou <= 1.0
        assert s.n_scored <= s.n_frames


def _mean_iou(samples, params):
    vals = [score_sample(s, params) for s in samples]
    vals = [v.iou for v in vals if v.n_scored > 0]
    return float(np.mean(vals)) if vals else 0.0


def test_objective_is_not_flat(usable):
    """Different physics must score differently, or sysid cannot work.

    Deliberately does NOT assert that the current defaults beat an inert
    board. As of the first measurement they do not (0.456 against 0.511):
    the unfitted touch forces are far too strong, and a board that barely
    moves tracks the real frames better. That is a finding about the
    defaults, not a property to encode as a test.
    """
    # 20 samples, not 8: with 8 the draw is noisy enough to show a 0.002
    # difference where 40 samples show 0.102, which failed the threshold for
    # sampling reasons rather than because the objective had gone flat.
    random.seed(1)
    sub = random.sample(usable, 20)
    p = SkateParams()
    inert = p.replace(touch_gain=1e-3, touch_force_max=1e-3)
    assert abs(_mean_iou(sub, p) - _mean_iou(sub, inert)) > 0.01


def test_inert_board_baseline_is_beatable_in_principle(usable):
    """The real board moves, so an inert sim cannot be the best possible fit.

    97% of usable samples show the real silhouette changing over the sample
    (median self-IoU 0.527). The inert score is therefore a ceiling on "never
    move", around 0.51 -- correct physics must beat it. This pins the target
    that Phase 3 has to clear.
    """
    random.seed(5)
    sub = random.sample(usable, 10)
    moved = 0
    counted = 0
    for s in sub:
        targets, _ = real_masks(s, 224, 103)
        ms = [m for m in targets if m is not None]
        if len(ms) < 3:
            continue
        counted += 1
        first = ms[0]
        worst = min(np.count_nonzero(first & m) / max(np.count_nonzero(first | m), 1)
                    for m in ms[1:])
        if worst < 0.9:
            moved += 1
    assert counted >= 3
    assert moved / counted > 0.7, "real board barely moves; objective would reward inertness"
