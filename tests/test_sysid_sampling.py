import sys
from types import SimpleNamespace

import numpy as np

from opensk.fit import sysid
from opensk.sim.params import SkateParams


def test_cma_candidates_share_generation_subset_and_held_uses_combined(monkeypatch):
    class FakeES:
        def __init__(self, x0, *_args, **_kwargs):
            self.x0 = np.asarray(x0)
            self.generation = 0

        def stop(self):
            return self.generation >= 2

        def ask(self):
            return [self.x0 + self.generation * 0.01,
                    self.x0 + self.generation * 0.01 + 0.001]

        def tell(self, _xs, _losses):
            self.generation += 1

    monkeypatch.setitem(sys.modules, "cma", SimpleNamespace(
        CMAEvolutionStrategy=FakeES))
    train = sysid.Corpus(samples=[object()] * 6, targets=[[]] * 6)
    held = sysid.Corpus(samples=[object()] * 2, targets=[[]] * 2)
    calls = []

    def combined(params, corpus, *, subsample=None, rng=None,
                 sample_indices=None):
        calls.append((corpus, None if sample_indices is None else
                      tuple(sorted(map(int, sample_indices)))))
        return float(params.touch_gain)

    monkeypatch.setattr(sysid, "mean_combined", combined)
    monkeypatch.setattr(sysid, "mean_activity", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(sysid, "mean_gain", lambda *_a, **_k: 0.3)
    monkeypatch.setattr(sysid, "mean_iou", lambda *_a, **_k: 0.4)
    _, report = sysid.fit(train, held=held, subsample=3,
                          base=SkateParams(), verbose=False)

    # First four calls are the two candidates in each generation.
    assert calls[0][1] == calls[1][1]
    assert calls[2][1] == calls[3][1]
    assert calls[0][1] != calls[2][1]
    assert "held_combined" in report and "held_combined_inert" in report
    assert any(corpus is held and indices is None for corpus, indices in calls)
