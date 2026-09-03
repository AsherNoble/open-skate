"""A deliberately small world model, to check the pipeline end to end.

The plan's verification step: collect rollouts, train a model, and confirm it
predicts HELD-OUT SIM rollouts before anyone tries real-data fine-tuning. The
point is not to be a good world model. The point is that a model trained on
what this environment produces can predict what this environment produces —
if it cannot, something upstream is wrong and no amount of model architecture
will rescue it.

So the bar is set against a baseline that cannot be beaten by accident:

  * **persistence** — predict that the next frame equals the current one. At
    30 fps most of a frame is unchanged, so persistence is strong, and a model
    that merely blurs the input will lose to it.
  * **mean frame** — predict the training set's average frame. Beats nothing
    but a broken model, which is exactly why it is worth printing.

A model that does not beat persistence has learned nothing about the action,
however low its loss looks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scores:
    """Mean squared error per pixel, lower better."""
    model: float
    persistence: float
    mean_frame: float

    @property
    def beats_persistence(self) -> bool:
        return self.model < self.persistence

    def __str__(self) -> str:
        return (f"model {self.model:.5f}  persistence {self.persistence:.5f}  "
                f"mean-frame {self.mean_frame:.5f}  "
                f"{'BEATS' if self.beats_persistence else 'LOSES TO'} persistence")


def frame_pairs(shard, stride: int = 1):
    """(current frame, action, next frame) triples from a shard.

    Only from episodes marked valid: an unstable episode's frames are a record
    of the solver failing, and a model that learns them learns that.
    """
    frames = shard.frames_float()          # (B, F, H, W, 3)
    keep = np.asarray(shard.valid, dtype=bool)
    frames, actions = frames[keep], np.asarray(shard.actions)[keep]
    cur = frames[:, :-stride].reshape(-1, *frames.shape[2:])
    nxt = frames[:, stride:].reshape(-1, *frames.shape[2:])
    reps = frames.shape[1] - stride
    act = np.repeat(actions, reps, axis=0)
    return cur, act, nxt


def baselines(cur: np.ndarray, nxt: np.ndarray, train_mean: np.ndarray) -> tuple:
    """(persistence MSE, mean-frame MSE) on the same data the model sees."""
    return (float(np.mean((cur - nxt) ** 2)),
            float(np.mean((train_mean[None] - nxt) ** 2)))


def train_linear(cur, act, nxt, *, ridge: float = 1e-3):
    """Least-squares next-frame predictor on [frame, action, 1].

    Linear on purpose. It is the simplest thing that can use the action at all,
    it has a closed form so there is no training loop to get wrong, and if a
    linear map on the action already beats persistence then the pipeline
    carries real action-conditioned signal. A deep model that beat persistence
    would leave that ambiguous.
    """
    n = len(cur)
    x = np.concatenate([cur.reshape(n, -1), act.reshape(n, -1),
                        np.ones((n, 1))], axis=1).astype(np.float64)
    y = nxt.reshape(n, -1).astype(np.float64)
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    w = np.linalg.solve(gram, x.T @ y)
    return w


def predict(w, cur, act):
    n = len(cur)
    x = np.concatenate([cur.reshape(n, -1), act.reshape(n, -1),
                        np.ones((n, 1))], axis=1)
    return (x @ w).reshape(cur.shape)


def evaluate(train_shard, held_shard, *, ridge: float = 1e-3) -> Scores:
    """Train on one shard, score on another. Never on the same episodes.

    Frames within an episode are almost identical to each other, so a split
    that cuts across frames rather than episodes leaks the answer and every
    model looks excellent.
    """
    cur, act, nxt = frame_pairs(train_shard)
    hcur, hact, hnxt = frame_pairs(held_shard)
    w = train_linear(cur, act, nxt, ridge=ridge)
    pers, meanf = baselines(hcur, hnxt, cur.mean(axis=0))
    return Scores(model=float(np.mean((predict(w, hcur, hact) - hnxt) ** 2)),
                  persistence=pers, mean_frame=meanf)
