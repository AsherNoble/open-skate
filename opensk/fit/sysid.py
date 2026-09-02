"""System identification: fit the physics to the real game.

CMA-ES over the parameters in `FIT_SPEC`, scoring candidates by mean silhouette
overlap against captured gameplay (`fit.objective`). Evolutionary rather than
gradient-based because the objective runs a contact simulation and a rasteriser,
neither of which is differentiable, and because the rig's existing pipeline
already reaches for CMA-ES on the same shape of problem.

Two things this deliberately does NOT do:

  * It does not fit the camera. The camera was calibrated separately against
    resting-board frames, and letting it move here would let the optimiser buy
    overlap by re-aiming the lens instead of correcting the physics.
  * It does not fit geometry we can measure with a ruler. Deck length, width
    and thickness are excluded from FIT_SPEC for the same reason.

Held-out samples are scored but never optimised against. Fitting until the
training set matches is easy and says nothing.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from ..pose.frames import Sample, iter_samples
from ..pose.render import SceneRenderer
from ..sim.core import SkateSim
from ..sim.params import FIT_KEYS, FIT_SPEC, SkateParams, from_vector, to_vector
from ..sim.touch import ON_DECK, TouchModel
from .objective import real_masks, score_sample

# Parameters the camera calibration owns. Excluded from the search.
CAMERA_KEYS = ("cam_fov_deg", "cam_distance", "cam_pitch_deg", "cam_lead_m",
               "cam_follow_tau")

PHYSICS_KEYS = tuple(k for k in FIT_KEYS if k not in CAMERA_KEYS)


@dataclass
class Corpus:
    """Samples with their real silhouettes pre-segmented.

    Segmentation is the expensive part and does not depend on the parameters,
    so it is done once. Without this, every CMA-ES evaluation would re-decode
    and re-threshold the same PNGs.
    """
    samples: list[Sample]
    targets: list[list]
    height: int = 224
    width: int = 103

    def __len__(self) -> int:
        return len(self.samples)

    def split(self, frac: float = 0.25, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.samples))
        n = int(round(frac * len(idx)))
        held, train = idx[:n], idx[n:]
        pick = lambda ii: Corpus([self.samples[i] for i in ii],
                                 [self.targets[i] for i in ii],
                                 self.height, self.width)
        return pick(train), pick(held)


CACHE_DIR = pathlib.Path(__file__).resolve().parents[2] / "cache" / "masks"


def _cache_path(sample: Sample, h: int, w: int) -> pathlib.Path:
    key = f"{sample.path.parent.name}__{sample.path.name}__{h}x{w}.npz"
    return CACHE_DIR / key


def _load_targets(sample: Sample, h: int, w: int, use_cache: bool):
    """Segmented masks for a sample, from disk cache when available.

    Segmentation decodes and thresholds a full 828x1792 PNG per frame, which
    dominates start-up: roughly 1300 images for a 120-sample corpus. The masks
    depend only on the frames and the render size, never on the parameters
    being fitted, so they are cached as packed bits.
    """
    if not use_cache:
        return real_masks(sample, h, w)[0]
    cp = _cache_path(sample, h, w)
    if cp.exists():
        try:
            z = np.load(cp)
            present = z["present"]
            packed = z["packed"]
            out, k = [], 0
            for ok in present:
                if not ok:
                    out.append(None)
                else:
                    out.append(np.unpackbits(packed[k])[: h * w].reshape(h, w).astype(bool))
                    k += 1
            return out
        except Exception:
            pass  # corrupt or stale cache entry: just recompute
    tg = real_masks(sample, h, w)[0]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    present = np.array([m is not None for m in tg])
    packed = np.array([np.packbits(m.reshape(-1)) for m in tg if m is not None]) \
        if present.any() else np.zeros((0, 1), dtype=np.uint8)
    np.savez_compressed(cp, present=present, packed=packed)
    return tg


def build_corpus(limit: int | None = None, *, height: int = 224,
                 min_frames: int = 4, require_motion: bool = True,
                 use_cache: bool = True, verbose: bool = True) -> Corpus:
    """Segment usable samples once, up front.

    Three filters, each measured rather than guessed:
      * the gesture must start on the deck -- 71.6% do, and the rest are flicks
        at empty screen that cannot constrain anything;
      * at least `min_frames` frames must survive the gameplay and segmentation
        filters (26% of frames are menu or unsegmentable);
      * the real board must actually move (`require_motion`), or the sample
        rewards a simulation that does nothing.
    """
    probe = SkateSim()
    probe.reset(seed=0)
    probe.step(200)
    tm = TouchModel(probe)
    renderer = SceneRenderer(probe, height=height)
    w, h = renderer.width, renderer.height

    samples: list[Sample] = []
    targets: list[list] = []
    seen = 0
    for s in iter_samples():
        if limit is not None and len(samples) >= limit:
            break
        seen += 1
        if tm.cast(float(s.waypoints[0][0]), float(s.waypoints[0][1]))[0] != ON_DECK:
            continue
        tg = _load_targets(s, h, w, use_cache)
        good = [m for m in tg if m is not None]
        if len(good) < min_frames:
            continue
        if require_motion:
            first = good[0]
            worst = min(np.count_nonzero(first & m) / max(np.count_nonzero(first | m), 1)
                        for m in good[1:])
            if worst > 0.9:      # silhouette never changed: no dynamics here
                continue
        samples.append(s)
        targets.append(tg)
    if verbose:
        print(f"corpus: {len(samples)} usable of {seen} scanned")
    return Corpus(samples, targets, h, w)


def mean_gain(params: SkateParams, corpus: Corpus, *,
              subsample: int | None = None, rng=None) -> float:
    """Mean overlap earned ABOVE a stationary board. Inertness scores 0.

    This is the objective to optimise. Optimising raw overlap failed in a
    specific and informative way: it reached 0.6604 on train but 0.6424 on
    held-out against 0.6444 for an inert board, having driven touch_gain from
    600 to 76. The optimiser found that the cheapest way to look like the real
    frames is to stop moving, because raw overlap is dominated by the board
    simply being roughly where it started.
    """
    sim = SkateSim(params)
    renderer = SceneRenderer(sim, height=corpus.height)
    idx = range(len(corpus.samples))
    if subsample is not None and subsample < len(corpus.samples):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(corpus.samples), subsample, replace=False)
    vals = []
    for i in idx:
        sc = score_sample(corpus.samples[i], params, targets=corpus.targets[i],
                          sim=sim, renderer=renderer)
        if sc.n_scored:
            vals.append(sc.gain)
    return float(np.mean(vals)) if vals else 0.0


def mean_iou(params: SkateParams, corpus: Corpus, *,
             subsample: int | None = None, rng=None) -> float:
    """Mean per-sample overlap. Samples that score no frames are skipped.

    Render size comes from the corpus, never from a caller argument: the cached
    masks were segmented at one resolution and a mismatch is a silent shape
    error at best.

    `subsample` scores a random subset per call, making the objective
    stochastic. CMA-ES tolerates that well, and it is the difference between a
    fit that takes an hour and one that takes a day.
    """
    sim = SkateSim(params)
    renderer = SceneRenderer(sim, height=corpus.height)
    idx = range(len(corpus.samples))
    if subsample is not None and subsample < len(corpus.samples):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(corpus.samples), subsample, replace=False)
    vals = []
    for i in idx:
        sc = score_sample(corpus.samples[i], params, targets=corpus.targets[i],
                          sim=sim, renderer=renderer)
        if sc.n_scored:
            vals.append(sc.iou)
    return float(np.mean(vals)) if vals else 0.0


def fit(corpus: Corpus, *, evals: int = 300, seed: int = 0,
        base: SkateParams | None = None, log_path: str | None = None,
        held: Corpus | None = None, subsample: int | None = None,
        verbose: bool = True):
    """CMA-ES over PHYSICS_KEYS. Returns (best_params, report)."""
    import cma

    base = base or SkateParams()
    lo = np.array([FIT_SPEC[k][0] for k in PHYSICS_KEYS])
    hi = np.array([FIT_SPEC[k][1] for k in PHYSICS_KEYS])
    x0 = np.array([getattr(base, k) for k in PHYSICS_KEYS])

    def to_params(x) -> SkateParams:
        vals = {k: float(np.clip(v, FIT_SPEC[k][0], FIT_SPEC[k][1]))
                for k, v in zip(PHYSICS_KEYS, x)}
        return base.replace(**vals)

    log = open(log_path, "w") if log_path else None
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(
        list(x0), 1.0,
        {"bounds": [list(lo), list(hi)], "seed": seed + 1, "maxfevals": evals,
         "verbose": -9, "CMA_stds": list((hi - lo) / 5.0)})

    # One generator for the whole run, so each generation scores a different
    # subset and the search cannot overfit one lucky draw.
    obj_rng = np.random.default_rng(seed + 17)
    best_iou, best_params, n = -1.0, base, 0
    while not es.stop():
        xs = es.ask()
        losses = []
        for x in xs:
            p = to_params(x)
            iou = mean_gain(p, corpus, subsample=subsample, rng=obj_rng)
            losses.append(-iou)
            n += 1
            if iou > best_iou:
                best_iou, best_params = iou, p
            if log:
                log.write(json.dumps({"type": "eval", "eval": n, "iou": iou,
                                      "params": dict(zip(PHYSICS_KEYS,
                                                         map(float, x)))}) + "\n")
                log.flush()
        es.tell(xs, losses)
        if verbose:
            print(f"  evals {n:4d}  best train gain {best_iou:+.4f}"
                  f"  ({time.time() - t0:.0f}s)")

    report = {"train_gain": best_iou, "evals": n,
              "seconds": time.time() - t0,
              "params": {k: getattr(best_params, k) for k in PHYSICS_KEYS}}
    if held is not None:
        # Gain on the full held-out set. Positive means the fitted physics
        # reproduces real motion better than not moving at all; zero or
        # negative means it does not, whatever the raw overlap says.
        report["held_gain"] = mean_gain(best_params, held)
        report["held_iou"] = mean_iou(best_params, held)
        report["held_gain_inert"] = mean_gain(
            best_params.replace(touch_gain=1e-3, touch_force_max=1e-3), held)
    if log:
        log.write(json.dumps({"type": "result", **report}) + "\n")
        log.close()
    return best_params, report
