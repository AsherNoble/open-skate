"""Fit the game camera from the board's silhouette in real frames.

The camera is the first thing that must be right. It decides where a screen
touch lands on the deck, so until it matches True Skate's, every force the
touch model applies is applied in the wrong place and the physics fit would
silently absorb the error.

THE CAMERA IS PER-CAPTURE-SESSION, NOT A CONSTANT. True Skate lets the player
move the camera, so a calibration fitted on one corpus does not describe
another. Measured: the real board is 0.2857 of frame height in the
self-labelled flick corpus but 0.1154 in the trick captures -- 40% the size,
and at a different height on screen (cy 0.5952 against 0.4892). Fitting
physics against a corpus with the wrong camera makes silhouette overlap a
measure of that mismatch rather than of dynamics, and no parameter search can
recover it. Re-calibrate for every corpus before fitting against it.

Calibration uses frames with the board at rest at the reset anchor, where its
pose is known by construction. Four silhouette features are matched:

    length, width  -- apparent size, which pins distance against deck geometry
    cy             -- vertical framing, which pins how far the camera leads
    taper          -- perspective foreshortening, which separates field of
                      view from distance (the dolly-zoom degeneracy)

Only board geometry we can measure with a ruler is held fixed. Any difference
between True Skate's deck size and ours is therefore absorbed into the fitted
distance -- which is the correct place for it, since what has to be right is
the mapping from screen point to deck point, not the metric distance itself.
"""
from __future__ import annotations

import numpy as np

from ..sim.core import SkateSim
from ..sim.params import SkateParams
from .render import SceneRenderer
from .segment import FEATURE_NAMES, mask_features

CAM_KEYS = ("cam_fov_deg", "cam_distance", "cam_pitch_deg", "cam_lead_m")


def sim_features(params: SkateParams, height: int = 448) -> np.ndarray | None:
    sim = SkateSim(params)
    sim.reset(seed=0)
    sim.step(400)  # let it settle exactly as the real board does at the anchor
    return mask_features(SceneRenderer(sim, height=height).board_pixels())


def objective(x: np.ndarray, target: np.ndarray, scale: np.ndarray,
              base: SkateParams) -> float:
    """Scaled residual between simulated and real silhouette features.

    Residuals are divided by each feature's own spread across the real corpus,
    so `cy` (MAD 0.0008) is not drowned out by `taper` (MAD 0.044).
    """
    p = base.replace(**dict(zip(CAM_KEYS, map(float, x))))
    f = sim_features(p)
    if f is None:
        return 1e3
    return float(np.sqrt(np.mean(((f - target) / scale) ** 2)))


def fit(target: np.ndarray, scale: np.ndarray, base: SkateParams | None = None,
        *, seed: int = 0, evals: int = 400, verbose: bool = True):
    """CMA-ES over the four camera parameters. Returns (params, report)."""
    import cma

    base = base or SkateParams()
    from ..sim.params import FIT_SPEC
    lo = np.array([FIT_SPEC[k][0] for k in CAM_KEYS])
    hi = np.array([FIT_SPEC[k][1] for k in CAM_KEYS])
    x0 = np.array([getattr(base, k) for k in CAM_KEYS])

    es = cma.CMAEvolutionStrategy(
        list(x0), 0.25,
        {"bounds": [list(lo), list(hi)], "seed": seed, "maxfevals": evals,
         "verbose": -9, "CMA_stds": list((hi - lo) / 6.0)})
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [objective(np.array(x), target, scale, base) for x in xs])
    best = np.array(es.result.xbest)
    fitted = base.replace(**dict(zip(CAM_KEYS, map(float, best))))
    report = {"loss": float(es.result.fbest),
              "params": dict(zip(CAM_KEYS, best.tolist())),
              "sim_features": sim_features(fitted).tolist(),
              "target_features": target.tolist()}
    if verbose:
        print("loss %.4f" % report["loss"])
        for k, v in report["params"].items():
            print("  %-16s %8.3f" % (k, v))
        print("  %-10s %9s %9s" % ("feature", "sim", "real"))
        for n, a, b in zip(FEATURE_NAMES, report["sim_features"], target):
            print("  %-10s %9.4f %9.4f" % (n, a, b))
    return fitted, report
