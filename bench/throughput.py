"""Episodes per hour, physics only, at a range of batch sizes.

The number this exists to produce is the one the whole project rests on. The
rig -- two physical iPhones driven at 1x real time -- evaluates about
**15,000 gestures per hour**. Open Skate is only worth building if a batched
GPU rollout clears that by a wide margin.

It does not on CPU. Measured on an M2 (see `results/THROUGHPUT.md`), batching
buys nothing at all, because `vmap` on CPU is B times the work done serially:

    batch 1 -> 12,339/h     batch 16 -> 13,304/h     batch 64 -> 13,348/h

That is a negative result, and it is not a refutation: single-scene MJX is
documented as roughly 10x slower than MuJoCo C, and the entire thesis is that
thousands of scenes run *at once*. This module is written to be run on a GPU,
where that claim can actually be tested.

Physics only. Rendering is measured separately, because it plausibly dominates
and the two numbers must not be confused with each other.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Result:
    batch: int
    compile_s: float
    run_s: float
    episodes_per_hour: float
    env_steps_per_s: float
    steps_per_episode: int


def _batched_gestures(batch: int, n_slots: int, seed: int = 0):
    """A batch of *different* gestures, sampled around the flick regime.

    Different gestures rather than one repeated, because identical inputs can
    let a compiler or a branch predictor flatter the measurement, and because a
    training batch is never uniform anyway.
    """
    import numpy as np

    from opensk.mjx.rollout import gesture_arrays

    rng = np.random.default_rng(seed)
    pts, segs, t0s = [], [], []
    for _ in range(batch):
        base = np.array([[0.50, 0.62], [0.50, 0.66], [0.50, 0.70]])
        recipe = {"gestures": [{"points": (base + rng.normal(0, 0.05, base.shape)).tolist(),
                                "duration": float(rng.uniform(0.08, 0.30)),
                                "easing_power": float(rng.uniform(0.6, 2.6))}],
                  "delays": []}
        p, s, t = gesture_arrays(recipe, n_slots=n_slots)
        pts.append(p); segs.append(s); t0s.append(t)
    return np.stack(pts), np.stack(segs), np.stack(t0s)


def benchmark(batch: int, *, n_steps: int | None = None, n_slots: int = 2,
              settle_steps: int = 50, seed: int = 0) -> Result:
    """Time one batched episode. Compile time is reported, never amortised in.

    Compile is excluded from the throughput figure because a training run pays
    it once; it is reported separately because it grew 4.3 s -> 24.6 s from
    B=1 to B=64 on CPU, and at B=16384 that could become the binding cost.
    """
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from opensk.mjx.parity import make_mjx
    from opensk.mjx.rollout import episode_length, rollout
    from opensk.sim.params import SkateParams

    params = SkateParams()
    mx, d0, cpu = make_mjx(params)
    step = jax.jit(lambda dd: mjx.step(mx, dd))
    for _ in range(settle_steps):        # let the board come to rest first
        d0 = step(d0)

    n = n_steps or episode_length(params)
    pts, segs, t0s = _batched_gestures(batch, n_slots, seed)
    pts, segs, t0s = jnp.asarray(pts), jnp.asarray(segs), jnp.asarray(t0s)

    def one(P, S, T):
        return rollout(mx, cpu.model, params, cpu.deck_bid,
                       sorted(cpu._deck_gids), d0, P, S, T, n, n_slots)

    run = jax.jit(jax.vmap(one, in_axes=(0, 0, 0)))

    t = time.perf_counter()
    out = run(pts, segs, t0s)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    out = run(pts, segs, t0s)
    jax.block_until_ready(out)
    run_s = time.perf_counter() - t

    return Result(batch=batch, compile_s=compile_s, run_s=run_s,
                  episodes_per_hour=batch / run_s * 3600.0,
                  env_steps_per_s=batch * n / run_s,
                  steps_per_episode=n)


# The rig's own ceiling: two iPhones at 1x real time. Everything is measured
# against this and nothing else.
RIG_EPISODES_PER_HOUR = 15_000


def sweep(batches=(1, 1024, 4096, 16384), **kw) -> list[Result]:
    results = []
    for b in batches:
        r = benchmark(b, **kw)
        results.append(r)
        print(f"batch {r.batch:>6}  compile {r.compile_s:7.1f}s  run {r.run_s:8.2f}s  "
              f"{r.episodes_per_hour:12,.0f} ep/h  "
              f"{r.episodes_per_hour / RIG_EPISODES_PER_HOUR:6.1f}x rig", flush=True)
    return results


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 1024, 4096, 16384])
    ap.add_argument("--steps", type=int, default=None,
                    help="substeps per episode; default is the full 2.3 s window")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args(argv)

    import jax
    print(f"jax {jax.__version__}  devices {jax.devices()}", flush=True)
    results = sweep(tuple(a.batches), n_steps=a.steps)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
