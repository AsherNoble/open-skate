# Throughput — measured, not projected

The rig baseline is **15,000 gesture evaluations per hour**: two physical
iPhones driven by Appium at 1x real time. Open Skate is only worth building if
a batched rollout clears that by a wide margin. This file records whether it
does.

Episode = one whole gesture over the 2.3 s capture window = **1150 substeps**,
sampled to 68 frames at 30 fps. Physics only; no rendering. Compile time is
reported separately and never amortised into the rate, because a training run
pays it once.

## Apple M2 (MJX-JAX on the CPU backend) — negative

| batch | compile | run | episodes/hour | vs rig |
|---|---|---|---|---|
| 1 | 4.3 s | 0.29 s | 12,339 | 0.8x |
| 16 | — | 4.33 s | 13,304 | 0.9x |
| 64 | 24.6 s | 17.26 s | 13,348 | 0.9x |

**Flat with batch size, and below the baseline.** On CPU `vmap` is B times the
work done serially, so batching buys nothing at all. This is exactly what the
MJX documentation predicts — single-scene MJX is roughly 10x slower than
MuJoCo C — and it refutes nothing, because the thesis was never that MJX is
fast; it is that thousands of scenes step at once.

## NVIDIA A10G (Modal, `gpu="A10G"`) — the thesis holds

| batch | compile | run | episodes/hour | vs rig |
|---|---|---|---|---|
| 1 | 13.8 s | 0.94 s | 3,842 | **0.3x** |
| 256 | 12.7 s | 2.02 s | 457,135 | 30x |
| 1024 | 14.6 s | 3.84 s | **959,373** | **64x** |
| 4096 | 61.7 s | 51.24 s | 287,768 | 19x |
| 16384 | 219.0 s | 210.86 s | 279,728 | 19x |

Three things this says, in order of importance:

1. **The premise survives.** At B=1024 the A10G runs 959K episodes/hour
   against the rig's 15K — a **64x** speedup, and the reason the port was
   worth doing.
2. **Batch size is the whole story, and B=1 is worse than the phone.** A
   single episode on an A10G runs at 0.3x the rig. Nothing about MJX is fast;
   the parallelism is.
3. **Throughput PEAKS near B=1024 and falls off hard.** B=4096 is 3.3x slower
   than B=1024, and B=16384 no better than B=4096 — the curve does not recover.
   Compile time goes 14.6 s -> 61.7 s -> 219 s over the same range. The A10G's
   24 GB saturates somewhere between 1024 and 4096 and the run starts spilling.
   Larger is not better: **run several 1024-wide batches rather than one wide
   one.** This is a per-GPU number and should be re-measured on anything with
   more memory before assuming it holds.

Compile time is a fixed 13-15 s up to B=1024, so a training run that keeps one
compiled batch size pays it once and never again.

## What is not measured here

Rendering. Pixels need MJX-Warp, which needs NVIDIA, and at 68 frames per
episode a 1024-wide batch is ~70K frames per call. Rendering plausibly
dominates physics and could erase the margin above; that measurement decides
whether pixel observations are viable at scale, and it is the next thing to
take.

## Rendering (A10G, MJX Warp batch renderer) — it does not erase the margin

128x64 RGB, the phone's 19.5:9 at a size a world model consumes. One `render`
call produces one frame for every world at once; an episode needs 68 of them.

| worlds | ms/call | frames/s | render a whole episode batch |
|---|---|---|---|
| 64 | 10.23 | 6,256 | 0.70 s |
| 256 | 10.02 | 25,541 | 0.68 s |
| 1024 | 10.73 | **95,463** | **0.73 s** |
| 4096 | 19.71 | 207,799 | 1.34 s |

**Per-call cost is nearly flat from 64 to 1024 worlds** — about 10 ms whatever
the width — so frames/s scales almost linearly and the renderer is far from
being the bottleneck it was expected to be.

At the B=1024 operating point:

| | seconds per 1024-episode batch |
|---|---|
| physics | 3.84 |
| rendering, 68 frames | 0.73 |
| **total** | **4.57** |

**~806,000 episodes/hour with pixels, still 54x the rig.** Rendering is 19%
overhead. **Pixels survive as the observation** and masks stay a fidelity
choice rather than a throughput necessity.

**Caveat, stated rather than buried:** this renders a static scene repeatedly.
It does not include whatever BVH refit a moving board costs per step. The
margin is wide enough that this is unlikely to change the answer, but it is not
the same measurement as rendering a live rollout, and the honest version of
this number comes from the batch-major rollout once that exists.

### What blocks pixels now is structural, not throughput

The render context **cannot be used under `vmap`** — its own docstring says
nworld is hardcoded because Warp allocates arrays JAX cannot see. The physics
rollout is env-major (`vmap` over independent episodes, each its own `scan`),
so the pixel path needs a batch-major rewrite: one batched `Data`, stepped and
rendered as a whole. That is the next piece of work, and the numbers above say
it is worth doing.

### Masks are not cheaper than pixels

`render_with_segmentation` returns RGB, depth and segmentation together:

| worlds | ms/call | frames/s | episode batch |
|---|---|---|---|
| 256 | 12.98 | 19,723 | 0.88 s |
| 1024 | 13.31 | 76,951 | 0.90 s |

24% dearer than RGB alone (0.90 s against 0.73 s at B=1024), not cheaper. So
the plan's "try masks first, they are the cheap mitigation" is only half right:
masks cost slightly MORE to produce. Their value is entirely in closing the
appearance gap between our render and True Skate's, not in throughput — and
that is now the only reason to choose them.
