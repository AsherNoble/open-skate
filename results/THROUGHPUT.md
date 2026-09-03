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

> **RETRACTED AND RE-MEASURED.** The first version of this section reported
> ~95K frames/s at 128x64. It was **wrong**: the renderer takes its image size
> from the CAMERA's `resolution`, not from `vis.global_.offwidth/offheight`,
> and with resolution unset it rendered **1x1 pixel images**. The rate was real
> and the images were single pixels, so the number said nothing about 128x64.
>
> Nothing in the summary statistics revealed this — a 1x1 render produces
> well-formed frames at a plausible rate, and `frac_nonbackground` was the only
> figure that looked wrong. It was caught by printing `rgb.shape` on an
> end-to-end run. **This is the third time in this project a "finding" turned
> out to be my own bug hidden by statistics**, and the second where the fix was
> to look at the actual pixels rather than a summary of them.
>
> The table below is the re-measurement at a real resolution.

128x64 RGB, the phone's 19.5:9 at a size a world model consumes. One `render`
call produces one frame for every world at once; an episode needs 68 of them.

| worlds | ms/call | frames/s | render a whole episode batch |
|---|---|---|---|
| 64 | 16.26 | 3,937 | 1.11 s |
| 256 | 13.38 | 19,129 | 0.91 s |
| 1024 | 12.75 | **80,330** | **0.87 s** |

**Per-call cost is nearly flat from 64 to 1024 worlds** — 13-16 ms whatever the
width — so frames/s scales almost linearly and the renderer is far from being
the bottleneck it was expected to be. (The 1x1 measurement got this part right
for the wrong reason: at one pixel per image the cost was almost entirely
per-call overhead. At a real resolution the per-call cost is only 25% higher,
which is what actually makes the conclusion survive.)

At the B=1024 operating point:

| | seconds per 1024-episode batch |
|---|---|
| physics | 3.84 |
| rendering, 68 frames | 0.87 |
| **total** | **4.71** |

**~782,000 episodes/hour with pixels, 52x the rig.** Rendering is 23%
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

`render_with_segmentation` returns RGB, depth and segmentation together, so it
is dearer than RGB alone rather than cheaper. At a real 128x64:

| worlds | ms/call | frames/s | episode batch |
|---|---|---|---|
| 256 | 16.35 | 15,662 | 1.11 s |
| 1024 | 13.91 | 73,642 | 0.95 s |

**9% dearer than RGB** at B=1024 (0.95 s against 0.87 s) — a smaller penalty
than the 1x1 run suggested (24%), but still a penalty.

Either way the plan's "try masks first, they are the cheap mitigation" is the
wrong shape: masks cost MORE to produce, and their value is entirely in closing
the appearance gap between our render and True Skate's. That is now the only
reason to choose them.

## A correctness trap in the Warp backend: contacts are silently dropped

Running real physics on the Warp backend (which the pixel path needs, since the
batch renderer only exists there) printed, on nearly every step:

    broadphase overflow - please increase nconmax to 9 or naconmax to 2078
    narrowphase overflow - please increase nconmax to 1 or naconmax to 54

The Warp backend **preallocates contact buffers and discards contacts past
them**. It prints and carries on rather than failing, so the simulation quietly
becomes one where the board is partly not touching the ground. Nothing in a
throughput number or an outcome summary would reveal this — a board that falls
through the floor still produces a perfectly well-formed trajectory.

The counts are totals across all worlds, so they scale with the batch. Sized at
`naconmax = njmax = 64 * batch`, the warnings stop.

This does **not** affect any number above it in this file: the physics sweep
runs the JAX backend, which sizes contacts dynamically, and the render sweep
never stepped. It does affect anything run on Warp from here on.

Note also that `make_data` takes `naconmax` and `njmax` — there is no `nconmax`
argument, despite the error message naming one.
