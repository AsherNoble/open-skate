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

Three things this says, in order of importance:

1. **The premise survives.** At B=1024 the A10G runs 959K episodes/hour
   against the rig's 15K — a **64x** speedup, and the reason the port was
   worth doing.
2. **Batch size is the whole story, and B=1 is worse than the phone.** A
   single episode on an A10G runs at 0.3x the rig. Nothing about MJX is fast;
   the parallelism is.
3. **Throughput PEAKS near B=1024 and falls off hard.** B=4096 is 3.3x slower
   than B=1024, and compile time goes 14.6 s -> 61.7 s. The A10G's 24 GB is
   saturated somewhere between the two and the run starts spilling. Larger is
   not better: **run several 1024-wide batches rather than one 4096-wide one.**

Compile time is a fixed 13-15 s up to B=1024, so a training run that keeps one
compiled batch size pays it once and never again.

## What is not measured here

Rendering. Pixels need MJX-Warp, which needs NVIDIA, and at 68 frames per
episode a 1024-wide batch is ~70K frames per call. Rendering plausibly
dominates physics and could erase the margin above; that measurement decides
whether pixel observations are viable at scale, and it is the next thing to
take.
