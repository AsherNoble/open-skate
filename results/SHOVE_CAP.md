# How hard can a drag push the board? Asking the corpus.

The ground shove — a finger dragging on the ground rather than the deck —
had no ceiling of its own. It first shared `touch_force_max` (854 N), which
stopped the solver going unstable but still put 854 N onto a 0.9 kg deck.

**Why it mattered enough to fix.** From the environment's own action prior, a
typical *physical* episode travels 10.7 m, reaches 2.26 m of height and reverses
direction mid-flight, driving the board to sx = −2.94 against a frustum of
|sx| < 1. Only 12–14% of physical episodes keep the board in frame, so most
rendered observations are blank. That is not a cosmetic fidelity issue; it
makes pixels untrainable.

## The measurement

Sweep the ceiling, score the corpus, and **judge on a held-out half** — train
score has already failed to predict held score in this project, so a sweep that
only ever looks at one set of samples cannot settle anything. 90 usable
samples, split 45/45 by index parity. Lower is better.

| cap (N) | half A | half B |
|---|---|---|
| 854 | 0.930612 | 0.970293 |
| 400 | 0.930418 | 0.972170 |
| 200 | 0.926849 | 0.979027 |
| **100** | **0.890931** | **0.921388** |
| 50 | 0.907704 | 0.929331 |
| 25 | 0.931198 | 0.952338 |
| 10 | 0.881829 | 0.933722 |
| 5 | 0.900168 | 0.944583 |

## What this says, and what it does not

**It says the fitted shove was too strong.** At 854 N the cap never binds — the
score is bit-identical to uncapped — so the corpus was never able to object.
Bring the ceiling down to where it does bind and the fit gets *better*, by ~4%
on both halves independently. The direction is agreed across the split, which
is the only reason it is worth acting on.

**It does not identify the minimum.** The curve is non-monotone: 200 N is worse
than 854 N on half B, 25 N is worse than 50 N on both, and 10 N beats 100 N on
half A while losing to it on half B. 100 N is chosen because it is the one value
at or near the minimum on *both* halves — not because the objective has a clean
optimum there. Treat 100 N as "the right order of magnitude, measured", not as
a fitted parameter.

**It is a change to the physics**, made on evidence rather than convenience: it
improves the objective on held-out data. The full refit that would settle
`ground_shove_gain` properly is still outstanding.

## What it bought, measured through the environment

Same 64 actions from the environment's own prior, before and after:

| | 854 N | 100 N |
|---|---|---|
| episodes that stayed physical | 33 / 64 | **51 / 64** |
| board framed, among physical episodes | 12–14% | **53%** |
| median displacement, all | 22.8 m | **9.7 m** |
| median displacement, physical | 10.7 m | **7.4 m** |

**Better, and not yet solved.** Half of physical episodes still lose the board,
and the median one still travels 7.4 m in 2.3 s — about 3 m/s, which is a fast
push rather than an impossible one, but far more than a flick should produce.

The ceiling was the blunt instrument; what remains is `ground_shove_gain`
itself, which sets how much thrust a given screen travel asks for before any
ceiling applies. The two are now fighting each other — a gain tuned without a
ceiling, and a ceiling measured against that gain — and only a refit of the
gain with the ceiling in place will settle it.
