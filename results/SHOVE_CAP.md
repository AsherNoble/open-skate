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

## The gain itself is NOT identified by this corpus

With the 100 N ceiling in place, sweeping `ground_shove_gain` over a 100× range
on the same two halves:

| gain | half A | half B | mean |
|---|---|---|---|
| 1.00 | 0.880237 | 0.940199 | 0.910218 |
| 5.00 | **0.861389** | 0.957138 | 0.909263 |
| 15.00 | 0.901718 | 0.932168 | 0.916943 |
| 40.00 | 0.955106 | 0.931769 | 0.943438 |
| 80.00 | 0.948481 | 0.921750 | 0.935116 |
| 118.82 | 0.890931 | **0.921388** | 0.906159 |

**The corpus cannot decide.** The mean varies by 0.04 across a hundredfold
change in the parameter — less than the 0.06 gap between the two halves — and
the halves prefer opposite ends of the range: A wants 5.0, B wants 118.82.

This also explains the fitted value. `ground_shove_gain` is 118.82 against a
bound of (1.0, 120.0): pinned at 99% of its ceiling, which is what a parameter
looks like when the objective cannot see it and the optimiser drifts to an
edge.

**And the reason is a capture convention, not a modelling failure.** The flick
corpus never pushes — gestures that land on the ground rather than the deck are
rare in it — so there is almost no evidence in the data about what a ground
drag does. Refitting the gain against these samples would be fitting noise, and
the two halves disagreeing about the direction is that noise showing.

### What would settle it

Capture gestures that actually touch the ground. The rig can do this — the
phones are available, and `execute_gesture_params` already has the push flag
that produced the corpus-convention bug earlier. A few hundred ground-drag
gestures with their outcomes would constrain this parameter directly, where
90 flicks constrain it not at all.

Until then `ground_shove_gain` stays at its fitted 118.82 with the 100 N
ceiling: **not because that value is right, but because nothing measured so far
says what is.**

## The gestures the deck fit throws away DO identify it

`build_corpus(want_deck=False)` returns the inverse set: gestures whose finger
never reaches the board. 113 of them in 1391 samples. They are the only samples
where the shove is the *only* force acting, so if anything constrains it, they
do.

| gain | half A | half B | mean |
|---|---|---|---|
| 1.00 | 0.549464 | 0.497692 | 0.523578 |
| 5.00 | 0.547869 | 0.497200 | 0.522535 |
| 15.00 | 0.556828 | 0.497241 | 0.527034 |
| 40.00 | 0.532345 | 0.463756 | 0.498050 |
| **80.00** | **0.526332** | **0.460710** | **0.493521** |
| 118.82 | 0.559124 | 0.460857 | 0.509991 |

**Both halves put the minimum at 80.** That is the thing the deck corpus could
not do — there, A wanted 5.0 and B wanted 118.82. Both halves here also prefer
the 40–80 region over either extreme.

**How strong is this, honestly.** Moderate, not decisive. The spread across the
whole gain range is 0.033 (A) and 0.037 (B), while the gap between the halves
is 0.098 — the two halves differ in absolute difficulty more than the parameter
moves either of them. Half B is nearly flat above 40 (0.4638 / 0.4607 / 0.4609),
so the agreement on 80 is carried mostly by half A. What is solid is the
*direction*: 118.82 is not supported, and the 40–80 region is.

**A tension worth stating.** On the deck corpus, 80 scores slightly worse than
118.82 for half A (0.9485 vs 0.8909) and the same for half B. So the two
corpora mildly disagree. The ground corpus wins the argument because it is the
one that can actually see the parameter: the deck corpus fails its own
identifiability check, and a preference expressed by an objective that cannot
distinguish a hundredfold change is not evidence.

`ground_shove_gain` set to **80.0**. Better supported than 118.82 — which was
pinned at 99% of its fit bound — and still not a substitute for a capture
designed to measure the push directly.
