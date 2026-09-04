# The stored parameters do not minimise the current objective

## The finding

On the flick corpus (n=120, two halves judged separately), the fitted
parameters score **worse than a board with no finger at all**:

| variant | half A | half B | mean |
|---|---|---|---|
| fitted | 1.00813 | 0.87594 | 0.94204 |
| no ground shove | 0.96754 | 0.88097 | 0.92426 |
| weak finger (10%) | 0.89213 | 0.74267 | 0.81740 |
| **no deck force** | **0.79461** | **0.64830** | **0.72146** |
| no finger at all | 0.81695 | 0.65314 | 0.73504 |

Lower is better. Removing the deck force improves the score by 23%.

## It is not the obvious things

**Not the mask change made the same day.** A controlled A/B on one fixed sample
list, cache disabled, scores both ways: filled −0.103, unfilled −0.080. Both
negative. Hole-filling makes it marginally worse and is not the cause.

**Not capture latency.** `score_sample` carries a `time_offset` for
command→pixel delay that the rig documents as uncalibrated, and the objective's
own source warns that a real 100–200 ms delay would compare every trajectory
against frames 3–6 too early, so that "even correct physics would score no
better than not moving" — exactly this symptom. Swept it:

| offset (s) | fitted | inert | margin |
|---|---|---|---|
| 0.00 | 0.96772 | 0.84416 | −0.12356 |
| 0.10 | 0.95811 | 0.84416 | −0.11396 |
| 0.20 | 0.91796 | 0.84416 | −0.07381 |
| 0.30 | 0.88870 | 0.84416 | −0.04454 |

The margin shrinks but **never flips**, and 0.30 s is already beyond any
credible delay. Latency is not the explanation.

**And not "the objective rewards stillness".** That failure was falsified once
before and would show as a minimum at zero force. It is not: "no deck force"
(0.721) beats "no finger at all" (0.735), so the objective wants *some* motion.

## What it is: the deck force is ~55x too strong

Sweeping `touch_gain`, both halves agree on a clean interior minimum:

| touch_gain | half A | half B | mean |
|---|---|---|---|
| 5 | 0.79293 | 0.64559 | 0.71926 |
| 20 | 0.78254 | 0.63557 | 0.70906 |
| 30 | 0.73956 | 0.60614 | 0.67285 |
| **45** | **0.71864** | **0.59451** | **0.65658** |
| 70 | 0.73426 | 0.61540 | 0.67483 |
| 120 | 0.84168 | 0.67742 | 0.75955 |
| 2490.4 (**stored**) | 1.00813 | 0.87594 | 0.94204 |

A smooth U with its floor at **45**, on both halves independently. The stored
value is **2490.4** — inside its (30, 3000) fit bounds, so nothing clamped it.
At the optimum the score is 30% better than at the stored value.

## The likely history

The plan records that the objective went through **three revisions**, each
forced by a measurement: raw silhouette IoU (rewarded stillness), gain over
stationary (still retreated), then departure curves plus phase-tolerant axis
aggregates. The stored parameters appear to have been fitted against an earlier
version and **never refitted after the objective was repaired**.

That single fact explains a cluster of otherwise unrelated observations:

- the fitted physics throws the board 10–22 m from ordinary gestures;
- `ground_shove_gain` sat pinned at 99% of its bound, unidentifiable;
- both force-reducing changes made today improved held-out scores on both
  halves — small corrections in the direction of a much larger one;
- an inert board beats the fit.

## What this invalidates

**Every fidelity number in the record predating this measurement**, including
the plan's "beats an inert board on held-out data — 1.08–1.37 against inert
1.49", which the table above directly contradicts. Those were measured either
with a different objective version or a different corpus, and they cannot be
compared with anything measured now.

Throughput, parity and rendering results are unaffected — none of them depend
on the fitted values.

## What to do

**A full joint refit against the current objective.** Not a single-parameter
edit: the other 26 fitted parameters were chosen to compensate for a deck force
55x too strong, so moving `touch_gain` alone could easily make the whole worse.
The sweep says where the objective points; it does not say what the jointly
optimal set is.

Until that refit lands, treat the stored `SkateParams` as **unvalidated against
the current objective**.

## What the overlay did and did not confirm

Standing rule 1 says render it and look. Done, on a gesture that does touch the
deck, at the stored gain and at the objective's optimum:

    gain 2490.4   IoU per frame  0.420  0.088  0.000  0.000  0.078
    gain   45.0   IoU per frame  0.420  0.088  0.000  0.000  0.084

**The overlay could not tell the two apart.** A 55x change in the deck force
produces near-identical silhouettes on this sample. So the sweep's preference
comes from the population, not from any single dramatic case, and the n=120
two-half sweep — not this picture — is what the finding rests on.

**What the picture did show is worse, and applies to both.** IoU collapses to
**zero by the third frame** at either gain: the simulated and real boards stop
overlapping at all within roughly 100 ms, and the sim silhouette drifts up the
frame while the real one holds its position. Whatever the right `touch_gain`
is, the trajectories part company almost immediately — which no amount of
retuning a single force constant is going to fix.

That is the more important open question, and it is not answered here.

## The joint refit: adjudicated, and it did not earn its complexity (3 Sep)

The bar was set before the search ran: *if a 22-parameter joint refit cannot
beat a one-line change, it has not earned its complexity.* 800 evaluations of
CMA-ES over `PHYSICS_KEYS`, 25 minutes, fitted on a 75-sample train half and
scored on the 75 it never saw.

| | TRAIN | HELD |
|---|---|---|
| stored | 0.984952 | 0.865380 |
| inert board | 0.767627 | 0.650036 |
| **stored + `touch_gain`=45** | **0.682059** | **0.586309** |
| stored + `touch_force_max`=53 | 0.906759 | 0.810759 |
| joint refit (22 params) | 0.731340 | 0.584292 |

**It ties on held and LOSES on train.** The held-out gap is 0.3% on n=75,
which is a tie. The train gap is not: 0.7313 against 0.6821, and train is
*exactly* what the search spent 800 evaluations minimising (`fit` minimises
`mean_combined` on random 20-sample subsamples of train; the log's "best train
activity" is a mislabelled print, not a different objective). A single-parameter
edit found a better point on the search's own objective than the search did.

**A reading of the refit that the data refuted.** The refit barely moved
`touch_gain` (2490 -> 1734) and instead cut `touch_force_max` 687 -> 53, which
looked like the same physical claim -- *the deck force is an order of magnitude
too strong* -- reached down a different parameter, and therefore like strong
corroboration. Scoring that change on its own says otherwise: `fmax53` alone
gets 0.811 held, barely better than stored's 0.865 and nowhere near gain 45's
0.586. The refit reached its score through a combination, not through an
equivalent single knob. **Not corroboration.**

**Adopted: `touch_gain = 45.0`.** The 22-parameter set is not adopted.

One honesty note on the comparison. `touch_gain = 45` was chosen by a sweep
that looked at both halves, so its held score is not a fully clean held-out
number, whereas the refit's is. That caveat only strengthens the verdict: the
refit's number is the *cleaner* one and it still loses on train.

### The measurement nearly did not survive the day

These numbers were first computed across a break in the objective. The
appearance pass named the visual cylinder wheels `vis_*`, and the fitting
silhouette selects geoms by that exact prefix, so the wheels entered the mask
the physics is fitted against -- undoing a decision recorded as measured
(wheels widen the silhouette ~26% against real frames matching within 1%).

No test moved. What caught it was the objective disagreeing with itself:
`stored` scored 0.866083 before the commit and 0.839273 after on the same
held-out half, while `inert` stayed **bit-identical** at 0.650036 -- because an
inert board lies flat with its wheels hidden under the deck and a flipping one
does not. That difference-with-an-identity is the whole diagnostic.

Renamed to `hw_`, `tests/test_silhouette_geoms.py` fails on the previous
commit, and `stored` returns to 0.865380. The residual 0.0007 is understood
and legitimate: collision spheres moved to a hidden group and visual cylinders
took their place, so occlusion of the deck outline changed slightly.

**Standing rule 3 earned its keep twice here** -- once on the statistic that
moved when it should not have, once on the one that did not move when it
should have.
