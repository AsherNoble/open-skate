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
