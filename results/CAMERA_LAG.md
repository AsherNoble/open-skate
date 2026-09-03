# The camera lag: costs 38% of observations, and the corpus cannot price it

## It is the lag, not the physics, that loses the board

Replaying the fitted chase camera over 64 collected trajectories, counting
episodes whose board centre stays inside the frustum for every frame:

| tau | framed, all frames (physical episodes) |
|---|---|
| 0.000 | **1.000** |
| 0.020 | 1.000 |
| 0.050 | 0.906 |
| 0.090 | 0.585 |
| **0.180 (fitted)** | **0.340** |
| 0.360 | 0.189 |

At tau = 0 — a camera rigidly locked to the board — **every** physical episode
stays framed. So the board never goes anywhere a camera could not follow, and
the episodes that lose it are lost by the lag. "The camera can't keep up" and
"the physics throws the board off the map" are indistinguishable in a
visibility number and need completely different fixes; this separates them.

(The frustum test is conservative — it requires the board's *centre* in frame,
while the board is extended and can be partly visible with its centre outside.
It agrees with rendered visibility on 70% of episodes and is the stricter of
the two, so read the ordering, not the level.)

## But the corpus cannot say what the lag should be

Swept against the corpus score on two halves. **Over a truncated range it looks
identified. It is not.**

| tau | half A | half B | mean |
|---|---|---|---|
| 0.00 | 0.996273 | 0.887182 | 0.941727 |
| 0.02 | 1.015257 | 0.943836 | 0.979546 |
| 0.05 | 0.945898 | 0.882727 | 0.914313 |
| 0.09 | 0.951594 | 0.954384 | 0.952989 |
| 0.18 | 0.948481 | 0.921750 | 0.935116 |
| 0.36 | 0.905516 | 0.838216 | 0.871866 |
| 0.60 | 0.899736 | 0.861811 | 0.880773 |
| 1.00 | 0.902944 | 0.858178 | 0.880561 |
| 2.00 | 0.892546 | 0.863262 | 0.877904 |
| 5.00 | 0.879222 | 0.884969 | 0.882095 |

Stopping at 0.36 — where the first sweep ended — both halves agree the minimum
is 0.36, and the range (0.11) exceeds the gap between halves. That looks like
identification. **Extend the range and it evaporates**: A's minimum runs to
5.00, B's stays at 0.36, and the spread collapses to 0.026 and 0.047.

**A sweep that stops at its own argmin has not found a minimum, it has found
its own edge.**

## And the overlay says why

Standing rule 1. Rendered silhouettes for the same gesture at tau = 0.18 and
tau = 2.0, an elevenfold change in the lag:

    tau 0.18   mask coverage  0.064  0.062  0.056  0.050  0.043
    tau 2.00   mask coverage  0.064  0.061  0.055  0.049  0.040

Visually near-identical. The objective can barely see this parameter, so its
preference between lag values is close to noise — exactly what two halves
pointing at opposite ends of an extended range implies.

## What follows

**Do not retune tau to improve framing.** It would buy observability with
fidelity that has not been measured, and the corpus cannot referee. The
tension is real and belongs to the user:

- keeping tau = 0.18 costs ~38% of rendered observations;
- lowering it toward 0 frames everything, but on no evidence that the game
  behaves that way;
- **widening the field of view instead** changes only what the camera SEES, not
  where a screen touch lands, so it buys framing without touching the physics.
  That is the cheap option and it is untested.

## An unrelated observation, recorded for later

The real masks do not look like the simulated ones. Sim renders a clean
elongated blob; the real segmentation is a peanut with a narrow waist and
internal holes — plausibly the rider's foot, or the deck graphic, or a
segmentation artefact. Nobody has checked which. It bears on every silhouette
score in this project and deserves its own look.
