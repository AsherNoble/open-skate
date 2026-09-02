# Round-trip transfer test — 3 Sep 2026

**Result: NEGATIVE. The physics does not transfer yet.**

## Method
1. `opensk/fit/optimise_gesture.py` searched Open Skate for a gesture producing
   a −360° roll about the deck's long axis. Target stated as rotation only —
   Open Skate has no notion of a trick name.
2. Best gesture: **−346° roll, +23° yaw, 13 cm pop, 0.49 s airborne, landed upright.**
3. Executed on iPhone_XR, 8 trials, board reset before each, **no push** (matching
   how the corpus was captured and how the gesture was optimised).
4. The rig's `detect_trick` OCR named whatever the real game did.

## What the real game reported

| trial | trick | status |
|---|---|---|
| 0, 2, 3 | — | not detected |
| 1 | POP SHOVE-IT | failed |
| 4 | LIPSLIDE | failed |
| 5 | NOSEBLUNT | failed |
| 6 | NOSE SLIDE + BOARDSLIDE + BOARDSLIDE | failed |
| 7 | BOARDSLIDE | failed |

**No flip. Nothing landed.**

## Reading it
- The one rotational trick detected was a POP SHOVE-IT — rotation about the board's
  NORMAL (yaw), where the simulation predicted rotation about its LONG AXIS (roll).
  The fit reproduces motion *magnitude and timing* (held-out activity 0.2572 vs
  inert 0.3849) without reproducing the **axis**, and the activity objective is
  blind to axis by construction: it measures how far the silhouette departs its
  start, not in which direction.
- Lipslide / boardslide / noseblunt all require a **rail or ledge**. The real board
  is translating across the park into obstacles. The sim replays on `FLAT_PARK`, so
  the board has nothing to hit — a first-order environment mismatch, and a likely
  contributor to the failed landings.
- Every trial failed, so this is not a near miss.

## What this does NOT invalidate
The held-out activity result stands on its own terms: the fitted physics tracks the
real board's motion magnitude and timing better than a stationary board. Transfer is
a strictly stronger claim and it is not yet supported.

## Next
1. Make the objective **axis-aware** — it cannot currently distinguish a flip from a
   shove-it, which is exactly the distinction that failed here.
2. Replay against the SLS park rather than flat ground, so obstacle interaction is
   represented.
3. Re-run this test after both. It is cheap and it is the only test that matters.


---

# Round-trip v2 — after the axis-aware objective (3 Sep 2026)

**Result: PARTIAL. The rotation axis now transfers. Landing does not.**

Refitted physics (held combined 0.9296 vs inert 2.1261), new gesture optimised for
+360° roll: sim gave **+370° roll, 20 cm pop, 0.54 s airborne, landing upright**.
10 trials on iPhone_XR, reset each time, no push.

| outcome | v1 (before) | v2 (after) |
|---|---|---|
| **flips — correct axis** | **0/8** | **2/10** |
| slides (board reaches rails) | 4/8 | 6/10 |
| shove-it (wrong axis) | 1/8 | 0/10 |
| nothing detected | 3/8 | 2/10 |
| **landed** | 0 | **0** |

v2 flips: `QUAD KICKFLIP + VARIAL TRIPLE KICKFLIP`, `720 DOUBLE FLIP` — both
rotations about the deck's long axis, which is what the simulation asked for. The
wrong-axis shove-it that motivated the fix did not recur.

## What this supports, and what it does not
- **Supported:** making the objective axis-aware changed real-device behaviour in the
  predicted direction. Flips appear where there were none.
- **NOT supported:** transfer. 2/10 is weak, nothing landed, and the dominant outcome
  is still slides — the board travelling into rails, i.e. the environment/translation
  mismatch, untouched by this change.
- The flips reported are *multi*-rotation (quad, 720 double) where the sim asked for
  one. The magnitude is over, not merely noisy.

## Next
1. **Translation is the dominant remaining error.** Six of ten trials ended on a rail.
   The sim replays on flat ground and cannot see translation anyway (board-locked
   camera), so nothing in the fit constrains it.
2. Over-rotation: the real board turns several times where the sim predicts one.
3. A faithful replica of the capture park would remove the obstacle mismatch.
