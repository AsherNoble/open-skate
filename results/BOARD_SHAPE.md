# The board's shape, measured against the game's own geometry (4 Sep)

## Why this exists

The project claimed the board's shape was matched — "deck outline fitted to the
real board, RMS 0.041", "camera within 1%". **Those are true statements about
ONE projection**: the chase camera's silhouette, the only view the frame corpus
contains. No other angle had ever been checked against anything, and the deck is
a strip of boxes with ellipsoid tips.

Recovering shape from monocular gameplay frames is an ill-posed inverse problem
— unknown per-frame pose, and concave depth unobservable from silhouettes at
any sample size. The game ships the geometry instead.

## The format

`res/*.bin` in the app bundle, unencrypted, four-character magics:
`OMSH`, `SKWH`, `SKDE`, `SKTR`. Layout, after a per-magic header:

```
u32         index count
u16[n_idx]  index buffer
u32         vertex count
f32[nv][3]  positions      <- two SEPARATE streams,
f32[nv][2]  UVs            <- not interleaved
```

The layout is not a guess. The game's own SPIR-V vertex shaders declare exactly
two inputs — `a_v4Position` (vec4 f32) and `a_v2TexCoord` (vec2 f32) — and a
vec4 position fed from three components is the ordinary case, giving 12 + 8 = 20
bytes per vertex.

**The hard part was two-stream vs interleaved**, because 20 bytes per vertex
divides the block either way and the arithmetic cannot tell them apart. Reading
it interleaved silently mixes UVs into the position columns; the tell was that
every position axis then shared one range (min −0.629 on all three). The
discriminating check is that decoded **UVs must land in [0, 1]** — they do on
all four supported files and do not under the interleaved reading.

### What is deliberately NOT decoded

- **`SKDE` (deck top) and `SKTR` (truck)** are multi-part containers whose
  header length is unknown. `deck.bin` carries indices reaching vertex 1251
  while the count read at the guessed offset says 945 — proof the offset is
  wrong, not a reason to pick a bigger number. `decode` raises.
- **Topology, for every file.** Read as a strip the meshes are not manifold
  (`wheel.bin`: 236 edges used once against 47 used twice); read as a list,
  `oldschool_wheels.bin` gives 110 triangles sharing no vertices at all.
  Neither is a surface. And two wheels of *different* tessellation (236 vs 330
  vertices) both produced exactly **110** triangles — a statistic that does not
  move when the input changes is a broken instrument. `triangles()` raises.

Positions and UVs are validated and sufficient for what follows: a point cloud
gives a plan-view outline and a side profile, and those are what the silhouette
is made of.

## Scale, recovered from two independent anchors

| assume | implies |
|---|---|
| deck is a standard **8.0 in** wide | length **31.84 in** (a standard 32 in deck) |
| deck length is our fitted **0.813 m** | width **8.04 in** |

Nothing in the data ties these together, so agreeing to **0.5%** is evidence
about the unit rather than a restatement of the assumption. **1 game unit ≈
39.86 mm.**

## The comparison

| | GAME | SIM | |
|---|---|---|---|
| deck length (m) | 0.8088 | 0.8130 | −0.5% |
| **deck width (m)** | **0.2032** | **0.1960** | **+3.7%** |
| deck tip width / waist | 0.469 | 0.500 | −6.1% |
| kick angle (deg) | ~22.1 | 19.0 | +16% |
| flat fraction | ~0.77 | 0.60 | +29% |

**Length is right; width is not.** The fitted deck is a 7.72 in board where the
game's is a standard 8.0 in — 3.7% narrow. Length and width come from raw
extents and are robust. **Kick angle and flat fraction come from a binned
profile heuristic and should be read as approximate**, not as fitted values.

The plan-view profile is flatter than the parametric model assumes: half-width
is constant to within 0.5% across the middle **68%** of the deck and only
tapers over the last ~12% at each end.

## What has NOT been changed

No parameter has moved. `deck_width` feeds the **collision** boxes as well as
the visual outline, so changing it alters contact — which would invalidate the
determinism baselines, the MJX parity gate and every fitted number at once.
Per the plan, that goes through an A/B on both corpus halves with the
segmentation cache off, not a direct edit.

Nothing from the game bundle is committed to this repo. The decoder and the
derived measurements are; the art is not.

---

## The A/B: the objective CANNOT identify deck width (4 Sep)

Scored on both halves, stored parameters, only `deck_width` moved. A deliberate
**absurd control** was included, because a change this small needs something to
prove the instrument can see it at all.

| deck_width | half A | half B |
|---|---|---|
| fitted 0.1960 (7.72 in) | 0.984952 | 0.865380 |
| game 0.2032 (8.0 in) | 0.982147 | 0.879247 |
| **absurd 0.2600 (10.2 in)** | **0.973483** | 0.865128 |

**The absurd 10.2 in plank scores BEST on half A and ties on half B.** That is
not a marginal result to be averaged away — it means the objective cannot
discriminate deck width, and if anything rewards a grotesquely wide deck.

Without the control this would have read as "0.2032 improves A, worsens B, not
identified" and been filed as a weak negative. With it, the conclusion is much
stronger and much worse: **`mean_combined` is not a valid instrument for shape
either.** That now sits alongside the finding that it ranks a *non-functional*
board as its best point.

**So `deck_width` was NOT changed.** Not because the measurement is doubtful —
it is the most direct evidence in this project, taken from the game's own
vertices with scale confirmed from two independent anchors — but because
changing fitted physics on the silence of a broken instrument is exactly the
mistake that produced the `touch_gain` mess.

### The part that matters most for whoever picks this up

**The width discrepancy and the camera calibration are entangled, and cannot be
separated by silhouette fitting.** `deck_width = 0.196` was itself obtained by
fitting the deck outline to real frames — through a camera whose FOV and
distance are themselves fitted. A board 3.7% too narrow and a camera 3.7% too
close produce the *same silhouette*. The mesh measurement is the first evidence
in this project that is independent of the camera, which makes it a way to
break that degeneracy:

> If the true deck is 0.2032 m and real frames still match at the old apparent
> size, then the camera scale is off by ~3.7% and it is the CAMERA that should
> move, not the deck.

That is a concrete, testable next step, and it may bear on the unexplained
sim-vs-real IoU collapse by frame 3 — a scale error would show up immediately
and grow, which is what that failure looks like.

## Physics impact of the width change, characterised

Same 24 gestures, CPU, fitted vs game width. Contact genuinely changes — 19 of
24 episodes differ — but the distribution barely moves:

| field | fitted | game width | shift |
|---|---|---|---|
| roll_deg | −13.727 | −20.331 | −0.018 sd |
| yaw_deg | −5.380 | −3.108 | +0.050 sd |
| peak_height | 0.138 | 0.157 | +0.090 sd |
| air_s | 0.474 | 0.476 | +0.003 sd |
| displacement | 3.726 | 3.322 | −0.101 sd |

Every mean moves less than 0.11 sd. Whatever the width should be, it is not
what makes this simulator violent.

## A plotting bug worth recording

The first outline drawing showed two sharp V-notches at roughly the truck
positions, which looks exactly like bolt recesses. It was an artifact: the mesh
is low-poly through the waist (24 vertices per bin), one bin was empty, and the
outline was drawn straight across it. The half-width is in fact constant to
within 0.5% from t = 0.175 to t = 0.825 — **the middle 65%** — and the empty bin
had no shape meaning at all. Checked by printing per-bin vertex counts before
believing the picture.

---

# THE BOARD NOW HAS THE GAME'S SHAPE (4 Sep)

Everything above measured the deck and changed nothing. This section is the
change. `opensk/sim/model/deck_profile.py`, `tests/test_deck_shape.py`.

## Three corrections to the measurements above

**1. The deck is 8.25 in, not 8.0 in — and the earlier scale was circular.**
The width above was taken from `deck_bottom.bin`, the UNDERSIDE surface, which
is inset ~3 mm from the deck's outer edge. `edge_top.bin` and
`edge_bottom.bin` — the perimeter band, 1664 vertices each, both decodable —
carry the true outline at **5.2496 u**, not 5.098. And the unit itself was
fixed by *assuming* the deck was 8.0 in, so reporting the deck's size from it
restated the assumption.

The scale is now anchored on the **wheel**, which is independent of the deck.
`wheel.bin` is 1.2546 u across; taken as a 50 mm street wheel that gives
**39.85 mm per unit**, and four further quantities then land on standard parts
without being asked to:

| quantity | game units | at 39.85 mm/u | standard? |
|---|---|---|---|
| deck width | 5.2496 | 209.2 mm = **8.24 in** | 8.25, a stock size |
| deck length | 20.2953 | 808.8 mm = **31.84 in** | its usual pairing |
| deck thickness | 0.3014 | **12.0 mm** | a 7-ply deck |
| oldschool wheel | 1.4447 | 57.6 mm | a cruiser wheel |

Four results from one assumed number. The previous "two independent anchors"
were the deck measured two ways.

A fifth check comes from inside our own model: `axle_halfwidth = 0.105` was
measured with a ruler and implies a 0.21 m deck. **The game's 0.2092 m agrees;
the fitted 0.196 m did not.** So the sim deck was 6.7% narrow, not 3.7%.

**2. More of the deck decodes than was thought.** `deck.bin` is `SKDE` and is
still refused, but the deck does not need it: `grip_tape.bin` (the top
surface), `deck_bottom.bin` (the underside) and the four `edge_*` files are
all `OMSH`. Between them they give the outline, the centreline and the
thickness. The plan's table listing `deck_bottom.bin` as `SKDE` was wrong.

**3. The kick is a progressive CURVE, and the deck dips before it.** The
"~22.1 deg kick angle, 0.77 flat fraction" above came from a binned heuristic
and was flagged approximate; it is. Measured along the centreline, the deck is
flat to within 0.1 mm out to t = 0.5, dips **1.2 mm** at t = 0.65, and only
then rises — through 10 deg at t = 0.75, 20 deg at t = 0.90, to a tip
**30 mm** above the flat. No straight ramp can produce that shape.

## What the deck was, and is

| | OLD | NEW | GAME |
|---|---|---|---|
| plan view | 11 constant-width boxes, taper `abs(u)**1.6` | one swept mesh | — |
| tips | 2 ellipsoids | the measured cap | — |
| side | one straight 19 deg ramp from t = 0.60 | the measured curve | — |
| width | 0.1960 m (7.72 in) | **0.2092 m** | 0.2092 m |
| length | 0.8130 m | **0.8088 m** | 0.8088 m |
| thickness | 0.0120 m | **0.0120 m** | 0.0120 m |
| tip rise | 52 mm | **31 mm** | 30 mm |
| kick starts | 0.255 m from centre | **0.31 m** | 0.31 m |
| concave | none | 5.6 mm | 5.6 mm |

The shape now lives in two measured tables rather than four scalars.
`kick_angle_deg`, `flat_fraction`, `deck_tip_width_frac` and
`deck_taper_power` are **deleted**, not left unused: a parameter that no
longer does anything is worse than no parameter, because someone tunes it and
nothing moves.

## How closely it matches, measured on the OUTPUT

The tables are the input. What renders, and what the fitting silhouette is
taken from, is the **compiled mesh** — so that is what is compared, vertex
cloud against vertex cloud, at 20 stations along the half-length:

| | rms | max |
|---|---|---|
| plan-view half-width | **1.28 mm** | 3.83 mm |
| centreline profile | **0.48 mm** | 1.29 mm |

On a 209 mm wide, 809 mm long deck. The 1.2 mm pre-kick dip is reproduced.

Two measurement bugs were found and fixed getting there, both of the kind this
project keeps hitting — a statistic that moved for a reason that was not the
physics:

* **the mid-surface, not the median.** Taking the median z of a centreline
  band mixes the top and bottom skins and jumps by the plate's whole thickness
  depending on how many of each land in a bin. It made BOTH profiles
  non-monotone, the game's included — which is what gave it away.
* **bins narrower than the mesh's ring spacing alias.** A fixed 20 mm bin
  falls between rings and reports whichever neighbour leaks in, showing up as
  a +-6 mm oscillation in the sim profile that is not in the mesh.

## Physics impact, characterised not suppressed

Contact geometry changed on purpose: the deck is wider, and the three
collision boxes became **seven** (a flat plus three per kick), because one
chord across the real curve puts the tail tip 12 mm low — a pop-height error,
not a cosmetic one. Same 32 random gestures, CPU:

| field | old | new | shift | KS | rho |
|---|---|---|---|---|---|
| roll_deg | −10.790 | −11.679 | −0.003 sd | 0.16 | 0.76 |
| yaw_deg | −2.968 | −10.215 | −0.180 sd | 0.16 | 0.60 |
| peak_height | 0.111 | 0.109 | −0.013 sd | 0.19 | 0.94 |
| air_s | 0.394 | 0.361 | −0.064 sd | 0.06 | 0.90 |
| displacement | 4.461 | 3.630 | −0.203 sd | 0.16 | 0.90 |

**Every episode differs and no outcome mean moves as much as 0.21 sd.** The
fitted parameters still mean approximately what they meant.

One individual gesture moved a great deal, and it is the informative one: the
tail-press ollie used for figures peaks at **0.275 m instead of 0.746 m**. A
tail 21 mm lower strikes the ground at a shallower pitch, so less rotation is
banked before the impulse. 0.746 m was never a plausible ollie; this is the
geometry correcting a number nobody had questioned.

## What was deliberately NOT done

**Nothing was fitted, and `mean_combined` was not consulted.** It cannot
discriminate a 10.2 in plank from an 8 in deck (the A/B above), so it has no
standing to judge a 6.7% width change. This change rests on the game's own
vertices instead. The corpus A/B belongs in the same commit as a REPLACEMENT
objective, not before one exists.

**No geometry from the game is stored in this repository.** What is committed
is two normalised tables and three ratios — 47 numbers — plus the decoder that
produced them.
