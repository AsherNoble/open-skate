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
