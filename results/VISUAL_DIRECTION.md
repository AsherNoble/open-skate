# Blender-directed neutral appearance pass

Implemented from the supplied day-plaza, indoor-hall and overcast-civic Blender
mocks as art direction. No Blender mesh, generated image, logo, branded park, game
texture or copyrighted asset is shipped. Runtime appearance is repository-owned
MJCF geometry plus MuJoCo procedural textures and materials.

## What changed

- A single fitted deck outline remains the silhouette source. Thin top and bottom
  render skins provide matte grip and a muted underside, while the exposed shell is
  warm plywood. An offset line and short nose tab make rotation and direction legible
  without constituting a graphic or logo.
- Visual-only rounded baseplates, hanger/yoke capsules, axles, bushings, cylindrical
  urethane treads and hub centres make hardware readable. The original physical
  hanger and spherical wheel contacts are unchanged and hidden from RGB.
- Physical park planes, boxes and capsules now have coincident `parkvis_` copies.
  Only the copies render; every copy and accent has `mass=0`, `contype=0` and
  `conaffinity=0`. Concrete uses fine low-contrast procedural aggregate, sparse
  irregular expansion joints, broad tonal repairs and restrained scuffs instead
  of the previous large checkerboard.
- The new compact `plaza` park arranges an unbranded ledge, flat rail, bank and
  ascending stairs inside the useful forward field of the existing fitted camera.
- Day, indoor and overcast swap coherent palettes, haze and key/fill lighting. A
  tighter high-resolution shadow volume strengthens contact shadows without adding
  extra shadow-map passes. Indoor adds a layered visual-only back wall, windows,
  dado, columns and beams rather than a near black backdrop; day and overcast add
  distant abstract civic massing.
- `SceneRenderer.for_mode(..., "training")` is the existing 64x128 raster.
  `"preview"` is 375x812 with the same projection and camera state.

## Invariance evidence

Before implementation, 1,200-step forced trajectories were recorded on both the
existing flat and SLS parks. After the complete geometry/material split, both arrays
were `numpy.array_equal` with maximum absolute difference `0.0` and the same combined
SHA-256:

```
e167dd8fb7f60f2d1eb18f70676fb65a4f13c16b1bfa78c277d61441e408eaf7
```

Tests additionally compare every contact-bearing geom field, compiled body mass,
inertia and armature across the three appearances, then require 700-step forced
trajectories to be bit-identical. RGB regression goldens run at 64x128. The board mask
is rendered with a group that excludes all RGB skins and hardware, and is required to
be identical across presets.

## Local renders

- `visual_direction/before_after_day.png` — prior appearance beside the final day plaza.
- `visual_direction/appearance_presets.png` — day, indoor and overcast from one state.
- `visual_direction/park_appearance_matrix.png` — the full 3x3 matrix in flat/plaza/SLS
  row order and day/indoor/overcast column order; each source tile remains 375x812.
- `visual_direction/training_samples.png` — the same 3x3 matrix at the exact 64x128
  training resolution, without rescaling before assembly.
- `visual_direction/after_hardware_detail.png` — rolled-board inspection of underside,
  trucks, axles, hubs and wheels.

The individual `preview_<park>_<appearance>.png` and
`training_<park>_<appearance>.png` files make every domain independently inspectable.
They can be regenerated without Modal or another service:

```bash
.venv/bin/python -m opensk.pose.preview --all-parks --mode preview \
  --prefix preview --contact-sheet results/visual_direction/park_appearance_matrix.png
.venv/bin/python -m opensk.pose.preview --all-parks --mode training \
  --prefix training --contact-sheet results/visual_direction/training_samples.png
```

## Honest comparison and cost

The result carries the mocks' neutral deck hierarchy, readable hardware, muted
concrete, modular obstacles and three coherent lighting domains. MuJoCo's classic
renderer does not reproduce Blender's area-light penumbra, bevel shader or material
response: the contact shadows are stronger and antialiased, but still visibly more
graphic than the Blender reference. The fitted camera's steep downward pitch also
means the stationary indoor frame is predominantly polished floor; the hall envelope
enters the chase view as the board advances. SLS likewise retains its authoritative
wide course, whose first obstacles are beyond the initial stationary crop. Neither
limitation was hidden by changing the fitted camera or collision layout.

On this Apple Silicon host, five 100-frame 375x812 trials measured a median 147.5 FPS
for the final day-plaza renderer versus 172.1 FPS at commit `7738d60` (14.3% slower).
Five 300-frame 64x128 trials measured 169.7 FPS versus 195.4 FPS (13.2% slower). The
final plaza RGB model has 92 geoms versus 68; both have two lights and one shadow map.
Pose-only construction strips every new environment/detail geom, all lights, assets
and camera bodies, and the bit-identical trajectory tests remain the relevant
performance/invariance guard there. The optional MJX Warp renderer is not installed
on this host, so no claim is made about its batch throughput.
