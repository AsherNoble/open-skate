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
  `conaffinity=0`. Concrete uses low-contrast procedural slabs plus sparse scuffs.
- The new compact `plaza` park arranges an unbranded ledge, flat rail, bank and
  ascending stairs inside the useful forward field of the existing fitted camera.
- Day, indoor and overcast swap coherent palettes, sky and key/fill lighting. Indoor
  adds a dark visual-only hall envelope. All use soft contact shadows.
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
- `visual_direction/after_hardware_detail.png` — rolled-board inspection of underside,
  trucks, axles, hubs and wheels.
