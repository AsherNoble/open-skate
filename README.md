# Open Skate

A physics simulator whose dynamics are **fitted to the real True Skate**, so that
gestures, policies and world models transfer back to a phone running the actual game.

It exists to break a throughput ceiling. Its sibling project `TrueSkate-AI` drives a
physical iPhone over Appium at 1x real-time — about 15K steps/hour, two phones wide.
Open Skate runs the same gestures against MuJoCo on CPU, and against MJX on GPU at
tens of thousands of environments in parallel.

**Success is transfer, not fun.** A gesture optimised in Open Skate has to land the
same trick on the phone. Everything here is arranged around measuring that.

## Design commitments

- **The action space is True Skate's, not an invented one.** The sim consumes the exact
  gesture recipe schema from `TrueSkate-AI/GESTURES.md` — normalised screen waypoints,
  `duration`, `easing_power`, `delays`, optional `spin` hold. So `trick_libraries/*.json`
  replays here unchanged, and a solution found here executes on the phone unchanged.
- **Tricks are emergent.** There is no ollie code, no kickflip code, no carving code.
  The deck's ends are collidable and the trucks steer off a tilted kingpin hinge; the
  pop and the carve fall out of contact and geometry. This is why the physics parameters
  are worth fitting at all.
- **Everything MJX can't do is banned up front**, and tested: sphere wheels (never
  cylinders), box/plane/capsule collision geometry only, Newton + implicitfast + condim 3.
- **`sim/`, `game/` and `fit/objective.py` are pure** — no wall clock, no rendering, no
  I/O, fixed timestep, seeded RNG. Rollouts are bitwise reproducible.
- **RGB geometry cannot affect the board.** Render-only deck skins, hardware and park
  copies have zero mass, no collision affinity and their own render groups. Hidden
  plane/box/capsule collision geometry remains authoritative, and silhouette fitting
  renders the deck outline alone.

## Appearance and local previews

`SkateSim(appearance=...)` and pixel `GestureEnv(appearance=...)` accept `"day"`,
`"indoor"` or `"overcast"`. The presets change procedural materials, sky and lights;
they do not change `SkateParams`, the camera, collision geometry or trajectories.
Training remains 64x128. Render all three presets at a device-like 375x812 locally:

```bash
.venv/bin/python -m opensk.pose.preview --output-dir results/visual_direction
```

Use `--mode training` to exercise the exact training raster, or `--park flat|plaza|sls`
to choose collision geometry. The default compact plaza is composed for the existing
fitted portrait chase camera; no projection or camera parameter is adjusted.

## Status

See the build plan at `~/.claude/plans/concurrent-sprouting-sifakis.md`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```
