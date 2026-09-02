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

## Status

See the build plan at `~/.claude/plans/concurrent-sprouting-sifakis.md`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```
