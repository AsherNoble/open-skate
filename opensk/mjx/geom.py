"""Analytic ray casts against the deck, replacing `mj_ray`.

`sim/touch.py` calls `mj_ray` on every physics substep to decide whether a
finger is on the deck or the ground, and to let a finger re-acquire the board
mid-gesture. MJX-JAX's general ray support is documented as slow, and a general
BVH query is wildly over-powered for what is actually being asked: does a ray
hit one of three known boxes, or the ground plane?

Both answers are closed-form. This module is the exact, vectorisable
replacement, validated against `mj_ray` itself in `tests/test_mjx_geom.py`.

Written with plain numpy semantics so it runs unchanged under `jax.numpy` —
no data-dependent branching, no early returns, so it survives `jit` and `vmap`.
"""
from __future__ import annotations

import numpy as np

# Sentinel for "no hit". Finite so arithmetic stays well-behaved under jit;
# large enough that any real hit wins a minimum.
NO_HIT = 1.0e9


def ray_plane(origin, direction, height: float = 0.0, xp=np):
    """Distance along `direction` to the horizontal plane z = height.

    Returns NO_HIT when the ray points away from, or runs parallel to, the
    plane. Never returns a negative distance: a plane behind the camera is not
    a hit.
    """
    dz = direction[..., 2]
    safe = xp.where(xp.abs(dz) < 1e-12, 1e-12, dz)
    t = (height - origin[..., 2]) / safe
    return xp.where((xp.abs(dz) < 1e-12) | (t <= 0.0), NO_HIT, t)


def ray_obb(origin, direction, box_pos, box_mat, box_half, xp=np):
    """Distance to an oriented box, by the slab method in the box's own frame.

    `box_mat` is the box's 3x3 rotation with columns as its axes, so the
    world->box transform is its transpose. Branchless: the miss case falls out
    of the slab comparison rather than an early return, which is what lets this
    run under vmap.
    """
    o = box_mat.T @ (origin - box_pos)
    d = box_mat.T @ direction
    safe = xp.where(xp.abs(d) < 1e-12, 1e-12, d)
    t1 = (-box_half - o) / safe
    t2 = (box_half - o) / safe
    tmin = xp.max(xp.minimum(t1, t2))
    tmax = xp.min(xp.maximum(t1, t2))
    # A ray parallel to a slab misses unless it already lies inside it.
    parallel_outside = xp.any((xp.abs(d) < 1e-12) & (xp.abs(o) > box_half))
    hit = (tmax >= xp.maximum(tmin, 0.0)) & (~parallel_outside)
    t = xp.where(tmin > 0.0, tmin, tmax)   # from inside the box, use the exit
    return xp.where(hit & (t > 0.0), t, NO_HIT)


# Anything at or past this distance is a miss, not a hit. Compared against
# rather than tested for equality because the slab arithmetic can perturb the
# sentinel.
MISS_THRESHOLD = 0.5 * NO_HIT


def cast_deck_or_ground(origin, direction, deck_boxes, ground_height=0.0, xp=np):
    """(hit_deck, distance, missed) for a screen ray.

    `deck_boxes` is a sequence of (pos, mat, half) for the deck's collision
    boxes. The deck wins only if it is genuinely nearer than the ground, so a
    ray passing over the board and landing beyond it is correctly a ground hit.

    `missed` -- hit nothing at all, a ray into the sky -- is a THIRD outcome and
    not a kind of ground hit. `mj_ray` reports it as a negative distance and the
    reference model makes such a finger permanently inert. Folding it into
    "ground" instead puts the contact point 1e9 m away, and the lever arm from
    the board's centre of mass then generates an arbitrarily large torque. That
    was measured: boards launched 6.5 m into the air and trajectories went NaN.
    """
    t_deck = NO_HIT
    for pos, mat, half in deck_boxes:
        t_deck = xp.minimum(t_deck, ray_obb(origin, direction, pos, mat, half, xp=xp))
    t_ground = ray_plane(origin, direction, ground_height, xp=xp)
    hit_deck = t_deck < t_ground
    t = xp.where(hit_deck, t_deck, t_ground)
    missed = t >= MISS_THRESHOLD
    # Clamp a miss to zero distance so no downstream arithmetic ever sees 1e9;
    # `missed` is what callers must branch on.
    return hit_deck & (~missed), xp.where(missed, 0.0, t), missed


def deck_boxes_from(sim) -> list:
    """Extract (pos, mat, half) for the deck's collision boxes from a SkateSim.

    Reads live geom poses, so it reflects the board wherever it currently is.
    """
    out = []
    for gid in sorted(sim._deck_gids):
        out.append((sim.data.geom_xpos[gid].copy(),
                    sim.data.geom_xmat[gid].reshape(3, 3).copy(),
                    sim.model.geom_size[gid].copy()))
    return out
