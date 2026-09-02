"""Build the Open Skate MJCF from a `SkateParams`.

The model is generated rather than stored as a static .xml because system
identification varies geometry-dependent quantities (wheelbase, kingpin
angle, masses), and those change the compiled inertia — so the model has to be
recompiled per candidate, not patched in place. Compilation costs ~1 ms, and
happens once per CMA-ES sample rather than once per step, so this is free.

MJX CONSTRAINTS ARE LOAD-BEARING HERE (see tests/test_mjx_compat.py):
  * wheels are SPHEREs, never cylinders — MJX-JAX cannot collide a cylinder
    with a box or mesh, and every park surface is a box;
  * collision geometry is planes, boxes and capsules only;
  * solver Newton, integrator implicitfast, condim 3.
Breaking any of these still runs fine on CPU and silently fails to port.
"""
from __future__ import annotations

import math

from ..params import SkateParams
from .parks import FLAT_PARK, PARKS  # noqa: F401  (re-exported)


def _deck_station(p: SkateParams, u: float, flat_half: float,
                  kick_len: float, ka: float) -> tuple[float, float]:
    """(x, z) of a point at normalised station `u` along the deck's centreline.

    u = 0 is the middle, +/-1 the tips. Follows the flat, then rises along the
    kick, so the visual outline tracks the same shape the collision boxes have.
    """
    half = 0.5 * p.deck_length
    x = u * half
    if abs(x) <= flat_half:
        return x, 0.0
    over = abs(x) - flat_half
    return (math.copysign(flat_half + over * math.cos(ka), x),
            over * math.sin(ka))


def _kick(p: SkateParams) -> tuple[float, float]:
    """(flat half-length, kick section length) along the deck's long axis."""
    flat_half = 0.5 * p.deck_length * p.flat_fraction
    kick_len = 0.5 * p.deck_length - flat_half
    return flat_half, kick_len


def ride_height(p: SkateParams) -> float:
    """Height of the deck's mid-plane above the ground at rest."""
    return p.wheel_radius + TRUCK_DROP + 0.5 * p.deck_thickness


# Axle centre to deck underside for a standard truck. Measured, not fitted.
TRUCK_DROP = 0.053

# Attributes shared by every visual-only geom: no contact, no inertia, and
# geom group 2 so `mj_ray` can be told to ignore them.
_VIS = 'contype="0" conaffinity="0" mass="0" group="2"'

# Deck collision boxes sit in group 3, which MuJoCo's renderer hides by
# default. They are coincident with the visual popsicle outline and would
# otherwise draw on top of it, blanking the rounded caps entirely.
_COL = 'group="3"' 


def build_board(p: SkateParams) -> str:
    """The board body, as an MJCF fragment rooted at a free joint."""
    flat_half, kick_len = _kick(p)
    hw, ht = 0.5 * p.deck_width, 0.5 * p.deck_thickness
    ka = math.radians(p.kick_angle_deg)
    # Centre of each kick section, in the deck frame: it starts at the end of
    # the flat and rises along the kick angle.
    kx = flat_half + 0.5 * kick_len * math.cos(ka)
    kz = 0.5 * kick_len * math.sin(ka)

    # Split deck mass by section length so the ends are not artificially heavy.
    flat_mass = p.deck_mass * p.flat_fraction
    kick_mass = 0.5 * p.deck_mass * (1.0 - p.flat_fraction)

    ang = math.radians(p.kingpin_angle_deg)
    cx, cz = math.cos(ang), math.sin(ang)
    lim = p.truck_limit_deg
    tz = -ht  # trucks mount to the deck's underside

    def truck(name: str, sign: int) -> str:
        # Front and rear kingpins tilt in opposite senses. That mirror is the
        # whole reason a lean turns the board instead of just leaning it: the
        # two axles steer in opposite directions, describing an arc.
        axis = f"{sign * cx:.6f} 0 {cz:.6f}"
        return f"""
      <body name="{name}_truck" pos="{sign * 0.5 * p.wheelbase:.6f} 0 {tz:.6f}">
        <joint name="{name}_steer" type="hinge" axis="{axis}" pos="0 0 0"
               stiffness="{p.truck_stiffness:.6f}" damping="{p.truck_damping:.6f}"
               limited="true" range="-{lim:.3f} {lim:.3f}" armature="0.0008"/>
        <geom name="{name}_hanger" type="capsule" size="0.010 {p.axle_halfwidth - 0.012:.6f}"
              fromto="0 -{p.axle_halfwidth - 0.012:.6f} -{TRUCK_DROP:.6f}
                      0 {p.axle_halfwidth - 0.012:.6f} -{TRUCK_DROP:.6f}"
              mass="{p.truck_mass:.6f}" friction="0.35 0.005 0.0001"
              condim="3" rgba="0.72 0.72 0.75 1"/>
        <body name="{name}_wheel_l" pos="0 {p.axle_halfwidth:.6f} -{TRUCK_DROP:.6f}">
          <joint name="{name}_wheel_l_spin" type="hinge" axis="0 1 0"
                 frictionloss="{p.wheel_frictionloss:.6f}" armature="0.00002"/>
          <geom name="{name}_wheel_l" type="sphere" size="{p.wheel_radius:.6f}"
                mass="{p.wheel_mass:.6f}"
                friction="{p.wheel_friction_slide:.6f} {p.wheel_friction_spin:.6f} {p.wheel_friction_roll:.6f}"
                condim="3" {_sol(p)} rgba="0.92 0.90 0.84 1"/>
        </body>
        <body name="{name}_wheel_r" pos="0 -{p.axle_halfwidth:.6f} -{TRUCK_DROP:.6f}">
          <joint name="{name}_wheel_r_spin" type="hinge" axis="0 1 0"
                 frictionloss="{p.wheel_frictionloss:.6f}" armature="0.00002"/>
          <geom name="{name}_wheel_r" type="sphere" size="{p.wheel_radius:.6f}"
                mass="{p.wheel_mass:.6f}"
                friction="{p.wheel_friction_slide:.6f} {p.wheel_friction_spin:.6f} {p.wheel_friction_roll:.6f}"
                condim="3" {_sol(p)} rgba="0.92 0.90 0.84 1"/>
        </body>
      </body>"""

    # A faithful VISUAL outline, separate from the collision boxes.
    #
    # True Skate's deck is a popsicle: widest at the middle, narrowing toward
    # both tips. Measured off real frames the width profile rises to a peak of
    # ~0.40 (width/length) and falls to ~0.19 at the ends, where a
    # constant-width outline renders nearly flat and reads too blunt. The
    # outline is therefore built as a run of slabs whose half-width follows a
    # taper, plus rounded tips.
    #
    # Collision keeps the three plain boxes: cheap, MJX-safe, same tip, so pop
    # mechanics are unchanged. Visual geoms carry contype=0/conaffinity=0 and
    # mass=0, and sit in geom group 2 so ray casts can exclude them.
    n_slab = 11
    vis = ""
    for i in range(n_slab):
        u0, u1 = -1.0 + 2.0 * i / n_slab, -1.0 + 2.0 * (i + 1) / n_slab
        um = 0.5 * (u0 + u1)
        # Half-width at this station, tapering from the centre to the tips.
        t = abs(um) ** p.deck_taper_power
        hw_i = hw * (1.0 - (1.0 - p.deck_tip_width_frac) * t)
        # Position along the deck, following the flat then the kick.
        xa, za = _deck_station(p, u0, flat_half, kick_len, ka)
        xb, zb = _deck_station(p, u1, flat_half, kick_len, ka)
        cxs, czs = 0.5 * (xa + xb), 0.5 * (za + zb)
        seg_len = math.hypot(xb - xa, zb - za)
        pitch = -math.degrees(math.atan2(zb - za, xb - xa))
        vis += f"""
      <geom name="vis_{i}" type="box"
            size="{0.5 * seg_len:.6f} {hw_i:.6f} {ht:.6f}"
            pos="{cxs:.6f} 0 {czs:.6f}" euler="0 {pitch:.4f} 0"
            {_VIS} rgba="0.21 0.20 0.23 1"/>"""
    # Rounded tips.
    for name, sgn in (("nose", 1), ("tail", -1)):
        xt, zt = _deck_station(p, sgn * 1.0, flat_half, kick_len, ka)
        r_tip = hw * p.deck_tip_width_frac
        vis += f"""
      <geom name="vis_{name}_tip" type="ellipsoid"
            size="{r_tip:.6f} {r_tip:.6f} {ht:.6f}"
            pos="{xt - sgn * r_tip * math.cos(ka):.6f} 0 {zt - sgn * 0.0:.6f}"
            euler="0 {-sgn * p.kick_angle_deg:.4f} 0"
            {_VIS} rgba="0.21 0.20 0.23 1"/>"""

    return f"""
    <body name="deck" pos="0 0 {ride_height(p):.6f}">
      <freejoint name="board"/>{vis}
      <geom name="deck_flat" {_COL} type="box" size="{flat_half:.6f} {hw:.6f} {ht:.6f}"
            pos="0 0 0" mass="{flat_mass:.6f}"
            friction="{p.deck_friction_slide:.6f} 0.005 0.0001"
            condim="3" {_sol(p)} rgba="0.20 0.20 0.24 1"/>
      <geom name="deck_nose" {_COL} type="box" size="{0.5 * kick_len:.6f} {hw:.6f} {ht:.6f}"
            pos="{kx:.6f} 0 {kz:.6f}" euler="0 -{p.kick_angle_deg:.4f} 0"
            mass="{kick_mass:.6f}"
            friction="{p.deck_friction_slide:.6f} 0.005 0.0001"
            condim="3" {_sol(p)} rgba="0.24 0.20 0.20 1"/>
      <geom name="deck_tail" {_COL} type="box" size="{0.5 * kick_len:.6f} {hw:.6f} {ht:.6f}"
            pos="-{kx:.6f} 0 {kz:.6f}" euler="0 {p.kick_angle_deg:.4f} 0"
            mass="{kick_mass:.6f}"
            friction="{p.deck_friction_slide:.6f} 0.005 0.0001"
            condim="3" {_sol(p)} rgba="0.24 0.20 0.20 1"/>
{truck("front", +1)}
{truck("rear", -1)}
    </body>"""


def _sol(p: SkateParams) -> str:
    return (f'solref="{p.contact_solref_time:.6f} {p.contact_solref_damp:.6f}" '
            f'solimp="{p.contact_solimp_dmin:.6f} {p.contact_solimp_dmax:.6f} '
            f'{p.contact_solimp_width:.6f}"')


def build_scene(p: SkateParams, park: str = FLAT_PARK) -> str:
    """Full MJCF document: options, the park, and the board."""
    return f"""<mujoco model="open_skate">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{p.timestep:.6f}" gravity="0 0 -{p.gravity:.6f}"
          integrator="implicitfast" solver="Newton" cone="pyramidal"
          iterations="30" ls_iterations="12"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/>
    <map znear="0.01" zfar="80"/>
    <!-- The offscreen framebuffer defaults to 640x480, which caps every
         render. Silhouette fitting runs small, but figures and inspection
         want the real capture's portrait shape. -->
    <global offheight="1024" offwidth="1024"/>
  </visual>
  <worldbody>
    <light pos="2 -2 4" dir="-0.4 0.4 -1" directional="true"/>
{park}
{build_board(p)}
  </worldbody>
</mujoco>
"""
