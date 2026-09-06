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

from ..appearance import (DEFAULT_APPEARANCE, AppearancePreset, resolve_appearance,
                          rgb, xyz)
from ..params import SkateParams
from . import deck_profile as dp
from .parks import FLAT_PARK, PARKS  # noqa: F401  (re-exported)


def deck_station(p: SkateParams, t: float) -> tuple[float, float]:
    """(x, z) on the deck's centreline at signed station `t` in [-1, 1].

    The shape comes from `deck_profile`, measured off the game's own mesh.
    Both the visual shell and the collision boxes are built from this one
    function, so they cannot drift apart.
    """
    return dp.station(t, 0.5 * p.deck_length)


# Collision-box boundaries in |t|. Three boxes per kick rather than one: the
# real kick is a progressive CURVE reaching ~21 deg, and a single chord across
# it puts the tail tip 12 mm low, which is a pop-height error, not a cosmetic
# one. Boxes, because MJX cannot collide anything else with a park.
_COL_EDGES = (0.0, 0.55, 0.78, 0.91, 1.0)

# How many points across the top (and again across the bottom) of the visual
# shell's cross-section, and the stations it is swept along. Denser toward the
# tips, where all of the curvature is.
_SHELL_K = 10
_SHELL_T = (0.0, 0.12, 0.24, 0.36, 0.46, 0.54, 0.60, 0.66, 0.71, 0.755,
            0.79, 0.82, 0.85, 0.875, 0.90, 0.92, 0.94, 0.955, 0.968, 0.98,
            0.99, 0.995)


def _shell(p: SkateParams) -> tuple[str, str]:
    """`vertex` and `face` attributes for the visual deck mesh.

    Generated here from the profile tables -- no geometry from the game is
    stored or shipped. The cross-section is a constant-thickness plate carrying
    the measured concave, swept along the centreline and tilted by its local
    slope, and closed at each tip with a fan to an apex vertex.
    """
    hw_max = 0.5 * p.deck_width
    ht = 0.5 * p.deck_thickness
    cc = dp.CONCAVE_FRAC * p.deck_width

    ts = [-t for t in reversed(_SHELL_T[1:])] + list(_SHELL_T)
    verts: list[tuple[float, float, float]] = []
    for t in ts:
        xc, zc = deck_station(p, t)
        pitch = dp.pitch_rad(t)
        sp, cp = math.sin(pitch), math.cos(pitch)
        w = hw_max * dp.half_width_frac(t)
        # The tip is rounded in THREE dimensions, not two. Without this the
        # cross-section keeps full thickness right up to the apex, and because
        # it is tilted by the kick it then pokes out PAST the apex -- the deck
        # ends in a chisel and measures 4 mm longer than it is.
        cap = 1.0
        if abs(t) > 0.96:
            cap = math.sqrt(max(0.0, 1.0 - ((abs(t) - 0.96) / 0.04) ** 2))
        ring = []
        for surface in (+1, -1):                 # top left->right, bottom back
            for k in range(_SHELL_K):
                f = k / (_SHELL_K - 1)
                s = (-1.0 + 2.0 * f) if surface > 0 else (1.0 - 2.0 * f)
                n = (-cc * (1.0 - s * s) + surface * ht) * cap
                ring.append((xc - n * sp, s * w, zc + n * cp))
        verts.extend(ring)

    ring_n = 2 * _SHELL_K
    faces: list[tuple[int, int, int]] = []
    for i in range(len(ts) - 1):
        a, b = i * ring_n, (i + 1) * ring_n
        for j in range(ring_n):
            j2 = (j + 1) % ring_n
            faces.append((a + j, b + j, a + j2))
            faces.append((a + j2, b + j, b + j2))
    for end, ring0 in ((-1.0, 0), (1.0, (len(ts) - 1) * ring_n)):
        xc, zc = deck_station(p, end)
        apex = len(verts)
        verts.append((xc, 0.0, zc))
        for j in range(ring_n):
            j2 = (j + 1) % ring_n
            tri = (ring0 + j, apex, ring0 + j2)
            faces.append(tri if end > 0 else (tri[0], tri[2], tri[1]))

    v = " ".join(f"{c:.5f}" for xyz in verts for c in xyz)
    f = " ".join(str(i) for tri in faces for i in tri)
    return v, f


def deck_asset(p: SkateParams) -> str:
    """Mesh assets for the outline, inset grip and underside skins."""
    v, f = _shell(p)
    top_v, top_f = _skin(p, +1)
    bottom_v, bottom_f = _skin(p, -1)
    return (f'    <mesh name="deck_shell" vertex="{v}" face="{f}"/>\n'
            f'    <mesh name="deck_grip" vertex="{top_v}" face="{top_f}"/>\n'
            f'    <mesh name="deck_underside" vertex="{bottom_v}" face="{bottom_f}"/>\n')


def _skin(p: SkateParams, surface: int) -> tuple[str, str]:
    """One slightly offset deck face, leaving the plywood perimeter exposed.

    ``surface`` is +1 for grip and -1 for the underside.  These are visual
    skins only.  The complete ``deck_shell`` remains the fitting silhouette,
    while the collision boxes remain the only contact-bearing deck geometry.
    """
    hw_max = 0.5 * p.deck_width
    ht = 0.5 * p.deck_thickness
    cc = dp.CONCAVE_FRAC * p.deck_width
    offset = surface * 0.00018
    ts = [-t for t in reversed(_SHELL_T[1:])] + list(_SHELL_T)
    verts: list[tuple[float, float, float]] = []
    for t in ts:
        xc, zc = deck_station(p, t)
        pitch = dp.pitch_rad(t)
        sp, cp = math.sin(pitch), math.cos(pitch)
        w = hw_max * dp.half_width_frac(t)
        cap = 1.0
        if abs(t) > 0.96:
            cap = math.sqrt(max(0.0, 1.0 - ((abs(t) - 0.96) / 0.04) ** 2))
        for k in range(_SHELL_K):
            s = -1.0 + 2.0 * k / (_SHELL_K - 1)
            n = (-cc * (1.0 - s * s) + surface * ht + offset) * cap
            verts.append((xc - n * sp, s * w, zc + n * cp))

    faces: list[tuple[int, int, int]] = []
    for i in range(len(ts) - 1):
        a, b = i * _SHELL_K, (i + 1) * _SHELL_K
        for k in range(_SHELL_K - 1):
            faces.append((a + k, b + k, a + k + 1))
            faces.append((a + k + 1, b + k, b + k + 1))
    for end, row in ((-1.0, 0), (1.0, (len(ts) - 1) * _SHELL_K)):
        xc, zc = deck_station(p, end)
        apex = len(verts)
        verts.append((xc, 0.0, zc))
        for k in range(_SHELL_K - 1):
            tri = (row + k, apex, row + k + 1)
            faces.append(tri if end > 0 else (tri[0], tri[2], tri[1]))
    v = " ".join(f"{c:.5f}" for point in verts for c in point)
    f = " ".join(str(i) for tri in faces for i in tri)
    return v, f


def ride_height(p: SkateParams) -> float:
    """Height of the deck's mid-plane above the ground at rest."""
    return p.wheel_radius + TRUCK_DROP + 0.5 * p.deck_thickness


# Axle centre to deck underside for a standard truck. Measured, not fitted.
#
# CORROBORATED, not replaced, by `truck.bin`: the game's truck puts its axle
# 0.475 of its own axle half-span below the deck face, which at our
# `axle_halfwidth` is 0.0499 m against this 0.053 -- 6% apart. Not close
# enough to move a contact-bearing number on, and it cannot be made closer:
# the truck mesh is stored at a different scale from the deck (its bolt
# rectangle is 36.7 x 63.4 mm at the deck's scale, matching no standard
# pattern), so its absolute size is only recoverable by assuming one. Three
# candidate anchors -- our own axle half-width, and the two axes of an
# old-school bolt pattern -- agree to within 3.5%, which is enough to SHAPE
# the truck and not enough to dimension it.
TRUCK_DROP = 0.053

# Wheel tread half-width, measured on `wheel.bin`: the rim holds full radius
# across 21.4 mm of the wheel's 39.4 mm total, the rest being the bearing
# hub. Visual only -- the colliding wheel is a sphere.
TREAD_HALF_WIDTH = 0.0107

# The visual truck, as fractions of `axle_halfwidth`, read off `truck.bin`.
# Shape only: see TRUCK_DROP above for why these are not dimensions.
_BASEPLATE = (0.522, 0.338, 0.045)      # half along-board, across, thick
_YOKE = (0.300, 0.340, 0.190)           # the hanger's upper body
_HANGER = (0.200, 0.550, 0.170)         # its lower body, at the axle


def _baseplate(p: SkateParams, name: str, sign: int) -> str:
    """The plate bolted to the deck. It belongs to the DECK body, not the
    truck: a real baseplate does not pivot, the hanger does."""
    a, b, c = (v * p.axle_halfwidth for v in _BASEPLATE)
    return f"""
      <geom name="hw_{name}_baseplate" type="ellipsoid"
            size="{a:.6f} {b:.6f} {max(c, 0.004):.6f}"
            pos="{sign * 0.5 * p.wheelbase:.6f} 0 {-0.5 * p.deck_thickness - c:.6f}"
            {_VIS} material="mat_truck_dark"/>
      <geom name="hw_{name}_bolt_l" type="cylinder" size="0.004 0.0015"
            euler="0 0 0"
            pos="{sign * 0.5 * p.wheelbase:.6f} {0.55 * b:.6f} {-0.5 * p.deck_thickness - 0.001:.6f}"
            {_VIS} material="mat_truck"/>
      <geom name="hw_{name}_bolt_r" type="cylinder" size="0.004 0.0015"
            euler="0 0 0"
            pos="{sign * 0.5 * p.wheelbase:.6f} {-0.55 * b:.6f} {-0.5 * p.deck_thickness - 0.001:.6f}"
            {_VIS} material="mat_truck"/>"""


def _hanger_visual(p: SkateParams, name: str) -> str:
    """Yoke, hanger and axle, drawn around the colliding capsule.

    The capsule stays exactly as it was and now sits in the collision group,
    so contact is untouched and the truck stops rendering as a bare bar.
    Geoms are named `hw_` -- they are hardware, and the fitting silhouette
    selects on `vis_`.
    """
    _, _, yc = (v * p.axle_halfwidth for v in _YOKE)
    _, hb, hc = (v * p.axle_halfwidth for v in _HANGER)
    return f"""
        <geom name="hw_{name}_yoke" type="capsule" size="{yc:.6f}"
              fromto="0 0 {-0.010:.6f} 0 0 {-TRUCK_DROP + hc:.6f}"
              {_VIS} material="mat_truck_dark"/>
        <geom name="hw_{name}_hanger" type="capsule" size="{hc:.6f}"
              fromto="0 {-hb:.6f} {-TRUCK_DROP:.6f}
                      0 {hb:.6f} {-TRUCK_DROP:.6f}"
              {_VIS} material="mat_truck"/>
        <geom name="hw_{name}_axle" type="cylinder" size="0.0045 {p.axle_halfwidth:.6f}"
              euler="90 0 0" pos="0 0 {-TRUCK_DROP:.6f}"
              {_VIS} material="mat_truck_dark"/>
        <geom name="hw_{name}_bushing" type="cylinder" size="0.012 0.008"
              pos="0 0 {-TRUCK_DROP + hc + yc:.6f}"
              {_VIS} material="mat_accent_secondary"/>"""

# Attributes shared by every visual-only geom: no contact, no inertia, and
# geom group 2 so `mj_ray` can be told to ignore them.
_VIS = 'contype="0" conaffinity="0" mass="0" group="2"'

# Deck collision boxes sit in group 3, which MuJoCo's renderer hides by
# default. They are coincident with the visual popsicle outline and would
# otherwise draw on top of it, blanking the rounded caps entirely.
_COL = 'group="3"' 


def build_board(p: SkateParams) -> str:
    """The board body, as an MJCF fragment rooted at a free joint."""
    hw, ht = 0.5 * p.deck_width, 0.5 * p.deck_thickness

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
              condim="3" {_COL} material="mat_truck"/>{_hanger_visual(p, name)}
        <body name="{name}_wheel_l" pos="0 {p.axle_halfwidth:.6f} -{TRUCK_DROP:.6f}">
          <joint name="{name}_wheel_l_spin" type="hinge" axis="0 1 0"
                 frictionloss="{p.wheel_frictionloss:.6f}" armature="0.00002"/>
          <geom name="{name}_wheel_l" {_COL} type="sphere" size="{p.wheel_radius:.6f}"
                mass="{p.wheel_mass:.6f}"
                friction="{p.wheel_friction_slide:.6f} {p.wheel_friction_spin:.6f} {p.wheel_friction_roll:.6f}"
                condim="3" {_sol(p)} material="mat_wheel"/>
          <!-- The COLLIDING wheel is a sphere and must stay one: MJX cannot
               collide a cylinder with a box or a mesh. This visual-only
               cylinder is what a wheel looks like; it has contype/conaffinity
               0, so it changes the picture and nothing else. -->
          <geom name="hw_{name}_wheel_l" type="cylinder"
                size="{p.wheel_radius:.6f} {TREAD_HALF_WIDTH:.6f}"
                euler="90 0 0" {_VIS} material="mat_wheel"/>
          <geom name="hw_{name}_wheel_l_hub" type="cylinder"
                size="{0.37 * p.wheel_radius:.6f} {TREAD_HALF_WIDTH + 0.0004:.6f}"
                euler="90 0 0" {_VIS} material="mat_wheel_hub"/>
        </body>
        <body name="{name}_wheel_r" pos="0 -{p.axle_halfwidth:.6f} -{TRUCK_DROP:.6f}">
          <joint name="{name}_wheel_r_spin" type="hinge" axis="0 1 0"
                 frictionloss="{p.wheel_frictionloss:.6f}" armature="0.00002"/>
          <geom name="{name}_wheel_r" {_COL} type="sphere" size="{p.wheel_radius:.6f}"
                mass="{p.wheel_mass:.6f}"
                friction="{p.wheel_friction_slide:.6f} {p.wheel_friction_spin:.6f} {p.wheel_friction_roll:.6f}"
                condim="3" {_sol(p)} material="mat_wheel"/>
          <geom name="hw_{name}_wheel_r" type="cylinder"
                size="{p.wheel_radius:.6f} {TREAD_HALF_WIDTH:.6f}"
                euler="90 0 0" {_VIS} material="mat_wheel"/>
          <geom name="hw_{name}_wheel_r_hub" type="cylinder"
                size="{0.37 * p.wheel_radius:.6f} {TREAD_HALF_WIDTH + 0.0004:.6f}"
                euler="90 0 0" {_VIS} material="mat_wheel_hub"/>
        </body>
      </body>"""

    # The VISUAL deck is one generated mesh, swept along the measured profile.
    # It replaced eleven constant-width boxes plus two ellipsoid tips, which
    # rendered as a staircase in plan view and a banana from the side -- see
    # `deck_profile` for what that cost. Named `vis_deck` because the fitting
    # silhouette selects on the `vis_` prefix, and this IS the outline the
    # physics should be fitted against.
    vis = f"""
      <!-- Group 1 is the canonical fitting silhouette.  RGB skins and art
           stay in group 2 so mask rendering can hide them explicitly. -->
      <geom name="vis_deck" type="mesh" mesh="deck_shell"
            contype="0" conaffinity="0" mass="0" group="1"
            material="mat_plywood"/>
      <geom name="boardfx_grip" type="mesh" mesh="deck_grip" {_VIS}
            material="mat_grip"/>
      <geom name="boardfx_underside" type="mesh" mesh="deck_underside" {_VIS}
            material="mat_deck_underside"/>
      <!-- A long offset register and a short nose tab make yaw, roll and
           nose/tail readable without becoming a deck graphic or a logo. -->
      <geom name="boardfx_register" type="box" size="0.115 0.006 0.00022"
            pos="-0.045 -0.030 0.00205" {_VIS} material="mat_accent"/>
      <geom name="boardfx_nose_tab" type="box" size="0.025 0.017 0.00025"
            pos="0.245 0.027 0.00275" {_VIS} material="mat_accent"/>
      <geom name="boardfx_under_register" type="box" size="0.085 0.005 0.00025"
            pos="0.055 0.032 -0.00645" {_VIS} material="mat_accent_secondary"/>"""

    # COLLISION: boxes following the same centreline. Each spans a segment of
    # the profile; its half-width is the profile's mean over that segment, so
    # the tip box does not collide as if it were full width. Mass is split by
    # segment length, keeping the total exactly `deck_mass`.
    col, spans = "", []
    for lo, hi in zip(_COL_EDGES[:-1], _COL_EDGES[1:]):
        spans.append((lo, hi))
    lengths = [hi - lo for lo, hi in spans]
    total = 2.0 * sum(lengths) - lengths[0]      # the flat box is not mirrored

    def box(name: str, t0: float, t1: float, mass: float) -> str:
        xa, za = deck_station(p, t0)
        xb, zb = deck_station(p, t1)
        seg = math.hypot(xb - xa, zb - za)
        pitch = -math.degrees(math.atan2(zb - za, xb - xa))
        n = 9
        w = hw * sum(dp.half_width_frac(t0 + (t1 - t0) * k / (n - 1))
                     for k in range(n)) / n
        return f"""
      <geom name="{name}" {_COL} type="box"
            size="{0.5 * seg:.6f} {w:.6f} {ht:.6f}"
            pos="{0.5 * (xa + xb):.6f} 0 {0.5 * (za + zb):.6f}"
            euler="0 {pitch:.4f} 0" mass="{mass:.6f}"
            friction="{p.deck_friction_slide:.6f} 0.005 0.0001"
            condim="3" {_sol(p)} rgba="0.20 0.20 0.24 1"/>"""

    lo, hi = spans[0]
    col += box("deck_flat", -hi, hi, p.deck_mass * (2.0 * lengths[0]) / total)
    for i, (lo, hi) in enumerate(spans[1:]):
        m = p.deck_mass * lengths[i + 1] / total
        col += box(f"deck_nose_{i}", lo, hi, m)
        col += box(f"deck_tail_{i}", -lo, -hi, m)

    return f"""
    <body name="deck" pos="0 0 {ride_height(p):.6f}">
      <freejoint name="board"/>{vis}{col}
{_baseplate(p, "front", +1)}{_baseplate(p, "rear", -1)}
{truck("front", +1)}
{truck("rear", -1)}
    </body>"""


def _sol(p: SkateParams) -> str:
    return (f'solref="{p.contact_solref_time:.6f} {p.contact_solref_damp:.6f}" '
            f'solimp="{p.contact_solimp_dmin:.6f} {p.contact_solimp_dmax:.6f} '
            f'{p.contact_solimp_width:.6f}"')


def build_scene(p: SkateParams, park: str = FLAT_PARK,
                appearance: str | AppearancePreset = DEFAULT_APPEARANCE) -> str:
    """Full MJCF document: physics plus one visual-only appearance preset."""
    a = resolve_appearance(appearance)
    return f"""<mujoco model="open_skate_{a.name}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{p.timestep:.6f}" gravity="0 0 -{p.gravity:.6f}"
          integrator="implicitfast" solver="Newton" cone="pyramidal"
          iterations="30" ls_iterations="12"/>
  <visual>
    <!-- Low camera fill plus world-space key/fill lights: form and contact
         shadows stay legible without the headlight flattening everything. -->
    <headlight ambient="{rgb(a.ambient)}" diffuse="0.120 0.123 0.128"
               specular="0.015 0.015 0.015"/>
    <map znear="0.01" zfar="80" shadowclip="10" shadowscale="0.85"/>
    <quality shadowsize="4096" offsamples="8" numslices="28" numstacks="18"/>
    <global offheight="1024" offwidth="1024"/>
  </visual>
  <asset>
{deck_asset(p)}    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="{rgb(a.sky_top)}" rgb2="{rgb(a.sky_horizon)}"
             width="256" height="256"/>
    <!-- Low-contrast slabs: edge marks are expansion joints, tiny checker
         contrast and random grain carry optical flow at 64x128. -->
    <texture name="tex_ground" type="2d" builtin="checker" mark="edge"
             rgb1="{rgb(a.ground_light)}" rgb2="{rgb(a.ground_dark)}"
             markrgb="{rgb(a.joint)}" width="1024" height="1024"/>
    <material name="mat_ground" texture="tex_ground" texrepeat="24 24"
              texuniform="true" specular="0.045" shininess="0.025"
              reflectance="0.01"/>
    <texture name="tex_grip" type="2d" builtin="flat"
             rgb1="{rgb(a.grip_dark)}" rgb2="{rgb(a.grip_light)}"
             width="128" height="512" random="0.22"/>
    <material name="mat_grip" texture="tex_grip" specular="0.02"
              shininess="0.01" reflectance="0.0"/>
    <material name="mat_deck_underside" rgba="{rgb(a.underside)} 1"
              specular="0.10" shininess="0.08"/>
    <material name="mat_plywood" rgba="{rgb(a.plywood)} 1"
              specular="0.07" shininess="0.04"/>
    <material name="mat_truck" rgba="{rgb(a.truck)} 1" specular="0.58"
              shininess="0.48" reflectance="0.06"/>
    <material name="mat_truck_dark" rgba="0.16 0.18 0.20 1" specular="0.35"
              shininess="0.28"/>
    <material name="mat_wheel" rgba="{rgb(a.wheel)} 1" specular="0.16"
              shininess="0.10"/>
    <material name="mat_wheel_hub" rgba="0.25 0.27 0.28 1" specular="0.40"
              shininess="0.32"/>
    <texture name="tex_concrete" type="2d" builtin="flat"
             rgb1="{rgb(a.concrete_light)}" rgb2="{rgb(a.concrete_dark)}"
             width="256" height="256" random="0.105"/>
    <material name="mat_concrete" texture="tex_concrete" texrepeat="4 4"
              texuniform="true" specular="0.055" shininess="0.025"/>
    <texture name="tex_ledge" type="2d" builtin="flat"
             rgb1="{rgb(a.ledge_light)}" rgb2="{rgb(a.ledge_dark)}"
             width="256" height="256" random="0.08"/>
    <material name="mat_ledge" texture="tex_ledge" texrepeat="2 2"
              texuniform="true" specular="0.11" shininess="0.09"/>
    <material name="mat_rail" rgba="0.18 0.23 0.26 1" specular="0.62"
              shininess="0.58" reflectance="0.10"/>
    <material name="mat_accent" rgba="{rgb(a.accent)} 1" specular="0.09"
              shininess="0.06"/>
    <material name="mat_accent_secondary" rgba="{rgb(a.accent_secondary)} 1"
              specular="0.09" shininess="0.06"/>
    <material name="mat_scuff" rgba="0.38 0.39 0.39 0.28"
              specular="0" shininess="0"/>
    <material name="mat_environment" rgba="0.10 0.11 0.13 1"
              specular="0.03" shininess="0.02"/>
  </asset>
  <worldbody>
    <light name="key" pos="{xyz(a.key_pos)}" dir="{xyz(a.key_dir)}"
           directional="true" castshadow="true"
           diffuse="{rgb(a.key_diffuse)}" specular="{rgb(a.key_specular)}"/>
    <light name="fill" pos="{xyz(a.fill_pos)}" dir="{xyz(a.fill_dir)}"
           directional="true" castshadow="false"
           diffuse="{rgb(a.fill_diffuse)}" specular="0 0 0"/>
    <!-- The chase camera, as a model element. The CPU renderer drives a free
         camera from `sim/camera.py` and ignores this one, but MJX's batch
         renderer can only render cameras that exist in the model, and its
         pose is written into `cam_xpos`/`cam_xmat` per world. The fovy is the
         fitted vertical field of view so both paths frame the board alike. -->
         The resolution is NOT optional and NOT cosmetic: MJX's batch
         renderer takes its image size from the CAMERA, not from
         `vis.global_.offwidth/offheight`, and an unset resolution renders
         1x1 images. That failure is invisible in every summary statistic --
         a 1x1 render still produces well-formed frames at a plausible rate.

         The camera rides a MOCAP body. Writing `cam_xpos`/`cam_xmat` directly
         does not work: those are OUTPUTS, recomputed from the model every time
         kinematics runs, so a chase camera written that way silently reverts
         to its MJCF pose. Measured -- the board was in frame 0 of every
         episode and in the last frame of 3%, including episodes that never
         moved 2 m. Mocap pose is an INPUT and survives. -->
    <body name="cam_mount" mocap="true" pos="0 0 1">
      <camera name="chase" mode="fixed" fovy="{p.cam_fov_deg:.4f}"
              resolution="{p.render_width} {p.render_height}"
              pos="0 0 0" quat="1 0 0 0"/>
    </body>
{a.environment}
{park}
{build_board(p)}
  </worldbody>
</mujoco>
"""
