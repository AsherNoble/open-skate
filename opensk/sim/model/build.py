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
    """The `<mesh>` asset for the visual deck shell."""
    v, f = _shell(p)
    return f'    <mesh name="deck_shell" vertex="{v}" face="{f}"/>\n'


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
              condim="3" material="mat_truck"/>
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
                size="{p.wheel_radius:.6f} 0.011"
                euler="90 0 0" {_VIS} material="mat_wheel"/>
        </body>
        <body name="{name}_wheel_r" pos="0 -{p.axle_halfwidth:.6f} -{TRUCK_DROP:.6f}">
          <joint name="{name}_wheel_r_spin" type="hinge" axis="0 1 0"
                 frictionloss="{p.wheel_frictionloss:.6f}" armature="0.00002"/>
          <geom name="{name}_wheel_r" {_COL} type="sphere" size="{p.wheel_radius:.6f}"
                mass="{p.wheel_mass:.6f}"
                friction="{p.wheel_friction_slide:.6f} {p.wheel_friction_spin:.6f} {p.wheel_friction_roll:.6f}"
                condim="3" {_sol(p)} material="mat_wheel"/>
          <geom name="hw_{name}_wheel_r" type="cylinder"
                size="{p.wheel_radius:.6f} 0.011"
                euler="90 0 0" {_VIS} material="mat_wheel"/>
        </body>
      </body>"""

    # The VISUAL deck is one generated mesh, swept along the measured profile.
    # It replaced eleven constant-width boxes plus two ellipsoid tips, which
    # rendered as a staircase in plan view and a banana from the side -- see
    # `deck_profile` for what that cost. Named `vis_deck` because the fitting
    # silhouette selects on the `vis_` prefix, and this IS the outline the
    # physics should be fitted against.
    vis = f"""
      <geom name="vis_deck" type="mesh" mesh="deck_shell" {_VIS}
            material="mat_grip"/>"""

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
    <!-- The headlight is a camera-mounted lamp: it lights every surface
         head-on, which flattens exactly the shading a world model could use
         to read orientation. Dropped to a low ambient fill so the SUN does
         the modelling and the board casts a shadow that says where it is
         relative to the ground -- the one depth cue a single 2D frame has. -->
    <headlight ambient="0.20 0.21 0.23" diffuse="0.12 0.12 0.13"
               specular="0.04 0.04 0.04"/>
    <map znear="0.01" zfar="80" shadowclip="6" shadowscale="1.2"/>
    <quality shadowsize="4096" offsamples="8"/>
    <!-- The offscreen framebuffer defaults to 640x480, which caps every
         render. Silhouette fitting runs small, but figures and inspection
         want the real capture's portrait shape. -->
    <global offheight="1024" offwidth="1024"/>
  </visual>
  <!-- APPEARANCE. Everything up to now was fitted against SILHOUETTES, where
       only shape matters and colour is discarded, so the render was never
       given any. That was fine while masks were the observation and stops
       being fine the moment pixels are: a world model trained on flat grey
       primitives has no chance against real True Skate frames. This is the
       cheap half of Phase 4 -- procedural textures and materials, no meshes,
       nothing that could upset MJX (textures and materials are visual-only). -->
  <asset>
{deck_asset(p)}    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.52 0.60 0.72" rgb2="0.86 0.89 0.93" width="256" height="256"/>
    <texture name="tex_ground" type="2d" builtin="checker" mark="cross"
             rgb1="0.60 0.60 0.58" rgb2="0.56 0.56 0.55"
             markrgb="0.66 0.66 0.64" width="512" height="512"/>
    <material name="mat_ground" texture="tex_ground" texrepeat="18 18"
              texuniform="true" specular="0.05" shininess="0.02"
              reflectance="0.02"/>
    <!-- Grip tape: near-black and matte, which is what it looks like from the
         chase camera. The noise is what stops a flat deck reading as a hole. -->
    <texture name="tex_grip" type="2d" builtin="flat" rgb1="0.10 0.10 0.11"
             rgb2="0.13 0.13 0.15" width="128" height="512" random="0.28"/>
    <material name="mat_grip" texture="tex_grip" specular="0.02"
              shininess="0.01" reflectance="0.0"/>
    <material name="mat_truck" rgba="0.74 0.75 0.78 1" specular="0.55"
              shininess="0.55" reflectance="0.08"/>
    <material name="mat_wheel" rgba="0.93 0.92 0.88 1" specular="0.18"
              shininess="0.12"/>
    <!-- Concrete, with grain. A flat fill reads as plastic and, more to the
         point, gives a moving camera nothing to parallax against. -->
    <texture name="tex_concrete" type="2d" builtin="flat"
             rgb1="0.78 0.77 0.74" rgb2="0.71 0.70 0.68"
             width="256" height="256" random="0.15"/>
    <material name="mat_concrete" texture="tex_concrete" texrepeat="3 3"
              texuniform="true" specular="0.06" shininess="0.03"/>
    <texture name="tex_ledge" type="2d" builtin="flat"
             rgb1="0.86 0.85 0.81" rgb2="0.80 0.79 0.76"
             width="256" height="256" random="0.12"/>
    <material name="mat_ledge" texture="tex_ledge" texrepeat="2 2"
              texuniform="true" specular="0.14" shininess="0.12"/>
    <material name="mat_rail" rgba="0.80 0.81 0.84 1" specular="0.7"
              shininess="0.7" reflectance="0.15"/>
    <!-- The contest flat. It was a FLAT yellow, and at the chase camera's
         framing that is almost the whole observation: a black board on an
         untextured field, with no optical flow to read translation from. The
         board-locked camera already hides translation in the silhouette (see
         the corpus note: 2 m of travel gives a bit-identical mask); an
         untextured floor hides it in the pixels too. The slab tiling is the
         signal that a world model can use to tell moving from still. -->
    <texture name="tex_plaza" type="2d" builtin="checker" mark="edge"
             rgb1="0.87 0.75 0.19" rgb2="0.83 0.71 0.17"
             markrgb="0.70 0.60 0.15" width="512" height="512"/>
    <material name="mat_plaza" texture="tex_plaza" texrepeat="4 4"
              texuniform="true" specular="0.05" shininess="0.03"
              reflectance="0.02"/>
  </asset>
  <worldbody>
    <!-- Sun and fill. A single unshadowed lamp gave the board no contact
         shadow at all, so a board resting on the flat and a board hovering
         0.3 m above it rendered identically. -->
    <light name="sun" pos="3 -4 6" dir="-0.35 0.45 -1" directional="true"
           castshadow="true" diffuse="0.72 0.71 0.68"
           specular="0.22 0.22 0.22"/>
    <light name="fill" pos="-5 4 5" dir="0.5 -0.4 -1" directional="true"
           castshadow="false" diffuse="0.20 0.21 0.24" specular="0 0 0"/>
    <!-- The chase camera, as a model element. The CPU renderer drives a free
         camera from `sim/camera.py` and ignores this one, but MJX's batch
         renderer can only render cameras that exist in the model, and its
         pose is written into `cam_xpos`/`cam_xmat` per world. The fovy is the
         fitted vertical field of view so both paths frame the board alike. -->
         The `resolution` is NOT optional and NOT cosmetic: MJX's batch
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
{park}
{build_board(p)}
  </worldbody>
</mujoco>
"""
