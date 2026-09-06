"""Skatepark geometry, as MJCF fragments.

Built in the idiom of an SLS contest course — the parks the capture rig runs on
(SLS 2015 Los Angeles on XR1, SLS 2016 Super Crown in the trick captures) plus
Skateboard GB 2024: a central flat run, a stair set with a handrail and a hubba
ledge down one side, a funbox with banks, a flat bar, a manual pad, and a
quarter pipe.

EVERY COLLIDABLE GEOM IS A PLANE, BOX OR CAPSULE. That is not stylistic. MJX
cannot collide a cylinder or an ellipsoid with a box or a mesh, and a park made
of boxes is exactly what the wheels have to roll on, so a cylinder coping would
work on CPU and silently fall through on GPU. Curved transitions are therefore
faceted from boxes rather than modelled as cylinders, and rails are capsules,
which collide with everything.

Dimensions are in metres and are real skatepark sizes: 0.16 m risers, 0.30 m
treads, a 0.30 m flat bar, a 1.4 m quarter pipe.  Every physical primitive has
an exactly coincident ``parkvis_`` copy for RGB.  The physical primitive stays
in hidden group 3; the copy has zero mass and no contact.  This lets art
direction evolve without turning a bevel, paint stripe or prettier rail into a
new physical claim.
"""
from __future__ import annotations

import math

_VIS = 'contype="0" conaffinity="0" mass="0" group="2"'
_COL = 'group="3"'

# (physical contact attributes, RGB material).  Visual material names never
# enter the contact primitive, so swapping an appearance preset cannot alter a
# physical model field.
_CONCRETE = ('friction="1.0 0.005 0.0001" condim="3"', "mat_concrete")
_LEDGE = ('friction="0.45 0.005 0.0001" condim="3"', "mat_ledge")
_RAIL = ('friction="0.22 0.004 0.0001" condim="3"', "mat_rail")
_PLAZA = ('friction="1.0 0.005 0.0001" condim="3"', "mat_ground")


def _box(name, size, pos, euler=None, style=_CONCRETE) -> str:
    e = f' euler="{euler}"' if euler else ""
    contact, material = style
    out = (f'\n    <geom name="{name}" type="box" size="{size}" pos="{pos}"'
           f'{e} {contact} {_COL}/>'
           f'\n    <geom name="parkvis_{name}" type="box" size="{size}" pos="{pos}"'
           f'{e} {_VIS} material="{material}"/>')
    # A narrow visual-only roll along axis-aligned top edges catches a soft
    # highlight and removes the razor-cut CG silhouette.  Collision keeps the
    # exact authoritative box above. Very thin/large slabs are intentionally
    # excluded so the ground plane and contest flat do not acquire a lip.
    if euler is None:
        sx, sy, sz = (float(value) for value in size.split())
        cx, cy, cz = (float(value) for value in pos.split())
        if sz >= 0.05 and sx <= 4.0 and sy <= 4.0:
            radius = min(0.012, max(0.004, 0.10 * min(sx, sy, sz)))
            z = cz + sz - 0.35 * radius
            x0, x1 = cx - sx + radius, cx + sx - radius
            y0, y1 = cy - sy + radius, cy + sy - radius
            # The two camera-facing edges do the perceptual work. Drawing all
            # four doubled this helper's geometry for no visible benefit.
            edges = (("x_l", x0, y0, z, x1, y0, z),
                     ("y_n", x0, y0, z, x0, y1, z))
            for edge, ax, ay, az, bx, by, bz in edges:
                out += (f'\n    <geom name="parkvis_{name}_soft_{edge}" type="capsule"'
                        f' fromto="{ax:.4f} {ay:.4f} {az:.4f} '
                        f'{bx:.4f} {by:.4f} {bz:.4f}" size="{radius:.4f}"'
                        f' {_VIS} material="{material}"/>')
    return out


def _capsule(name, fromto, radius, style=_RAIL) -> str:
    contact, material = style
    return (f'\n    <geom name="{name}" type="capsule" fromto="{fromto}"'
            f' size="{radius}" {contact} {_COL}/>'
            f'\n    <geom name="parkvis_{name}" type="capsule" fromto="{fromto}"'
            f' size="{radius}" {_VIS} material="{material}"/>')


def _accent_box(name, size, pos, euler=None, *, secondary=False) -> str:
    """A paint/powder-coat cue which can never become a contact surface."""
    e = f' euler="{euler}"' if euler else ""
    material = "mat_accent_secondary" if secondary else "mat_accent"
    return (f'\n    <geom name="fx_{name}" type="box" size="{size}" pos="{pos}"'
            f'{e} {_VIS} material="{material}"/>')


def _ground() -> str:
    """Physical plane plus irregular, subtle RGB-only concrete detail."""
    out = """
    <geom name="ground" type="plane" size="60 60 0.1" pos="0 0 0"
          friction="1.0 0.005 0.0001" condim="3" group="3"/>
    <geom name="parkvis_ground" type="plane" size="60 60 0.1" pos="0 0 0"
          contype="0" conaffinity="0" mass="0" group="2" material="mat_ground"/>"""
    # Expansion joints form varied slabs rather than a repeating checker. They
    # stay just above the rendered plane and can never enter contacts or rays.
    for i, x in enumerate((-5.8, -2.7, 1.05, 4.25, 8.1, 12.7)):
        out += (f'\n    <geom name="fx_ground_joint_cross_{i}" type="box"'
                f' size="0.0045 12 0.00035" pos="{x} 0 0.0007"'
                f' {_VIS} material="mat_ground_joint"/>')
    for i, y in enumerate((-4.1, -1.15, 2.35, 5.8)):
        out += (f'\n    <geom name="fx_ground_joint_long_{i}" type="box"'
                f' size="30 0.0045 0.00035" pos="0 {y} 0.0007"'
                f' {_VIS} material="mat_ground_joint"/>')

    # Broad tonal repairs break up large areas without introducing clutter.
    for i, (x, y, sx, sy) in enumerate((
        (-0.9, 1.25, 0.80, 0.42),
        (2.70, -2.35, 1.15, 0.55),
        (6.20, 1.10, 1.45, 0.66),
    )):
        out += (f'\n    <geom name="fx_ground_patch_{i}" type="box"'
                f' size="{sx} {sy} 0.00025" pos="{x} {y} 0.0010"'
                f' {_VIS} material="mat_ground_patch"/>')

    # Sparse and asymmetric on purpose: these fine marks stop any crop looking
    # computer-perfect, while their low alpha avoids photorealistic clutter.
    for i, (x0, y0, x1, y1) in enumerate((
        (-1.7, -1.2, -1.35, -1.04),
        (1.2, 0.9, 1.65, 0.72),
        (3.6, -0.8, 3.92, -0.52),
        (5.4, 1.4, 5.82, 1.48),
        (7.1, -1.5, 7.48, -1.72),
        (-5.0, 2.2, -4.55, 2.05),
    )):
        out += (f'\n    <geom name="fx_ground_scuff_{i}" type="capsule"'
                f' fromto="{x0} {y0} 0.0015 {x1} {y1} 0.0015" size="0.0025"'
                f' {_VIS} material="mat_scuff"/>')
    return out


def _stair_set(x0: float, y0: float, steps: int = 6, rise: float = 0.16,
               tread: float = 0.30, width: float = 4.0) -> str:
    """A stair set, plus the handrail and hubba ledge flanking it.

    Each step is a box resting on the ground rather than a thin tread, so a
    wheel that clips a riser hits solid geometry instead of falling through a
    gap.
    """
    out = ""
    top = steps * rise
    for i in range(steps):
        h = top - i * rise           # this step's top height
        depth = tread
        cx = x0 + tread * (i + 0.5)
        out += _box(f"step_{i}", f"{depth / 2:.3f} {width / 2:.3f} {h / 2:.3f}",
                    f"{cx:.3f} {y0:.3f} {h / 2:.3f}")
        if i in (0, steps - 1):
            out += _accent_box(
                f"step_lip_{i}", f"0.012 {width / 2:.3f} 0.005",
                f"{x0 + tread * i + 0.010:.3f} {y0:.3f} {h + 0.005:.3f}",
                secondary=(i == steps - 1))
    run = steps * tread
    # Handrail: down the slope, one side of the stairs.
    ry = y0 - width / 2 + 0.45
    out += _capsule("handrail",
                    f"{x0:.3f} {ry:.3f} {top + 0.30:.3f} "
                    f"{x0 + run:.3f} {ry:.3f} 0.30", 0.03)
    # Hubba: a sloped ledge down the other side.
    hy = y0 + width / 2 - 0.30
    slope = math.degrees(math.atan2(top, run))
    out += _box("hubba", f"{math.hypot(run, top) / 2:.3f} 0.28 0.14",
                f"{x0 + run / 2:.3f} {hy:.3f} {top / 2 + 0.10:.3f}",
                euler=f"0 {slope:.2f} 0", style=_LEDGE)
    return out


def _quarter_pipe(x0: float, y0: float, height: float = 1.4,
                  width: float = 6.0, facets: int = 7) -> str:
    """A transition faceted from boxes.

    A true quarter pipe is a cylinder, which MJX will not collide with a box,
    so the curve is approximated by angled slabs. Seven facets over 90 degrees
    is about 13 degrees per facet -- coarse enough to be cheap, fine enough
    that a 27 mm wheel does not catch on the joins.
    """
    out = ""
    r = height
    for i in range(facets):
        a0 = (math.pi / 2) * i / facets
        a1 = (math.pi / 2) * (i + 1) / facets
        # Points on the arc, measured from the top of the transition.
        p0 = (r * math.sin(a0), r - r * math.cos(a0))
        p1 = (r * math.sin(a1), r - r * math.cos(a1))
        cx, cz = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
        seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        pitch = -math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        out += _box(f"qp_{i}", f"{seg / 2:.3f} {width / 2:.3f} 0.10",
                    f"{x0 + cx:.3f} {y0:.3f} {cz:.3f}",
                    euler=f"0 {pitch:.2f} 0")
    # Deck behind the lip, and the coping as a capsule (capsules collide with
    # everything under MJX, unlike a cylinder).
    out += _box("qp_deck", f"1.2 {width / 2:.3f} {height / 2:.3f}",
                f"{x0 - 1.2:.3f} {y0:.3f} {height / 2:.3f}", style=_PLAZA)
    out += _capsule("qp_coping",
                    f"{x0:.3f} {y0 - width / 2:.3f} {height:.3f} "
                    f"{x0:.3f} {y0 + width / 2:.3f} {height:.3f}", 0.035)
    return out


def _funbox(x0: float, y0: float, length: float = 3.0, width: float = 2.4,
            height: float = 0.42) -> str:
    """A flat-top box with a bank up each end."""
    out = _box("funbox_top", f"{length / 2:.3f} {width / 2:.3f} {height / 2:.3f}",
               f"{x0:.3f} {y0:.3f} {height / 2:.3f}", style=_LEDGE)
    out += _accent_box("funbox_edge", f"{length / 2:.3f} 0.018 0.006",
                       f"{x0:.3f} {y0 - width / 2 + 0.018:.3f} {height + 0.006:.3f}")
    run = 1.5
    slope = math.degrees(math.atan2(height, run))
    for name, sgn in (("funbox_bank_a", -1), ("funbox_bank_b", 1)):
        out += _box(name, f"{math.hypot(run, height) / 2:.3f} {width / 2:.3f} 0.06",
                    f"{x0 + sgn * (length / 2 + run / 2):.3f} {y0:.3f} "
                    f"{height / 2:.3f}",
                    euler=f"0 {-sgn * slope:.2f} 0")
    return out


def sls_park() -> str:
    """A contest-style course: flat run, stairs + rail + hubba, funbox, QP."""
    park = _ground()
    # The contest flat is neutral concrete.  Restrained paint on obstacle lips
    # carries the accent; the ground itself remains useful training signal.
    park += _box("contest_flat", "9.0 5.0 0.02", "0 0 0.01",
                 style=_PLAZA)
    park += _stair_set(x0=9.0, y0=0.0)
    park += _funbox(x0=-6.0, y0=2.5)
    park += _capsule("flat_bar", "-9.0 -3.0 0.30 -4.0 -3.0 0.30", 0.03)
    park += _box("manual_pad", "1.6 0.9 0.075", "-1.5 -3.6 0.075", style=_LEDGE)
    park += _accent_box("manual_pad_edge", "1.6 0.016 0.006",
                        "-1.5 -4.484 0.156", secondary=True)
    park += _quarter_pipe(x0=-13.0, y0=0.0)
    return park


def plaza_park() -> str:
    """Compact, unbranded modules composed for the fitted chase camera.

    The older SLS course is spatially broad: from its reset anchor the fitted
    phone camera sees floor, while its nearest forward obstacle lies near the
    horizon.  This compact course changes the *park*, not the camera, placing a
    low ledge, rail, bank and ascending stair set in the useful forward field.
    Every visible module is still paired with authoritative collision geometry
    by the helpers above.
    """
    park = _ground()

    # Low ledge on camera-left, including one muted powder-coated edge.
    park += _box("plaza_ledge", "0.72 0.36 0.12", "1.62 0.36 0.12",
                 style=_LEDGE)
    park += _accent_box("plaza_ledge_edge", "0.72 0.014 0.005",
                        "1.62 0.014 0.245")

    # Flat bar on camera-right, with physical capsule uprights.  Capsules are
    # both more legible in RGB and portable to MJX contacts.
    park += _capsule("plaza_rail", "1.05 -0.30 0.27 2.18 -0.30 0.27", 0.024)
    park += _capsule("plaza_rail_post_a", "1.17 -0.30 0.03 1.17 -0.30 0.27", 0.020)
    park += _capsule("plaza_rail_post_b", "2.06 -0.30 0.03 2.06 -0.30 0.27", 0.020)

    # A bank beyond the first two modules.  One box is both a predictable MJX
    # contact surface and a clean, modular silhouette.
    bank_run, bank_height = 1.25, 0.42
    bank_slope = math.degrees(math.atan2(bank_height, bank_run))
    park += _box("plaza_bank", f"{math.hypot(bank_run, bank_height) / 2:.3f} 0.62 0.055",
                 "2.75 0.88 0.210", euler=f"0 {-bank_slope:.2f} 0")
    park += _accent_box("plaza_bank_lip", "0.018 0.62 0.006",
                        "3.355 0.88 0.426", secondary=True)

    # Three ascending steps on the opposite side.  Unlike the legacy SLS set,
    # the low riser faces the reset anchor, so it reads as stairs rather than a
    # one-metre wall in the chase view.
    x0, y0, tread, rise, width = 2.45, -1.38, 0.28, 0.14, 1.20
    for i in range(3):
        height = (i + 1) * rise
        cx = x0 + (i + 0.5) * tread
        park += _box(f"plaza_step_{i}", f"{tread / 2:.3f} {width / 2:.3f} {height / 2:.3f}",
                     f"{cx:.3f} {y0:.3f} {height / 2:.3f}")
    park += _accent_box("plaza_step_lip", f"0.012 {width / 2:.3f} 0.005",
                        f"{x0 + 0.010:.3f} {y0:.3f} {rise + 0.005:.3f}")
    return park


# Ground only. The default for fitting work, where the board is reset to a flat
# anchor and never reaches an obstacle.
FLAT_PARK = _ground()

PARKS = {"flat": FLAT_PARK, "plaza": plaza_park(), "sls": sls_park()}
