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
treads, a 0.30 m flat bar, a 1.4 m quarter pipe.
"""
from __future__ import annotations

import math

# Ground only. The default for fitting work, where the board is reset to a flat
# anchor and never reaches an obstacle.
FLAT_PARK = """
    <geom name="ground" type="plane" size="60 60 0.1" pos="0 0 0"
          friction="1.0 0.005 0.0001" condim="3" material="mat_ground"/>"""

_CONCRETE = 'friction="1.0 0.005 0.0001" condim="3" material="mat_concrete"'
_LEDGE = 'friction="0.45 0.005 0.0001" condim="3" material="mat_ledge"'
_RAIL = 'friction="0.22 0.004 0.0001" condim="3" material="mat_rail"'
_DECK_YELLOW = 'friction="1.0 0.005 0.0001" condim="3" material="mat_plaza"'


def _box(name, size, pos, euler=None, style=_CONCRETE) -> str:
    e = f' euler="{euler}"' if euler else ""
    return (f'\n    <geom name="{name}" type="box" size="{size}" pos="{pos}"'
            f'{e} {style}/>')


def _capsule(name, fromto, radius, style=_RAIL) -> str:
    return (f'\n    <geom name="{name}" type="capsule" fromto="{fromto}"'
            f' size="{radius}" {style}/>')


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
                f"{x0 - 1.2:.3f} {y0:.3f} {height / 2:.3f}", style=_DECK_YELLOW)
    out += _capsule("qp_coping",
                    f"{x0:.3f} {y0 - width / 2:.3f} {height:.3f} "
                    f"{x0:.3f} {y0 + width / 2:.3f} {height:.3f}", 0.035)
    return out


def _funbox(x0: float, y0: float, length: float = 3.0, width: float = 2.4,
            height: float = 0.42) -> str:
    """A flat-top box with a bank up each end."""
    out = _box("funbox_top", f"{length / 2:.3f} {width / 2:.3f} {height / 2:.3f}",
               f"{x0:.3f} {y0:.3f} {height / 2:.3f}", style=_LEDGE)
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
    park = """
    <geom name="ground" type="plane" size="60 60 0.1" pos="0 0 0"
          friction="1.0 0.005 0.0001" condim="3" material="mat_ground"/>"""
    # The yellow contest flat the board is reset onto, raised a hair so it
    # reads as a surface rather than z-fighting with the ground plane.
    park += _box("contest_flat", "9.0 5.0 0.02", "0 0 0.01",
                 style=_DECK_YELLOW)
    park += _stair_set(x0=9.0, y0=0.0)
    park += _funbox(x0=-6.0, y0=2.5)
    park += _capsule("flat_bar", "-9.0 -3.0 0.30 -4.0 -3.0 0.30", 0.03)
    park += _box("manual_pad", "1.6 0.9 0.075", "-1.5 -3.6 0.075", style=_LEDGE)
    park += _quarter_pipe(x0=-13.0, y0=0.0)
    return park


PARKS = {"flat": FLAT_PARK, "sls": sls_park()}
