"""The fitted parameter vector for Open Skate.

Everything that system identification is allowed to change lives here, and
nowhere else. `SkateParams` is the single source of truth: the MJCF builder
reads it, the touch model reads it, the camera reads it, and `fit/sysid.py`
optimises over exactly the fields listed in `FIT_SPEC`.

Units are SI throughout (metres, kilograms, seconds, radians).

Defaults are physically plausible starting values measured off a real
skateboard, NOT fitted values. They exist so the sim runs before Phase 3;
they are the initial mean of the CMA-ES search, not an answer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class SkateParams:
    # --- deck geometry (a 32" x 8.25" popsicle deck) -----------------------
    deck_length: float = 0.813
    deck_width: float = 0.196
    deck_thickness: float = 0.012
    deck_mass: float = 1.40
    # Wheelbase: axle-to-axle. Real decks run 0.33-0.38 m.
    wheelbase: float = 0.360
    # Kicktails. The nose and tail angle up off the flat, and the tail angle
    # is what sets how far the deck must pitch before it strikes the ground —
    # i.e. it governs the pop directly. Measured, not fitted.
    kick_angle_deg: float = 19.0
    flat_fraction: float = 0.60
    # Plan-view shape, fitted to the real deck's silhouette rather than
    # guessed: 0.196 m wide (a 7.7" deck) where 0.210 was assumed, giving a
    # rendered width/length profile matching the measured one to RMS 0.041.
    # Plan-view taper. A popsicle deck is widest at the middle and narrows
    # toward both tips; measured from real frames the width profile rises from
    # ~0.22 to a peak ~0.40 and falls to ~0.19 (width/length), where a
    # constant-width outline renders almost flat. `deck_tip_width_frac` is the
    # width at the very tip as a fraction of the centre width, and
    # `deck_taper_power` how sharply it narrows. Visual only -- collision stays
    # the MJX-safe boxes.
    deck_tip_width_frac: float = 0.50
    deck_taper_power: float = 1.6

    # --- trucks ------------------------------------------------------------
    truck_mass: float = 0.350
    # Kingpin angle measured from the deck plane. ~50 deg is standard; this
    # is what converts deck roll into wheel steer, so it dominates carving.
    kingpin_angle_deg: float = 50.0
    # Bushing stiffness/damping about the steering axis. The single most
    # important pair for how the board feels: too stiff and it will not
    # carve, too soft and it wobbles.
    truck_stiffness: float = 30.0
    truck_damping: float = 0.9
    # Mechanical stop on the steering hinge (bushings bottoming out).
    truck_limit_deg: float = 35.0

    # --- wheels ------------------------------------------------------------
    # SPHERE geoms, not cylinders: MJX-JAX cannot collide a cylinder with a
    # box or a mesh, and every park surface is a box. See the plan.
    wheel_radius: float = 0.027
    wheel_mass: float = 0.055
    # Axle half-separation; wheels sit at +/- this in the deck's y axis.
    axle_halfwidth: float = 0.105
    wheel_friction_slide: float = 1.30
    wheel_friction_spin: float = 0.006
    wheel_friction_roll: float = 0.0004
    # Bearing drag, as joint friction loss on the wheel hinge.
    wheel_frictionloss: float = 0.0012

    # --- deck contact (the ends strike the ground; this is the pop) --------
    deck_friction_slide: float = 0.55
    # solref/solimp govern MuJoCo's soft contact. The tail-strike impulse is
    # exactly what sets ollie height, so these are prime fit targets.
    contact_solref_time: float = 0.010
    contact_solref_damp: float = 1.00
    contact_solimp_dmin: float = 0.92
    contact_solimp_dmax: float = 0.98
    contact_solimp_width: float = 0.001

    # --- touch model (screen gesture -> force on the deck) -----------------
    # A finger is a capped spring-damper between the fingertip target and the
    # material point on the deck it grabbed.
    # These three are the initial mean of the CMA-ES search, so they need to
    # sit in a basin where the board actually behaves: below roughly
    # gain*slip_distance = 100 N the tail never reaches the ground and no
    # gesture can ollie, which would leave sysid optimising over a flat region.
    # Coulomb limit on the finger's grip: the contact point sticks to the deck
    # while the tangential pull stays under touch_friction times the normal
    # press, and slips toward the fingertip beyond it. This is what lets one
    # model do both tricks -- a tail press is mostly normal force, so it
    # sticks and pops; a flick is mostly tangential, so it slips to the rail
    # and rolls the board.
    touch_friction: float = 1.8
    touch_gain: float = 600.0
    touch_damping: float = 12.0
    touch_force_max: float = 250.0
    # Retained for reporting only; no longer gates release. See sim/touch.py:
    # a hard slip release left the finger disengaged for 76% of the median
    # real gesture, and only 0.1% of the 1391 real gestures are short enough
    # to complete without tripping it.
    touch_slip_distance: float = 0.16

    # --- camera (part of the physics model: it defines where a touch lands) -
    # Fitted to the real game: 128 resting-board frames, matching silhouette
    # length, width and cy to under one percent. Taper is NOT matched (0.82 vs
    # 0.69) and a faithful popsicle outline did not fix it -- the real mask is
    # a dark-pixel threshold that swallows the board's SHADOW near the lower
    # end, widening the near third, while our rendered mask is exact and
    # shadowless. Taper is a segmentation artefact here, not a camera cue; it
    # stays in the objective only because its own MAD down-weights it ~15x.
    # See pose/calibrate_camera.py.
    cam_fov_deg: float = 40.182
    # The camera AIMS AT the board, so its height is not free: it follows from
    # distance and pitch. Carrying an independent height as well would give
    # three parameters for two degrees of freedom and leave sysid a flat
    # direction to wander along.
    cam_distance: float = 2.560
    cam_pitch_deg: float = -35.552
    # The camera aims this far AHEAD of the board along its heading, which is
    # what pushes the board below screen centre (measured at cy = 0.582, not
    # 0.5) so the rider can see what is coming. Identifiable, unlike an
    # independent camera height.
    cam_lead_m: float = 0.283
    # First-order follow lag, seconds. 0 = rigidly locked to the board.
    cam_follow_tau: float = 0.18
    # Screen aspect (width/height). All three rig devices are 19.5:9 to within
    # 0.05%, so this is a constant, not a per-device value. See GESTURES.md.
    screen_aspect: float = 375.0 / 812.0

    # --- push --------------------------------------------------------------
    # A drag landing on the ground rather than the deck is a push. Modelled as
    # an IMPULSE proportional to how far the finger travels across the screen,
    # spread over the gesture -- not as a force proportional to drag speed.
    # Speed-proportional blows up: PUSH_DURATION is 0.02 s on the device, so a
    # standard push covers 0.37 screen units at ~19 units/s and any sane gain
    # launches the board. Distance is also stable under the millisecond
    # quantisation of segment durations. N*s per unit of normalised screen travel.
    push_impulse_gain: float = 12.5

    # --- spin button -------------------------------------------------------
    # True Skate's rotate button, which curved drags cannot express. Held by a
    # second finger; modelled as a yaw torque about the deck normal.
    spin_torque: float = 1.6

    # --- integrator --------------------------------------------------------
    # 500 Hz, comfortably above True Skate's 120 Hz, so contact resolution is
    # not the thing that differs between us and it.
    timestep: float = 0.002
    gravity: float = 9.81

    def replace(self, **kw) -> "SkateParams":
        return replace(self, **kw)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str) -> "SkateParams":
        with open(path) as fh:
            return cls(**json.load(fh))


# Fields sysid may vary, with (low, high) bounds. Geometry we can measure with
# a ruler is deliberately excluded — fitting it would let the optimiser buy
# trajectory error down by building a board that does not exist.
FIT_SPEC: dict[str, tuple[float, float]] = {
    "deck_mass":             (0.9, 2.2),
    "wheelbase":             (0.32, 0.40),
    "truck_mass":            (0.20, 0.55),
    "kingpin_angle_deg":     (38.0, 63.0),
    "truck_stiffness":       (5.0, 120.0),
    "truck_damping":         (0.05, 6.0),
    "truck_limit_deg":       (20.0, 50.0),
    "wheel_mass":            (0.02, 0.12),
    "wheel_friction_slide":  (0.4, 2.5),
    "wheel_friction_spin":   (0.0005, 0.05),
    "wheel_frictionloss":    (0.0, 0.02),
    "deck_friction_slide":   (0.15, 1.5),
    "contact_solref_time":   (0.002, 0.05),
    "contact_solref_damp":   (0.3, 2.5),
    "contact_solimp_dmin":   (0.60, 0.97),
    "contact_solimp_dmax":   (0.90, 0.999),
    "touch_gain":            (30.0, 900.0),
    "touch_friction":        (0.15, 4.0),
    "touch_damping":         (0.5, 60.0),
    "touch_force_max":       (15.0, 400.0),
    "cam_fov_deg":           (25.0, 70.0),
    "cam_distance":          (0.8, 5.0),
    "cam_pitch_deg":         (-85.0, -10.0),
    "cam_lead_m":            (-0.5, 2.0),
    "cam_follow_tau":        (0.0, 0.6),
    "spin_torque":           (0.1, 12.0),
    "push_impulse_gain":     (1.0, 60.0),
}

FIT_KEYS: tuple[str, ...] = tuple(FIT_SPEC)

_ALL = {f.name for f in fields(SkateParams)}
assert set(FIT_SPEC) <= _ALL, f"FIT_SPEC names not on SkateParams: {set(FIT_SPEC) - _ALL}"


def to_vector(p: SkateParams) -> list[float]:
    """SkateParams -> the flat vector CMA-ES searches over."""
    return [getattr(p, k) for k in FIT_KEYS]


def from_vector(v, base: SkateParams | None = None) -> SkateParams:
    """Flat vector -> SkateParams, clamped to FIT_SPEC bounds.

    Clamping (rather than rejecting) keeps a CMA-ES sample that strays out of
    bounds usable, which matters because the optimiser proposes freely and a
    rejected sample still costs a generation slot.
    """
    base = base or SkateParams()
    out = {}
    for k, x in zip(FIT_KEYS, v):
        lo, hi = FIT_SPEC[k]
        x = float(x)
        if x != x:  # NaN
            x = 0.5 * (lo + hi)
        out[k] = min(hi, max(lo, x))
    return base.replace(**out)
