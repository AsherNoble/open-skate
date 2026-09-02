"""Gesture sampling, replicating TrueSkate-AI's on-device execution exactly.

This is the transfer interface. A gesture recipe here must trace the same
screen path, at the same speed, as the same recipe does through Appium's W3C
Actions on the phone — otherwise a solution found in Open Skate means nothing
on the device, and every fitted physics parameter has absorbed the difference.

Two details are replicated deliberately, and both contradict the prose in
GESTURES.md ("velocity proportional to t^easing_power"), which does not
describe what the code does:

  1. `easing_power` maps progress -> NORMALISED TIME, and is evaluated only at
     segment boundaries (`easing_to_segment_durations` in touch_actions.py).
     It redistributes time *between* segments; it does not shape velocity
     *within* one. Each segment is traversed at constant speed.
  2. Per-segment durations are truncated to whole milliseconds and so
     generally do NOT sum to `duration`. The device sees the quantised total,
     so the sim must too.

Cross-checked against the real source by tests/test_gesture_parity.py.
"""
from __future__ import annotations

import numpy as np

# Mirrored from TrueSkate-AI/src/trueskate_ai/sim/gestures.py. The parity test
# reads that file and fails if these drift.
X_BOUND_MIN, X_BOUND_MAX = 0.12, 1.0
Y_BOUND_MIN, Y_BOUND_MAX = 0.12, 0.88
DEFAULT_SPIN_BUTTON_XY = (0.0604, 0.4040)

PUSH_PRE_DELAY = 0.5
PUSH_DURATION = 0.02
PUSH_EASING = 2.0
PUSH_START = (0.7658, 0.3044)
PUSH_END = (0.7658, 0.6797)
RESET_BUTTON_XY = (0.5, 0.0558)


def segment_durations_ms(n_segments: int, total_ms: int,
                         easing_power: float) -> list[int]:
    """Per-segment durations in whole ms — `easing_to_segment_durations`."""
    if easing_power == 1.0:
        return [max(1, total_ms // n_segments)] * n_segments
    b = [(i / n_segments) ** easing_power for i in range(n_segments + 1)]
    raw = [b[i + 1] - b[i] for i in range(n_segments)]
    s = sum(raw)
    return [max(1, int(d / s * total_ms)) for d in raw]


class GesturePath:
    """A single touch path: waypoints, duration, easing. Screen space."""

    def __init__(self, points, duration: float, easing_power: float = 1.0):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError("a gesture needs >= 2 [x, y] waypoints")
        self.points = pts
        self.easing_power = float(easing_power)
        self.nominal_duration = float(duration)
        n = len(pts) - 1
        self.seg_ms = segment_durations_ms(n, int(duration * 1000), easing_power)
        self.seg_t = np.concatenate([[0.0], np.cumsum(self.seg_ms) / 1000.0])
        # The executed length, after millisecond truncation. Not `duration`.
        self.duration = float(self.seg_t[-1])

    def position_at(self, t: float) -> np.ndarray:
        """Normalised screen position at time `t` seconds after touch-down."""
        if t <= 0.0:
            return self.points[0].copy()
        if t >= self.duration:
            return self.points[-1].copy()
        i = int(np.searchsorted(self.seg_t, t, side="right") - 1)
        i = min(i, len(self.points) - 2)
        span = self.seg_t[i + 1] - self.seg_t[i]
        u = 0.0 if span <= 0 else (t - self.seg_t[i]) / span
        return self.points[i] + u * (self.points[i + 1] - self.points[i])

    def __repr__(self) -> str:
        return (f"GesturePath({len(self.points)}pts, {self.duration:.3f}s, "
                f"ease={self.easing_power:.2f})")


def push_path() -> GesturePath:
    """The standard board push, from the shared PUSH_* constants."""
    return GesturePath([PUSH_START, PUSH_END], PUSH_DURATION, PUSH_EASING)


def schedule_recipe(recipe: dict) -> list[tuple[float, GesturePath]]:
    """Recipe -> [(start_time, path)], using the delay chain.

    `delays[i]` is the gap between the END of gesture i and the start of
    gesture i+1, and may be negative (overlapping fingers), which the device
    handles by bundling both into one W3C payload.
    """
    gestures = recipe["gestures"]
    delays = list(recipe.get("delays") or [])
    if len(delays) < len(gestures) - 1:
        delays += [0.0] * (len(gestures) - 1 - len(delays))
    out, t = [], 0.0
    for i, g in enumerate(gestures):
        path = GesturePath(g["points"], g["duration"], g.get("easing_power", 1.0))
        out.append((t, path))
        if i < len(gestures) - 1:
            t += path.duration + float(delays[i])
    return out


def spin_window(recipe: dict, total: float) -> tuple[float, float] | None:
    """Absolute (start, end) seconds the rotate button is held, if at all.

    Recipe spin blocks store t_start/t_end as fractions of the schedule's
    total duration; SLS `spin_flick` samples store absolute seconds. Both
    shapes appear in the corpus, so both are accepted here.
    """
    s = recipe.get("spin")
    if not s or not s.get("enabled", True):
        return None
    if "spin_hold_start_s" in s:
        return float(s["spin_hold_start_s"]), float(s["spin_hold_end_s"])
    return float(s["t_start"]) * total, float(s["t_end"]) * total
