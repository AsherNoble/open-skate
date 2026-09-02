"""Open Skate's gesture sampling must match the device pipeline exactly.

Everything in this project rests on one claim: a recipe executed here does
what the same recipe does on the phone. These tests check that claim against
TrueSkate-AI's actual source rather than against its documentation — the two
disagree about `easing_power`, and the source is what runs.

Skipped (not failed) when the sibling repo is absent, so Open Skate stays
testable standalone.
"""
import json
import pathlib
import re
import subprocess

import pytest

from opensk.sim import gesture_spec as gs

RIG = pathlib.Path("/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI")
RIG_PY = RIG / ".venv/bin/python"

pytestmark = pytest.mark.skipif(
    not (RIG / "src/trueskate_ai/sim/gestures.py").exists(),
    reason="TrueSkate-AI checkout not present",
)


def test_shared_constants_have_not_drifted():
    """These are mirrored, not imported (the rig module pulls in Appium).

    Mirroring is only safe if drift is loud, which is what this is for.
    """
    src = (RIG / "src/trueskate_ai/sim/gestures.py").read_text()

    def const(name):
        m = re.search(rf"^{name}: float = ([\d.]+)", src, re.M)
        return float(m.group(1))

    def pair(name):
        m = re.search(rf"^{name}: tuple\[float, float\] = \(\s*([\d.]+),\s*([\d.]+),?\s*\)",
                      src, re.M)
        return (float(m.group(1)), float(m.group(2)))

    assert gs.X_BOUND_MIN == const("X_BOUND_MIN")
    assert gs.X_BOUND_MAX == const("X_BOUND_MAX")
    assert gs.Y_BOUND_MIN == const("Y_BOUND_MIN")
    assert gs.Y_BOUND_MAX == const("Y_BOUND_MAX")
    assert gs.PUSH_DURATION == const("PUSH_DURATION")
    assert gs.PUSH_EASING == const("PUSH_EASING")
    assert gs.PUSH_PRE_DELAY == const("PUSH_PRE_DELAY")
    assert gs.PUSH_START == pair("PUSH_START")
    assert gs.PUSH_END == pair("PUSH_END")
    assert gs.DEFAULT_SPIN_BUTTON_XY == pair("DEFAULT_SPIN_BUTTON_XY")


CASES = [(2, 350, 1.0), (2, 350, 1.8), (2, 350, 0.4), (3, 500, 2.0),
         (2, 30, 3.0), (4, 800, 0.3), (2, 120, 1.0), (3, 47, 2.7)]


@pytest.mark.skipif(not RIG_PY.exists(), reason="rig venv not present")
def test_segment_durations_match_the_device_implementation():
    """Byte-for-byte agreement with `easing_to_segment_durations`.

    Run in the rig's own venv because touch_actions imports Selenium at module
    scope; reimplementing it here to avoid that would defeat the purpose.
    """
    prog = (
        "import sys, json;"
        f"sys.path.insert(0, {str(RIG / 'src')!r});"
        "from trueskate_ai.sim.touch_actions import easing_to_segment_durations as f;"
        f"cases = {CASES!r};"
        # Mirror curved_drag's own dispatch: gesture_recipe passes easing=None
        # when easing_power == 1.0, and curved_drag then splits evenly rather
        # than calling easing_to_segment_durations at all.
        "print(json.dumps([([max(1, ms // n)] * n) if p == 1.0"
        " else f(n, ms, (lambda t, p=p: t ** p)) for n, ms, p in cases]))"
    )
    out = subprocess.run([str(RIG_PY), "-c", prog], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    for (n, ms, p), expected in zip(CASES, json.loads(out.stdout)):
        assert gs.segment_durations_ms(n, ms, p) == expected, (n, ms, p)


def test_executed_duration_is_millisecond_quantised():
    """The device runs the truncated total, not the requested one."""
    g = gs.GesturePath([[0.5, 0.6], [0.45, 0.5], [0.6, 0.3]], 0.350, 1.8)
    assert g.duration != g.nominal_duration
    assert g.duration == sum(g.seg_ms) / 1000.0


def test_path_endpoints_and_monotonic_progress():
    g = gs.GesturePath([[0.2, 0.8], [0.5, 0.5], [0.9, 0.15]], 0.4, 2.0)
    assert g.position_at(-1.0).tolist() == [0.2, 0.8]
    assert g.position_at(99.0).tolist() == [0.9, 0.15]
    # Waypoints are hit exactly at segment boundaries.
    assert g.position_at(g.seg_t[1]).tolist() == pytest.approx([0.5, 0.5])
