"""Behavioural guards on the board model.

These assert emergent behaviour, never scripted behaviour: there is no ollie
code and no carving code, so if the truck geometry or contact parameters
regress, these are what notice.
"""
import numpy as np

from opensk.sim.core import SkateSim
from opensk.sim.model.build import ride_height
from opensk.sim.params import SkateParams


def test_rests_level_on_four_wheels():
    sim = SkateSim()
    sim.reset(seed=0)
    st = sim.step(1500)
    assert st.wheel_contact.all()
    assert not st.deck_contact
    assert abs(st.pos[2] - ride_height(sim.params)) < 2e-3
    assert abs(st.quat[0]) > 0.999  # still level


def _carve(roll_torque: float, seconds: float = 2.0) -> float:
    """Lateral displacement after leaning a rolling board. Returns metres."""
    sim = SkateSim()
    sim.reset(seed=0, speed=3.0)
    bid = sim.deck_bid
    for _ in range(int(seconds / sim.params.timestep)):
        sim.data.xfrc_applied[bid, 3] = roll_torque
        sim.step()
    return float(sim.state().pos[1])


def test_lean_produces_carve_and_is_symmetric():
    """A lean must turn the board, via truck steer alone -- no yaw is applied.

    This is the single most important emergent property of the model: it is
    what makes the kingpin angle and bushing stiffness meaningful parameters
    for system identification to fit.
    """
    left, right = _carve(+1.5), _carve(-1.5)
    assert left < -0.5, f"lean did not carve: y={left}"
    assert right > 0.5, f"opposite lean did not carve: y={right}"
    assert abs(left + right) < 0.05 * abs(left), "carving is asymmetric"


def test_rolls_straight_when_level():
    sim = SkateSim()
    sim.reset(seed=0, speed=3.0)
    st = sim.step(1000)
    assert abs(st.pos[1]) < 1e-3
    assert st.pos[0] > 2.5


def test_tail_press_pops_the_board():
    """Pressing the tail must lift the deck off its ride height.

    No ollie code exists: the tail strikes the ground and the contact impulse
    does the rest. If contact params or the kicktail geometry regress, the
    pop dies and this catches it.
    """
    sim = SkateSim()
    sim.reset(seed=0, speed=2.0)
    peak, struck = 0.0, False
    for i in range(700):
        if i < 110:
            sim.apply_force([0, 0, -90], sim.body_point([-0.36, 0, 0.006]))
        st = sim.step()
        struck |= st.deck_contact
        peak = max(peak, st.pos[2])
    assert struck, "tail never reached the ground"
    assert peak > ride_height(sim.params) + 0.08, f"no pop: peak {peak:.3f} m"


def test_screen_gesture_ollies_the_board():
    """End-to-end: a screen-space gesture must be able to pop the board.

    This is the transfer pipeline in one assertion -- gesture path sampling,
    camera ray, deck grab, spring-damper force, tail strike, pop. Nothing in
    it knows what an ollie is. It also guards the touch defaults: below
    roughly touch_gain * touch_slip_distance = 100 N the tail cannot reach the
    ground, no gesture can ollie, and system identification would be starting
    from a flat region of the objective.
    """
    from opensk.sim.touch import TouchModel

    sim = SkateSim()
    sim.reset(seed=0)
    touch = TouchModel(sim)
    # Starts on the tail (screen y 0.515 lands on the deck's rear end once the
    # push has rolled it forward) and drags down-screen, which presses the
    # tail into the ground.
    recipe = {"gestures": [{"points": [[0.5, 0.515], [0.5, 0.595], [0.5, 0.675]],
                            "duration": 0.13, "easing_power": 2.2}],
              "delays": []}
    traj = touch.run(recipe, push=True, settle=0.9)

    assert any(s.deck_contact for s in traj), "tail never struck the ground"
    peak = max(s.pos[2] for s in traj)
    assert peak > ride_height(sim.params) + 0.10, f"no pop: {peak:.3f} m"
    assert any(s.airborne for s in traj), "board never left the ground"


def test_push_reaches_a_plausible_speed():
    """The standard push must roll the board, not launch it.

    An earlier speed-proportional push model hit 1032 N and 7.3 m/s on the
    device's 0.02 s swipe, which carried the board out from under every
    subsequent gesture. Kept as a guard on the impulse formulation.
    """
    from opensk.sim.touch import TouchModel

    sim = SkateSim()
    sim.reset(seed=0)
    TouchModel(sim)._run_push()
    speed = float(np.linalg.norm(sim.state().linvel[:2]))
    assert 0.5 < speed < 4.0, f"implausible push speed {speed:.2f} m/s"
