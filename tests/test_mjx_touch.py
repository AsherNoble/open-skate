"""The ported touch model must produce the CPU model's forces.

`sim/touch.py` is the reference: its stick-slip Coulomb contact, its
mid-gesture re-acquisition and its material-point grab were each arrived at by
measurement against real captures, and every one of them changed how the board
behaves. A port that merely resembles it would be a different simulator, and
the fitted parameters would not mean the same thing.
"""
import numpy as np

from opensk.mjx.touch import (KIND_DECK, _quat_from_mat, finger_force,
                              initial_finger)
from opensk.sim.core import SkateSim
from opensk.sim.gesture_spec import GesturePath
from opensk.sim.params import SkateParams
from opensk.sim.touch import Finger, TouchModel


def _tail_press_path():
    """A tail press that engages the deck, so the comparison exercises contact."""
    return GesturePath(np.array([[0.5, 0.62], [0.5, 0.70], [0.5, 0.78]]), 0.13, 2.2)


def test_quat_from_matrix_is_exact():
    """Branchless reconstruction -- the usual four-case form cannot be traced."""
    sim = SkateSim(SkateParams())
    sim.reset(seed=0)
    sim.step(200)
    R = sim.data.xmat[sim.deck_bid].reshape(3, 3)
    got = np.asarray(_quat_from_mat(R))
    ref = np.asarray(sim.data.qpos[3:7])
    assert np.abs(np.abs(got) - np.abs(ref)).max() < 1e-9


def test_forces_match_the_cpu_touch_model():
    p = SkateParams()
    sim = SkateSim(p)
    sim.reset(seed=0)
    sim.step(200)
    gids = sorted(sim._deck_gids)
    tm = TouchModel(sim)
    path = _tail_press_path()
    cpu_finger = Finger(path, 0.0)
    state = initial_finger()

    worst = 0.0
    engaged = 0
    for i in range(8):
        t = i * 0.018
        nx, ny = path.position_at(t)
        before = sim.data.xfrc_applied[sim.deck_bid].copy()
        tm._apply_finger(cpu_finger, t, 0.002)
        cpu_force = (sim.data.xfrc_applied[sim.deck_bid, :3] - before[:3]).copy()
        sim.data.xfrc_applied[:] = 0.0

        force, _, state = finger_force(state, nx, ny, sim.model, sim.data,
                                       sim.deck_bid, gids, p, 0.002)
        worst = max(worst, float(np.abs(cpu_force - np.asarray(force)).max()))
        engaged += int(state.kind == KIND_DECK)

    assert engaged >= 6, "gesture never engaged the deck; the test proves nothing"
    assert worst < 1e-5, f"ported forces differ from CPU by {worst:.2e} N"
