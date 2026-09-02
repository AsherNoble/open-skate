"""Park geometry must be skateable and MJX-portable.

The MJX rules matter more here than anywhere else in the model: a park is what
the wheels roll on, so a cylinder coping or an ellipsoid transition would work
perfectly on CPU and drop the board through the floor on GPU.
"""
import mujoco
import numpy as np
import pytest

from opensk.sim.core import SkateSim
from opensk.sim.model.build import build_scene
from opensk.sim.model.parks import PARKS
from opensk.sim.params import SkateParams

BANNED = {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID}


@pytest.mark.parametrize("name", sorted(PARKS))
def test_park_is_mjx_safe(name):
    """Every COLLIDABLE geom must be a type MJX can collide with a box."""
    m = mujoco.MjModel.from_xml_string(build_scene(SkateParams(), PARKS[name]))
    bad = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
           for i in range(m.ngeom)
           if m.geom_type[i] in BANNED
           and (m.geom_contype[i] or m.geom_conaffinity[i])]
    assert not bad, f"{name}: MJX cannot collide these reliably: {bad}"
    assert set(m.geom_condim) <= {1, 3, 4, 6}


def test_board_rests_level_on_the_park():
    sim = SkateSim(SkateParams(), PARKS["sls"])
    sim.reset(seed=0)
    st = sim.step(1200)
    assert st.wheel_contact.all()
    assert abs(st.quat[0]) > 0.99, "board did not settle level on the flat"


@pytest.mark.parametrize("x,y,heading,speed,label", [
    (4.0, 0.0, 0.0, 4.0, "stairs"),
    (-9.0, 0.0, np.pi, 5.0, "quarter pipe"),
    (-11.0, -3.0, 0.0, 3.0, "flat bar"),
    (-4.0, 2.5, np.pi, 3.5, "funbox"),
])
def test_board_does_not_fall_through_obstacles(x, y, heading, speed, label):
    """Rolling at each obstacle must keep the board above the floor.

    Faceted transitions and stacked stair boxes are exactly where a contact
    gap would open up, and a board that drops through the world is silent
    otherwise -- the simulation happily keeps running.
    """
    sim = SkateSim(SkateParams(), PARKS["sls"])
    sim.reset(seed=0, pos=(x, y), heading=heading, speed=speed)
    lowest = 9.0
    for _ in range(int(3.0 / sim.params.timestep)):
        lowest = min(lowest, sim.step().pos[2])
    assert lowest > -0.15, f"{label}: board fell through (z={lowest:.3f})"
