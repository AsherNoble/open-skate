"""Guard MJX (GPU) portability.

MJX-JAX implements a subset of MuJoCo. Every violation below still runs
perfectly on CPU and silently fails to port, which is the worst possible
failure mode: it would surface as wrong physics on the GPU run, months after
the model changed. So it is a test, not a comment.
"""
import mujoco
import pytest

from opensk.sim.model.build import FLAT_PARK, build_scene
from opensk.sim.params import SkateParams

BANNED = {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID}


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_scene(SkateParams(), FLAT_PARK))


def test_no_cylinder_or_ellipsoid_geoms(model):
    """MJX-JAX cannot collide these with a box or mesh, and parks are boxes."""
    bad = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_type[i] in BANNED
    ]
    assert not bad, f"MJX cannot collide these reliably: {bad}"


def test_no_collidable_meshes(model):
    """Visual meshes are fine, but they must not participate in contact."""
    bad = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert not bad, f"collidable meshes break MJX portability: {bad}"


def test_solver_and_integrator(model):
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL


def test_condim_supported(model):
    """MJX-JAX supports condim in {1, 3, 4, 6}; we standardise on 3."""
    assert set(model.geom_condim) <= {1, 3, 4, 6}
